"""把各元件串起來：頻道對話紀錄、回覆流程、背景記憶寫入的緩衝。與 Discord 無關，方便測試。"""
from __future__ import annotations

import asyncio
import logging
import re
from collections import defaultdict, deque
from datetime import date, datetime
from typing import Callable
from urllib.parse import unquote_plus

import httpx

from . import prompts, throttle, trace, vision
from .config import Settings
from .embedding import Embedder, build_embedder
from .llm import LMStudioClient
from .memory.store import MemoryStore
from .memory.writer import ChatLine, MemoryWriter
from .profile.store import ProfileFiles
from .rag.decompose import Person, decompose
from .rag.refine import recall
from .rag.rerank import LocalReranker, build_reranker
from .rag.retrieve import Retriever
from .tools import archive as archive_tools
from .tools.web import TOOL_SPECS, WebTools

_TO_TW = None
# 回覆開頭的 [名字]，或對話紀錄格式 [小傑 回覆 阿明「…」]（引用片段可能很長、含 ]）
_NAME_PREFIX = re.compile(r"^\s*(?:\[[^\]\n「]{1,32}回覆[^\n]{0,200}?」\]|\[(?!貼圖)[^\]\n]{1,32}\])\s*")


# D-27：成員的名字可以上網查；只有打過 /不要記我 的人，名字不送到外面（web_search 的關鍵字、open_url 的網址都算）
OPTOUT_SEARCH_BLOCKED = "（這次要查的內容有說過「不要記我」的人的名字，不能送到網路上。拿掉那個名字再查，或直接回答。）"


def _is_alnum(ch: str) -> bool:
    return ch.isascii() and ch.isalnum()


def names_in(query: str, names) -> list[str]:
    """字串裡出現的名字（2 個字以上才算，避免單字誤判）。
    名字全是英文字母／數字：要整個詞相符（前後不是英文字母或數字），不分大小寫；
    其他（中文、或中英夾雜）照子字串比對，寧可多擋。"""
    q = query.lower()
    hit = set()
    for n in names:
        if len(n) < 2:
            continue
        low = n.lower()
        if all(_is_alnum(c) for c in low):
            if re.search(r"(?<![a-z0-9])" + re.escape(low) + r"(?![a-z0-9])", q):
                hit.add(n)
        elif low in q:
            hit.add(n)
    return sorted(hit)


def _decode_url(url: str) -> str:
    """網址解碼到不再變（%E9%98%BF、+ 當空白、被編碼兩次的都解開），名字才比對得到。"""
    for _ in range(3):
        nxt = unquote_plus(url)
        if nxt == url:
            break
        url = nxt
    return url


def web_names_hit(tool: str, args: dict, names) -> list[str]:
    """這次上網工具呼叫會送出去的內容裡，出現了哪些不准送出的名字。其他工具一律回 []。"""
    if tool == "web_search":
        return names_in(str(args.get("query", "")), names)
    if tool == "open_url":
        return names_in(_decode_url(str(args.get("url", ""))), names)
    return []


def to_traditional(text: str) -> str:
    """模型偶爾冒出簡體字：一律轉成台灣繁體（含用語，例如視頻→影片）。原本就是繁體的字不會變。"""
    global _TO_TW
    if _TO_TW is None:
        from opencc import OpenCC
        _TO_TW = OpenCC("s2twp")
    return _TO_TW.convert(text)

log = logging.getLogger(__name__)


def to_messages(lines: list[ChatLine]) -> list[dict]:
    """群聊紀錄轉成 chat messages：連續的使用者訊息合併成一則，Yuki 的話是 assistant。"""
    msgs: list[dict] = []
    for line in lines:
        role = "assistant" if line.is_bot else "user"
        content = line.text if line.is_bot else line.render_chat()
        if msgs and msgs[-1]["role"] == role:
            msgs[-1]["content"] += "\n" + content
        else:
            msgs.append({"role": role, "content": content})
    while msgs and msgs[0]["role"] == "assistant":
        msgs.pop(0)
    return msgs


def memory_chunks(lines: list[tuple[ChatLine, str]], batch: int, gap_minutes: int
                  ) -> list[list[tuple[ChatLine, str]]]:
    """把補抓的對話切成要交給 MemoryWriter 的段落：滿 batch 則，或兩則間隔超過 gap_minutes 就切。"""
    chunks: list[list[tuple[ChatLine, str]]] = []
    cur: list[tuple[ChatLine, str]] = []
    prev: datetime | None = None
    for line, ts in lines:
        t = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
        if cur and (len(cur) >= batch or (prev and (t - prev).total_seconds() > gap_minutes * 60)):
            chunks.append(cur)
            cur = []
        cur.append((line, ts))
        prev = t
    if cur:
        chunks.append(cur)
    return chunks


class Brain:
    def __init__(self, settings: Settings, llm: LMStudioClient, store: MemoryStore, embedder: Embedder | None,
                 reranker: LocalReranker | None):
        self.settings = settings
        self.llm = llm
        self.store = store
        self.embedder = embedder
        self.reranker = reranker
        self.retriever = Retriever(store, embedder, reranker, top_n=settings.retrieve_top_n,
                                   final_k=settings.retrieve_final_k, rrf_k=settings.rrf_k,
                                   rerank_min_score=settings.rerank_min_score,
                                   time_boost=settings.time_boost, use_focus=settings.rerank_focus,
                                   order_top_n=settings.order_top_n, fallback_k=settings.rerank_fallback)
        self.writer = MemoryWriter(store, embedder, llm, self.retriever,
                                   dedup_similarity=settings.dedup_similarity,
                                   max_ops=settings.max_ops_per_batch,
                                   max_deletes=settings.max_deletes_per_batch,
                                   whiteboard_max_items=settings.whiteboard_max_items)
        self.profiles = ProfileFiles(settings.profile_dir)
        self.web = (WebTools(settings.searxng_url, result_chars=settings.web_result_chars)
                    if settings.web_enabled else None)
        self.history: dict[str, deque[ChatLine]] = defaultdict(lambda: deque(maxlen=settings.history_messages))
        self._pending: dict[str, list[ChatLine]] = defaultdict(list)
        self._pending_guild: dict[str, str] = {}
        self._pending_people: dict[str, dict[str, Person]] = defaultdict(dict)
        self._idle_timers: dict[str, asyncio.TimerHandle] = {}
        self._tasks: set[asyncio.Task] = set()
        self._img_http = httpx.AsyncClient(follow_redirects=False)
        self.archive_db = None  # bot 設定：封存區（agentic 檢索用）
        # bot 設定：打過 /不要記我 的人（user ID）、他們在各伺服器的名字。不設就當沒有人退出
        self.opted_out: Callable[[], set[str]] = set
        self.optout_names: Callable[[], set[str]] = set

    @property
    def agentic(self) -> bool:
        return self.settings.retrieval == "agentic"

    @classmethod
    def build(cls, settings: Settings) -> "Brain":
        llm = LMStudioClient(settings.lmstudio_base_url, settings.llm_model,
                             temperature=settings.llm_temperature, max_tokens=settings.llm_max_tokens,
                             timeout=settings.llm_timeout,
                             reasoning_effort=settings.llm_reasoning_effort,
                             json_reasoning_effort=settings.llm_json_reasoning_effort,
                             max_concurrency=settings.llm_max_concurrency)
        if settings.retrieval == "agentic":
            # 不載入 embedding / reranker：事實記憶庫只剩白板和 /記憶 清單在用
            return cls(settings, llm, MemoryStore(settings.db_path, settings.embed_model), None, None)
        embedder = build_embedder(settings)
        store = MemoryStore(settings.db_path, embedder.model_name)
        return cls(settings, llm, store, embedder, build_reranker(settings))

    async def warmup(self) -> None:
        """先把模型載進記憶體，避免第一個人等很久。"""
        if self.embedder:
            await self.embedder.embed_queries(["warmup"])
        if self.reranker:
            await self.reranker.score("warmup", ["warmup"])
        await self.llm.resolve_model()
        log.info("模型暖機完成")

    async def aclose(self) -> None:
        for key in list(self._pending):
            await self.flush(key)
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        await self.llm.aclose()
        await self._img_http.aclose()
        if self.web:
            await self.web.aclose()

    # ---------- 對話紀錄 ----------
    def observe(self, channel_key: str, line: ChatLine) -> bool:
        """加進頻道對話紀錄；連續重複的同一句（同人同內容）只留一則，回傳有沒有加進去。"""
        h = self.history[channel_key]
        if h and throttle.is_repeat(h[-1], line):
            return False
        h.append(line)
        return True

    # ---------- 啟動補抓 ----------
    async def catch_up(self, guild_id: str, channel_key: str, lines: list[tuple[ChatLine, str]]) -> int:
        """補抓到的訊息（[(訊息, 'YYYY-MM-DD HH:MM:SS')]，舊到新）：
        填進頻道對話紀錄，再分段整理成記憶（記憶日期用訊息當時的時間）。回傳整理了幾段。"""
        h = self.history[channel_key]
        live = list(h)  # 補抓期間已經即時收到的訊息排在後面
        h.clear()
        h.extend([line for line, _ in lines[-h.maxlen:]] + live)

        s = self.settings
        if self.agentic or not (s.catchup_memory and s.passive_enabled) or s.catchup_memory_max <= 0:
            return 0
        chunks = memory_chunks(lines[-s.catchup_memory_max:], s.passive_batch, s.catchup_gap_minutes)
        for chunk in chunks:
            await self.writer.process(guild_id, [l for l, _ in chunk], created_at=chunk[-1][1])
        return len(chunks)

    # ---------- 看圖片 ----------
    async def collect_images(self, lines: list[ChatLine], *, spoken: bool = False) -> list:
        """最近幾則裡的圖片（最新優先、有上限），下載縮圖；語音回覆不帶圖。失敗只是少圖。"""
        s = self.settings
        if spoken or not s.vision_enabled or s.vision_max_images <= 0:
            return []
        try:
            items = await vision.collect(lines, n=s.vision_max_images, long_side=s.vision_reply_long_side,
                                         recent=s.vision_recent_lines, client=self._img_http,
                                         max_bytes=s.vision_max_mb * 1024 * 1024)
        except Exception:
            log.warning("取圖失敗，只用文字回覆", exc_info=True)
            return []
        if items and (tr := trace.current()):
            tr.images = [{"msg_id": line.msg_id} for line, _ in items]
        return items

    # ---------- 回覆 ----------
    async def reply(self, guild_id: str, channel_key: str, author: Person, text: str, *,
                    mentions: list[Person] = (), members: list[Person] = (), voice: str = "",
                    spoken: bool = False, owner: str = "", archive_guild: str = "",
                    background: list[str] = (), chimed: bool = False, emoji: str = "") -> str:
        """emoji：群組表情清單（給 prompt 用；後處理在 bot.py）。
        archive_guild：agentic 檢索查哪個伺服器的封存區（測試伺服器借主伺服器的）。
        background：插話判斷給的 need_background（B4 第 2 層），回覆前先查；chimed：這次是她自己決定插話。"""
        if self.agentic:
            return await self._reply_agentic(guild_id, channel_key, author, text, mentions=mentions,
                                             members=members, voice=voice, spoken=spoken, owner=owner,
                                             archive_guild=archive_guild or guild_id,
                                             background=background, chimed=chimed, emoji=emoji)
        today = date.today()
        subs = await decompose(text, author, list(mentions), list(members),
                               wake_words=self.settings.wake_words,
                               llm=self.llm if self.settings.decompose_with_llm else None,
                               max_subqueries=self.settings.max_subqueries, today=today)
        s = self.settings
        memories = await recall(self.retriever, guild_id, subs, llm=self.llm, rewrite=s.query_rewrite,
                                rerank=s.llm_rerank, rerank_top=s.llm_rerank_top, today=today)
        log.info("子查詢 %s 時間 %s → 使用 %d 筆記憶 %s",
                 [(s.text, s.subject.name if s.subject else None) for s in subs],
                 subs[0].time if subs else None, len(memories), [m.id for m in memories])
        whiteboard = [t for _, t, _ in self.store.whiteboard(guild_id)]
        profiles = self.pick_profiles(guild_id, channel_key, author, mentions, subs)
        log.info("人物檔案：%s", [n for n, _ in profiles])
        lines = list(self.history[channel_key])
        images = await self.collect_images(lines, spoken=spoken)
        system = prompts.reply_system(self.settings.bot_name, whiteboard, memories, today, profiles,
                                      web=self.web is not None, voice=voice, spoken=spoken,
                                      owner=owner, vision=bool(images), background=background, chimed=chimed,
                                      emoji=emoji)
        messages = [{"role": "system", "content": system}] + vision.attach(to_messages(lines), images)
        if self.web:
            blocked = self.blocked_names([author, *mentions, *members])

            async def call_tool(name: str, args: dict) -> str:
                return await self.web_call(name, args, blocked)

            text = await self.llm.chat_tools(messages, TOOL_SPECS, call_tool,
                                             max_rounds=self.settings.web_max_rounds)
        else:
            text = await self.llm.chat(messages)
        # 模型有時學對話紀錄的格式，在開頭加上 [名字]
        text = _NAME_PREFIX.sub("", text)
        return to_traditional(text)

    def blocked_names(self, people) -> set[str]:
        """不准送上網的名字：打過 /不要記我 的人（bot 給的全部伺服器的名字，加上這次對話名單裡同一個人的名字）。"""
        out = set(self.optout_names())
        ids = self.opted_out()
        out |= {p.name for p in people if p.id in ids}
        return out - {self.settings.bot_name}

    async def web_call(self, name: str, args: dict, blocked: set[str]) -> str:
        if hit := web_names_hit(name, args, blocked):
            log.warning("擋下上網：%s 含有退出的人的名字（%d 個）", name, len(hit))
            return OPTOUT_SEARCH_BLOCKED
        return await self.web.call(name, args)

    # ---------- Agentic 檢索（開發規格 6.4）----------
    def profile_text(self, guild_id: str, user_id: str, opted_out: set[str] | None = None) -> str:
        """這個人的人物檔案；這個伺服器沒有就借 PROFILE_FALLBACK_GUILD 的。打過 /不要記我 的人一律沒有。"""
        s = self.settings
        if user_id in (self.opted_out() if opted_out is None else opted_out):
            return ""
        text = self.profiles.for_prompt(guild_id, user_id, s.profile_prompt_chars)
        if not text and s.profile_fallback_guild and s.profile_fallback_guild != guild_id:
            text = self.profiles.for_prompt(s.profile_fallback_guild, user_id, s.profile_prompt_chars)
        return text

    def archive_tools(self, guild_id: str, archive_guild: str, members) -> "archive_tools.ArchiveTools":
        def resolve(name: str) -> set[str]:
            n = name.strip().lower()
            exact = {p.id for p in members if p.name.lower() == n}
            return exact or {p.id for p in members if n and n in p.name.lower()}

        def profile(name: str) -> str:
            for uid in resolve(name):
                if text := self.profile_text(guild_id, uid):
                    return text
            return ""

        return archive_tools.ArchiveTools(self.archive_db, archive_guild, resolve_person=resolve, profile=profile,
                                          skip_authors=self.opted_out())

    async def _reply_agentic(self, guild_id: str, channel_key: str, author: Person, text: str, *, mentions,
                             members, voice: str, spoken: bool, owner: str, archive_guild: str,
                             background=(), chimed: bool = False, emoji: str = "") -> str:
        """不先檢索：只帶發言者（和 @ 到的人）的人物檔案；群組的事讓 Yuki 自己用工具查。"""
        s = self.settings
        today = date.today()
        whiteboard = [t for _, t, _ in self.store.whiteboard(guild_id)]
        profiles = self.pick_profiles(guild_id, channel_key, author, mentions, recent=False)
        lines = list(self.history[channel_key])
        images = await self.collect_images(lines, spoken=spoken)
        system = prompts.reply_system(s.bot_name, whiteboard, None, today, profiles, web=self.web is not None,
                                      voice=voice, spoken=spoken, owner=owner,
                                      archive=self.archive_db is not None, vision=bool(images),
                                      background=background, chimed=chimed, emoji=emoji)
        messages = [{"role": "system", "content": system}] + vision.attach(to_messages(lines), images)

        everyone = [author, *mentions, *members]
        specs: list[dict] = []
        tools = None
        if self.archive_db is not None:
            tools = self.archive_tools(guild_id, archive_guild, everyone)
            specs += archive_tools.TOOL_SPECS
        blocked = self.blocked_names(everyone) if self.web else set()
        if self.web:
            specs += TOOL_SPECS

        async def call_tool(name: str, args: dict) -> str:
            if name in archive_tools.NAMES and tools is not None:
                return await tools.call(name, args)
            if self.web:
                return await self.web_call(name, args, blocked)
            return f"（沒有 {name} 這個工具）"

        rounds = s.voice_max_rounds if spoken else max(s.web_max_rounds, 3)
        log.info("人物檔案：%s", [n for n, _ in profiles])
        if specs:
            reply = await self.llm.chat_tools(messages, specs, call_tool, max_rounds=rounds)
        else:
            reply = await self.llm.chat(messages)
        if tools is not None and tools.seen:
            log.info("查到的訊息：%d 則", len(set(tools.seen)))
        self.last_seen = list(tools.seen) if tools is not None else []
        return to_traditional(_NAME_PREFIX.sub("", reply))

    def pick_profiles(self, guild_id: str, channel_key: str, author: Person, mentions=(),
                      subs=(), *, recent: bool = True) -> list[tuple[str, str]]:
        """要放進 prompt 的人物檔案：發言者 → 被提到／問到的人 → 最近在頻道講話的人，有檔案的才算。"""
        s = self.settings
        if not s.profile_enabled or s.profile_max_people <= 0:
            return []
        people: dict[str, Person] = {author.id: author}
        for p in list(mentions) + [q.subject for q in subs if q.subject]:
            people.setdefault(p.id, p)
        for line in (reversed(self.history[channel_key]) if recent else ()):
            if not line.is_bot:
                people.setdefault(line.author.id, line.author)
        out = []
        gone = self.opted_out()
        for p in people.values():
            text = self.profile_text(guild_id, p.id, gone)  # 測試伺服器沒有檔案：借用主伺服器同一個人的
            if text:
                out.append((p.name, text))
            if len(out) >= s.profile_max_people:
                break
        return out

    # ---------- 背景記憶寫入 ----------
    def buffer(self, guild_id: str, channel_key: str, line: ChatLine, mentions: list[Person] = ()) -> None:
        """累積對話，滿 N 則人類訊息或閒置一段時間後，批次交給 MemoryWriter。
        agentic 檢索時不寫事實記憶（封存區本身就是記憶）。"""
        if self.agentic:
            return
        if not line.is_bot and line.author.id in self.opted_out():
            return  # /不要記我 的人講的話不寫進記憶
        self._pending[channel_key].append(line)
        self._pending_guild[channel_key] = guild_id
        for p in mentions:
            self._pending_people[channel_key][p.id] = p
        humans = sum(1 for l in self._pending[channel_key] if not l.is_bot)
        if humans >= self.settings.passive_batch:
            self._spawn(self.flush(channel_key))
            return
        if timer := self._idle_timers.pop(channel_key, None):
            timer.cancel()
        loop = asyncio.get_running_loop()
        self._idle_timers[channel_key] = loop.call_later(
            self.settings.passive_idle_seconds, lambda: self._spawn(self.flush(channel_key)))

    async def flush(self, channel_key: str) -> None:
        if timer := self._idle_timers.pop(channel_key, None):
            timer.cancel()
        lines = self._pending.pop(channel_key, [])
        people = list(self._pending_people.pop(channel_key, {}).values())
        guild_id = self._pending_guild.pop(channel_key, None)
        if guild_id and any(not l.is_bot for l in lines):
            await self.writer.process(guild_id, lines, extra_people=people)

    def _spawn(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._tasks.add(task)

        def done(t: asyncio.Task):
            self._tasks.discard(t)
            if not t.cancelled() and t.exception():
                log.error("背景任務失敗", exc_info=t.exception())

        task.add_done_callback(done)
