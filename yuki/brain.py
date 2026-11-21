"""把各元件串起來：頻道對話紀錄、回覆流程（Yuki 自己用工具查封存區、上網）。與 Discord 無關，方便測試。"""
from __future__ import annotations

import logging
import re
import time
from collections import defaultdict, deque
from datetime import date
from typing import Callable
from urllib.parse import unquote_plus

import httpx

from . import prompts, reply_check, throttle, trace, vision
from .chat import ChatLine, Person
from .config import Settings
from .llm import LMStudioClient
from .profile.store import ProfileFiles
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


OPENING_CHARS = 6      # 開頭取前幾個字
RECENT_OPENINGS = 4    # 看她最近幾則回覆
_OPEN_STRIP = " \t\n,，.。!！?？~～…、:：「」"
_OPEN_CUT = re.compile(r"[,，.。!！?？~～…、:：\n]")
_AT = re.compile(r"@([^\s@，,。！？!?]+)")
# D-36：常見的台灣髒話（不求完整）。「幹」排除幹嘛、幹什麼、幹話、能幹、樹幹這類不是罵人的詞
SWEAR = re.compile(r"(?<![能樹苦骨才主實精])幹(?![嘛麼啥什部活勁練架話事瞪])|靠北|靠腰|靠杯|媽的|他媽|你媽|三小|北七|"
                   r"白癡|白痴|智障|雞掰|機掰|雞巴|機歪|操你|操妳|操他|操機|淦|去死")
SWEAR_SCOPE = 5  # 看最近幾則別人的訊息（不算她自己的，包含正在回的那則）


def has_swear(text: str) -> bool:
    return bool(SWEAR.search(text))


def others_swore(lines: list[ChatLine], n: int = SWEAR_SCOPE) -> bool:
    """最近 n 則別人的訊息裡有沒有人講髒話（D-36：別人先講她才跟著講）。"""
    others = [line for line in lines if not line.is_bot][-n:]
    return any(has_swear(line.text) for line in others)


def recent_openings(lines: list[ChatLine], n: int = RECENT_OPENINGS, k: int = OPENING_CHARS) -> list[str]:
    """她最近 n 則回覆的開頭（舊到新）：去掉前面的標點，取前 k 個字，遇到標點就截斷。"""
    out: list[str] = []
    for line in reversed(lines):
        if not line.is_bot:
            continue
        head = _OPEN_CUT.split(line.text.strip(_OPEN_STRIP), 1)[0][:k].strip()
        if head:
            out.append(head)
        if len(out) >= n:
            break
    return out[::-1]


def latest_note(lines: list[ChatLine], bot_name: str, mentions=()) -> str:
    """這次回覆的提示（R2）：要接哪一則、最新一則 @ 到的人是誰、她最近幾則的開頭、D-36 髒話那句。都是程式算出來的事實；只有髒話那句可說時回空字串（那句改放 system）。"""
    if not lines or lines[-1].is_bot:
        return ""
    last = lines[-1]
    parts: list[str] = []
    block = 0  # 最後一段連續 user 訊息有幾則（to_messages 會合併成一則）
    for line in reversed(lines):
        if line.is_bot:
            break
        block += 1
    if block > 1:
        parts.append(prompts.LATEST_NOTE_LAST.format(author=last.author.name))
    me = bot_name.lower()
    names = [p.name for p in mentions if p.name.lower() != me and f"@{p.name}" in last.text]
    for m in _AT.finditer(last.text):
        n = m.group(1)
        if n.lower() != me and not any(n.startswith(x) or x.startswith(n) for x in names):
            names.append(n)
    if names:
        parts.append(prompts.LATEST_NOTE_AT.format(names="、".join(names)))
    if heads := recent_openings(lines):
        parts.append(prompts.LATEST_NOTE_OPENINGS.format(items="".join(f"「{h}」" for h in heads)))
    if not parts:
        return ""
    return prompts.latest_note(parts + [swear_note(lines)])


def swear_note(lines: list[ChatLine]) -> str:
    """D-36：最近 5 則別人的訊息有人講髒話 → 可以跟著講；都沒有 → 這次不要講。"""
    return prompts.LATEST_NOTE_SWEAR_OK if others_swore(lines) else prompts.LATEST_NOTE_NO_SWEAR


def reply_messages(system: str, lines: list[ChatLine], bot_name: str, mentions=(), *, note: bool = True) -> list[dict]:
    """回覆用的 messages：system＋對話紀錄，最新一則 user 訊息後面用括號附上這次回覆的提示（latest_note）。
    R2 實測：提示放 system 最後時，她比較常去查聊天紀錄、開頭重複也壓不太下來，所以接在最新一則後面。
    D-36 髒話那句每次都要有：有其他提示時跟著一起附在後面；沒有（例如只有一句提問）就只放 system 最後，
    單獨接一個括號在提問後面會讓查聊天紀錄變差（R2 實測）。
    note=False：不附提示（R3 回覆壞掉時的最後一招），髒話那句照樣放 system。"""
    msgs = to_messages(lines)
    if not lines or lines[-1].is_bot or not msgs or msgs[-1]["role"] != "user":
        return [{"role": "system", "content": system}] + msgs
    if note and (text := latest_note(lines, bot_name, mentions)):
        msgs[-1] = {**msgs[-1], "content": msgs[-1]["content"] + "\n\n（" + text + "）"}
    else:
        system += "\n\n" + swear_note(lines)
    return [{"role": "system", "content": system}] + msgs


BROKEN_RETRIES = 1  # 回覆壞掉（reply_check）時，同樣的內容再生幾次；還是壞就不附提示再生一次，那次照用


def recent_others(lines: list[ChatLine], n: int = 5) -> list[tuple[str, str]]:
    """最近 n 則別人的訊息 [(名字, 內容)]，舊到新（給 reply_check 比對照抄、轉述）。"""
    return [(l.author.name, l.text) for l in lines if not l.is_bot][-n:]


async def checked_chat(gen, system: str, lines: list[ChatLine], bot_name: str, mentions=(), *,
                       images=(), stats: dict | None = None) -> str:
    """產生回覆，程式檢查有沒有壞掉（R3）：壞了就同樣的內容再生 BROKEN_RETRIES 次，
    還是壞就不附「這次回覆」提示再生一次（R2 前的組法），那次不再檢查、照用。
    gen：async (messages) -> 原始回覆文字。回傳已經拿掉開頭 [名字]、轉成台灣繁體的回覆。
    stats：有給就記下 reasons（每次判壞的原因）、rejected（被擋下的回覆）、extra_ms（重生多花的毫秒）、fallback（有沒有用到最後一招）。"""
    others = recent_others(lines)

    async def once(note: bool) -> str:
        msgs = vision.attach(reply_messages(system, lines, bot_name, mentions, note=note), list(images))
        return to_traditional(reply_check.SELF_PREFIX.sub("", _NAME_PREFIX.sub("", await gen(msgs))))

    reply = await once(True)
    reasons: list[str] = []
    rejected: list[str] = []
    t0 = time.monotonic()
    for attempt in range(BROKEN_RETRIES + 1):
        why = reply_check.broken_reason(reply, others=others)
        if not why:
            break
        reasons.append(why)
        rejected.append(reply)
        log.info("回覆壞掉（%s），重生：%r", why, reply[:60])
        if tr := trace.current():
            tr.degrade(f"broken_{why}")
        if attempt < BROKEN_RETRIES:
            reply = await once(True)
        else:
            reply = await once(False)
            if stats is not None:
                stats["fallback"] = True
    if stats is not None:
        stats["reasons"] = reasons
        if rejected:
            stats["rejected"] = rejected
        stats["extra_ms"] = round((time.monotonic() - t0) * 1000) if reasons else 0
    return reply


class Brain:
    def __init__(self, settings: Settings, llm: LMStudioClient):
        self.settings = settings
        self.llm = llm
        self.profiles = ProfileFiles(settings.profile_dir)
        self.web = (WebTools(settings.searxng_url, result_chars=settings.web_result_chars)
                    if settings.web_enabled else None)
        self.history: dict[str, deque[ChatLine]] = defaultdict(lambda: deque(maxlen=settings.history_messages))
        self._img_http = httpx.AsyncClient(follow_redirects=False)
        self.archive_db = None  # bot 設定：封存區（Yuki 用工具查的聊天紀錄）
        # bot 設定：打過 /不要記我 的人（user ID）、他們在各伺服器的名字。不設就當沒有人退出
        self.opted_out: Callable[[], set[str]] = set
        self.optout_names: Callable[[], set[str]] = set
        self.last_check: dict = {}  # 上一次回覆的壞掉檢查結果（checked_chat 的 stats；評估腳本讀）

    @classmethod
    def build(cls, settings: Settings) -> "Brain":
        llm = LMStudioClient(settings.lmstudio_base_url, settings.llm_model,
                             temperature=settings.llm_temperature, max_tokens=settings.llm_max_tokens,
                             timeout=settings.llm_timeout,
                             reasoning_effort=settings.llm_reasoning_effort,
                             json_reasoning_effort=settings.llm_json_reasoning_effort,
                             max_concurrency=settings.llm_max_concurrency)
        return cls(settings, llm)

    async def warmup(self) -> None:
        """先確認 LM Studio 的模型，避免第一個人等很久。"""
        await self.llm.resolve_model()
        log.info("模型暖機完成")

    async def aclose(self) -> None:
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
    def catch_up(self, channel_key: str, lines: list[ChatLine]) -> None:
        """補抓到的訊息（舊到新）填進頻道對話紀錄；補抓期間已經即時收到的訊息排在後面。"""
        h = self.history[channel_key]
        live = list(h)
        h.clear()
        h.extend(list(lines)[-h.maxlen:] + live)

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
                    background: list[str] = (), chimed: bool = False) -> str:
        """不先檢索：只帶發言者（和 @ 到的人）的人物檔案；群組的事讓 Yuki 自己用工具查（開發規格 6.4）。
        群組表情不在這裡：回話寫好後 bot.py 另外挑（D-37，emoji_pick）。
        archive_guild：查哪個伺服器的封存區（測試伺服器借主伺服器的），空字串＝guild_id。
        background：插話判斷給的 need_background（B4 第 2 層），回覆前先查；chimed：這次是她自己決定插話。"""
        s = self.settings
        profiles = self.pick_profiles(guild_id, channel_key, author, mentions, recent=False)
        lines = list(self.history[channel_key])
        images = await self.collect_images(lines, spoken=spoken)
        system = prompts.reply_system(s.bot_name, date.today(), profiles, web=self.web is not None,
                                      voice=voice, spoken=spoken, owner=owner,
                                      archive=self.archive_db is not None, vision=bool(images),
                                      background=background, chimed=chimed)

        everyone = [author, *mentions, *members]
        specs: list[dict] = []
        tools = None
        if self.archive_db is not None:
            tools = self.archive_tools(guild_id, archive_guild or guild_id, everyone)
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

        async def gen(messages):
            if specs:
                return await self.llm.chat_tools(messages, specs, call_tool, max_rounds=rounds)
            return await self.llm.chat(messages)

        # 程式檢查回覆有沒有壞掉、壞了重生（R3）；模型學對話紀錄格式在開頭加的 [名字] 也在裡面拿掉
        self.last_check = {}
        reply = await checked_chat(gen, system, lines, s.bot_name, mentions, images=images, stats=self.last_check)
        if tools is not None and tools.seen:
            log.info("查到的訊息：%d 則", len(set(tools.seen)))
        return reply

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

    # ---------- 查封存區、人物檔案（開發規格 6.4）----------
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

    def pick_profiles(self, guild_id: str, channel_key: str, author: Person, mentions=(),
                      *, recent: bool = True) -> list[tuple[str, str]]:
        """要放進 prompt 的人物檔案：發言者 → 被提到／問到的人 → 最近在頻道講話的人，有檔案的才算。"""
        s = self.settings
        if not s.profile_enabled or s.profile_max_people <= 0:
            return []
        people: dict[str, Person] = {author.id: author}
        for p in mentions:
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
