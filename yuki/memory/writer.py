"""背景記憶寫入：LLM 只「提議」操作，所有操作都要通過程式端的守門規則才會寫進資料庫。"""
from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field

from .. import prompts
from ..embedding import Embedder
from ..llm import LLMError, LMStudioClient
from ..rag.decompose import Person, SubQuery
from ..rag.retrieve import Retriever
from .store import Memory, MemoryStore

log = logging.getLogger(__name__)

GROUP = "群組"


REPLY_SNIPPET = 40


@dataclass
class ReplyRef:
    """這則訊息回覆的是誰的哪一句（B2）。"""
    name: str            # 被回覆者的顯示名稱
    is_bot: bool = False  # 被回覆的是 Yuki 自己
    snippet: str = ""    # 被回覆那句的前 40 字

    def render(self) -> str:
        snippet = " ".join((self.snippet or "").split())[:REPLY_SNIPPET]
        who = "回覆你" if self.is_bot else f"回覆 {self.name}"
        return who + (f"「{snippet}」" if snippet else "")


@dataclass
class ChatLine:
    author: Person
    text: str
    is_bot: bool = False
    images: list[str] = field(default_factory=list)  # 圖片網址（Discord CDN），不是圖片內容
    msg_id: str = ""
    reply_to: ReplyRef | None = None
    reply_id: str = ""  # 回覆的那則訊息 ID；還沒補到 reply_to 時用來補抓，補不到就清成 ""

    def render(self) -> str:
        return f"[{self.author.name}] {self.text}"

    def render_chat(self) -> str:
        """給 LLM 看的對話紀錄格式：有回覆關係時寫成 [小傑 回覆 阿明「…」] 內容。"""
        if self.reply_to is None:
            return self.render()
        return f"[{self.author.name} {self.reply_to.render()}] {self.text}"


@dataclass
class Applied:
    action: str
    memory_id: str
    text: str = ""


class MemoryWriter:
    def __init__(self, store: MemoryStore, embedder: Embedder, llm: LMStudioClient, retriever: Retriever, *,
                 dedup_similarity: float = 0.90, max_ops: int = 5, max_deletes: int = 2,
                 whiteboard_max_items: int = 20):
        self.store = store
        self.embedder = embedder
        self.llm = llm
        self.retriever = retriever
        self.dedup_similarity = dedup_similarity
        self.max_ops = max_ops
        self.max_deletes = max_deletes
        self.whiteboard_max_items = whiteboard_max_items
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    async def process(self, guild_id: str, lines: list[ChatLine], *, context: list[Memory] | None = None,
                      extra_people: list[Person] = (), created_at: str | None = None) -> list[Applied]:
        """created_at：補抓的舊對話用訊息當時的時間，不然記憶日期會變成補抓那天。"""
        human = [l for l in lines if not l.is_bot]
        if not human:
            return []
        people = list({p.id: p for p in [l.author for l in human] + list(extra_people)}.values())
        async with self._locks[guild_id]:
            if context is None:
                context = await self._context(guild_id, human)
            messages = [
                {"role": "system", "content": prompts.WRITER_SYSTEM},
                {"role": "user", "content": prompts.writer_user(
                    "\n".join(l.render() for l in lines), [p.name for p in people], context)},
            ]
            try:
                data = await self.llm.chat_json(messages, prompts.WRITER_SCHEMA, name="memory_ops")
            except LLMError as e:
                log.warning("記憶寫入略過（LLM 失敗）：%s", e)
                return []
            if not data:
                return []
            return await self.apply(guild_id, data, context, people, author_id=human[-1].author.id,
                                    created_at=created_at)

    async def _context(self, guild_id: str, lines: list[ChatLine]) -> list[Memory]:
        """用每句話各自去撈發言者名下的舊記憶，讓 LLM 有東西可以 update。
        一句一個子查詢：整段黏在一起會變成多主題的長查詢，rerank 分數會全部偏低。"""
        subs = [SubQuery(f"{l.author.name} {l.text}", l.author) for l in lines[-8:]]
        return await self.retriever.retrieve(guild_id, subs, k=10)

    def _resolve(self, name: str, people: list[Person]) -> tuple[str | None, str]:
        name = (name or "").strip()
        if not name or name == GROUP:
            return None, GROUP
        for p in people:
            if p.name.lower() == name.lower():
                return p.id, p.name
        return None, name

    async def apply(self, guild_id: str, data: dict, context: list[Memory], people: list[Person], *,
                    author_id: str | None = None, created_at: str | None = None) -> list[Applied]:
        allowed = {m.id for m in context}
        applied: list[Applied] = []
        deletes = 0
        for op in (data.get("ops") or [])[: self.max_ops]:
            if not isinstance(op, dict):
                continue
            action = op.get("action")
            mem_id = str(op.get("id") or "").strip()
            text = str(op.get("text") or "").strip()

            if action in ("update", "delete") and mem_id not in allowed:
                log.warning("拒絕 %s：ID %s 不在這次檢索結果內", action, mem_id)
                continue

            if action == "delete":
                if deletes >= self.max_deletes:
                    log.warning("拒絕 delete %s：超過單批刪除上限", mem_id)
                    continue
                if self.store.delete(mem_id):
                    deletes += 1
                    applied.append(Applied("delete", mem_id))
                continue

            if not text:
                continue
            subject_id, subject_name = self._resolve(op.get("subject", ""), people)
            if subject_name != GROUP and subject_name.lower() not in text.lower():
                text = f"{subject_name} {text}"  # 保證主詞是名字，日後檢索才找得到
            vec = (await self.embedder.embed_docs([text]))[0]

            if action == "add":
                near = self.store.dense_search(guild_id, vec, 1, subject_id)
                if near and near[0][1] >= self.dedup_similarity:
                    mem_id, action = near[0][0], "update"
                    old = self.store.get(mem_id)
                    if old and old.text == text:
                        continue
                else:
                    mem = self.store.add(guild_id, text, vec, subject_id=subject_id,
                                         subject_name=subject_name, author_id=author_id,
                                         created_at=created_at)
                    applied.append(Applied("add", mem.id, text))
                    continue

            if action == "update" and self.store.update(mem_id, text, vec):
                applied.append(Applied("update", mem_id, text))

        for a in applied:
            log.info("記憶 %s [%s] %s", a.action, a.memory_id, a.text)

        board = str(data.get("whiteboard") or "").strip()
        if board:
            self.store.whiteboard_add(guild_id, board, created_at)
            log.info("白板新增：%s", board)
            await self._compact_whiteboard(guild_id)
        return applied

    async def _compact_whiteboard(self, guild_id: str) -> None:
        items = [t for _, t, _ in self.store.whiteboard(guild_id)]
        if len(items) <= self.whiteboard_max_items:
            return
        target = self.whiteboard_max_items * 2 // 3
        messages = [
            {"role": "system", "content": prompts.WHITEBOARD_SYSTEM.format(target=target)},
            {"role": "user", "content": "\n".join(f"- {t}" for t in items)},
        ]
        try:
            data = await self.llm.chat_json(messages, prompts.WHITEBOARD_SCHEMA, name="whiteboard")
            new_items = [str(i).strip() for i in (data or {}).get("items", []) if str(i).strip()]
        except LLMError:
            new_items = []
        if not new_items:
            new_items = items[-target:]  # LLM 失敗就保留最新的
        self.store.whiteboard_replace(guild_id, new_items[: self.whiteboard_max_items])
