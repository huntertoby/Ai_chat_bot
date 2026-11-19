"""聊天的基本型別：人、對話紀錄的一行、回覆關係。全專案共用（原本在 rag/decompose.py、memory/writer.py）。"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Person:
    id: str
    name: str


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
