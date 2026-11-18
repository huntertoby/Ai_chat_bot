"""看圖片（開發規格 6.9）：從 Discord 訊息取圖片網址、下載縮圖、轉成 data URL 附進 LLM 的 messages。
圖片只在記憶體裡，不存檔；任何一步失敗都只是少一張圖，不會讓回覆失敗。"""
from __future__ import annotations

import asyncio
import base64
import io
import logging
import re
from typing import Any, Iterable
from urllib.parse import urlparse

import httpx

log = logging.getLogger(__name__)

REPLY_LONG_SIDE = 1280   # 回覆用（被叫到／決定插話後）
GLANCE_LONG_SIDE = 512   # 插話判斷用（B2）
REPLY_MAX_IMAGES = 5
GLANCE_MAX_IMAGES = 3
RECENT_LINES = 10        # 從最近幾則訊息裡找圖
MAX_BYTES = 10 * 1024 * 1024
TIMEOUT = 5.0

ALLOWED_HOSTS = ("cdn.discordapp.com", "media.discordapp.net")
# 連結預覽圖（YouTube 等）：Discord 代理過的那份，網址還是 Discord 的
_PROXY_HOST = re.compile(r"images-ext-\d+\.discordapp\.net")
_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".gif")
IMAGE_PLACEHOLDER = "[圖片]"


def is_allowed_url(url: str) -> bool:
    try:
        p = urlparse(url)
    except ValueError:
        return False
    host = (p.hostname or "").lower()
    return p.scheme == "https" and (host in ALLOWED_HOSTS or bool(_PROXY_HOST.fullmatch(host)))


def _is_image_attachment(a: Any) -> bool:
    ctype = (getattr(a, "content_type", None) or "").lower()
    if ctype.startswith("image/"):
        return True
    name = (getattr(a, "filename", None) or urlparse(getattr(a, "url", "") or "").path).lower()
    return name.endswith(_EXTS)


def image_urls(message: Any) -> list[str]:
    """訊息裡的圖片網址（附件、embed 的圖片／縮圖）；只收 Discord CDN，其他網址不抓。"""
    out: list[str] = []

    def add(url: str | None) -> None:
        if url and is_allowed_url(url) and url not in out:
            out.append(url)

    for a in getattr(message, "attachments", None) or []:
        if _is_image_attachment(a):
            add(getattr(a, "url", None))
    for e in getattr(message, "embeds", None) or []:
        for part in (getattr(e, "image", None), getattr(e, "thumbnail", None)):
            if part:  # 外部圖片用 Discord 代理的 proxy_url，原始網址（例如 YouTube）不抓
                add(getattr(part, "proxy_url", None) or getattr(part, "url", None))
    return out


def _shrink(data: bytes, long_side: int) -> bytes:
    from PIL import Image

    with Image.open(io.BytesIO(data)) as im:
        im.seek(0)  # gif 取第一格
        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGBA")
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.getchannel("A"))
            im = bg
        else:
            im = im.convert("RGB")
        im.thumbnail((long_side, long_side))  # 只縮不放大
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=85)
        return buf.getvalue()


async def fetch_image(url: str, long_side: int = REPLY_LONG_SIDE, *, client: httpx.AsyncClient | None = None,
                      max_bytes: int = MAX_BYTES, timeout: float = TIMEOUT) -> str | None:
    """下載並縮圖，回傳 data URL；失敗回 None（寫 warning，不丟例外）。"""
    if not is_allowed_url(url):
        log.warning("不是 Discord CDN 的圖片網址，不抓")
        return None
    own = client is None
    http = client or httpx.AsyncClient()
    try:
        chunks: list[bytes] = []
        size = 0
        async with http.stream("GET", url, timeout=timeout) as resp:
            resp.raise_for_status()
            async for chunk in resp.aiter_bytes():
                size += len(chunk)
                if size > max_bytes:
                    log.warning("圖片超過大小上限，略過")
                    return None
                chunks.append(chunk)
        jpeg = await asyncio.to_thread(_shrink, b"".join(chunks), long_side)
        return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()
    except Exception as e:
        log.warning("圖片下載或縮圖失敗：%r", e)
        return None
    finally:
        if own:
            await http.aclose()


def pick(lines: Iterable[Any], n: int, recent: int = RECENT_LINES) -> list[tuple[Any, str]]:
    """最近 `recent` 則訊息裡的圖片，最新的優先、最多 n 張；回傳 [(訊息, 網址)]，新到舊。"""
    out: list[tuple[Any, str]] = []
    for line in reversed(list(lines)[-recent:]):
        if getattr(line, "is_bot", False):
            continue
        for url in reversed(getattr(line, "images", None) or []):
            if len(out) >= n:
                return out
            out.append((line, url))
    return out


async def collect(lines: Iterable[Any], *, n: int = REPLY_MAX_IMAGES, long_side: int = REPLY_LONG_SIDE,
                  recent: int = RECENT_LINES, client: httpx.AsyncClient | None = None,
                  max_bytes: int = MAX_BYTES) -> list[tuple[Any, str]]:
    """挑圖並下載；回傳 [(訊息, data URL)]，舊到新（對話順序）。下載失敗的略過。"""
    picked = pick(lines, n, recent)
    if not picked:
        return []
    data = await asyncio.gather(*(fetch_image(u, long_side, client=client, max_bytes=max_bytes) for _, u in picked))
    got = [(line, d) for (line, _), d in zip(picked, data) if d]
    got.reverse()
    return got


async def glance_images(lines: Iterable[Any], n: int = GLANCE_MAX_IMAGES, *, long_side: int = GLANCE_LONG_SIDE,
                        recent: int = RECENT_LINES, client: httpx.AsyncClient | None = None) -> list[str]:
    """低解析度版本（插話判斷用，B2）：最近最多 n 張、長邊 long_side；回傳 data URL 清單。"""
    return [d for _, d in await collect(lines, n=n, long_side=long_side, recent=recent, client=client)]


def label(line: Any) -> str:
    snippet = (getattr(line, "text", "") or "").strip().replace("\n", " ")[:20]
    said = f"那則訊息寫：{snippet}" if snippet and snippet != "（貼了圖片）" else "那則訊息沒有文字"
    return f"（{line.author.name} 貼的圖，{said}）"


def attach(messages: list[dict], items: list[tuple[Any, str]]) -> list[dict]:
    """把圖片附在最後一則 user message 的 content（OpenAI 格式）。每張圖前面一行說明是誰貼的。
    回傳新的 messages（不改動傳入的）；沒有圖或沒有 user message 就原樣回傳。"""
    if not items:
        return messages
    idx = next((i for i in range(len(messages) - 1, -1, -1) if messages[i]["role"] == "user"), None)
    if idx is None:
        return messages
    parts: list[dict] = [{"type": "text", "text": messages[idx]["content"]}]
    for line, url in items:
        parts.append({"type": "text", "text": label(line)})
        parts.append({"type": "image_url", "image_url": {"url": url}})
    out = list(messages)
    out[idx] = {**messages[idx], "content": parts}
    return out


def redact(messages: list) -> list:
    """追蹤紀錄用：image_url 的內容（base64）換成「[圖片]」。"""
    out = []
    for m in messages:
        c = m.get("content") if isinstance(m, dict) else None
        if isinstance(c, list):
            c = [{"type": "text", "text": IMAGE_PLACEHOLDER}
                 if isinstance(p, dict) and p.get("type") == "image_url" else p for p in c]
            m = {**m, "content": c}
        out.append(m)
    return out
