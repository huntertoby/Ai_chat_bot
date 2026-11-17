"""Yuki 的上網工具：搜尋（自架 SearXNG）、打開網頁、看 YouTube（讀字幕）。

給 LLM 用 tool calling 呼叫。只有兩個工具，本機模型工具越少越穩：
- web_search(query)
- open_url(url)：一般網頁抽出正文；YouTube 連結改抓標題 + 字幕

安全：open_url 不准連到內網（127.0.0.1、192.168.x、100.64.x…），也檢查每一次轉址，
不然群組裡任何人都能叫 Yuki 去打 LM Studio 或路由器的管理頁面。
"""
from __future__ import annotations

import asyncio
import ipaddress
import logging
import re
import socket
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx

log = logging.getLogger(__name__)

TOOL_SPECS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "上網搜尋。需要最新消息、你不知道的事實、比賽結果、價格、新聞時使用。",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "搜尋關鍵字"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_url",
            "description": "打開一個網址讀內容。YouTube 影片會讀標題和字幕，等於看過影片。"
                           "有人貼連結要你看、或搜尋結果需要看細節時使用。",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string", "description": "完整網址"}},
                "required": ["url"],
            },
        },
    },
]

_UA = "Mozilla/5.0 (compatible; YukiBot/1.0; Discord chat bot)"  # 冒充 Chrome 反而會被 Wikipedia 等網站擋
_YT = re.compile(r"(?:youtube\.com/(?:watch\?(?:.*&)?v=|shorts/|live/|embed/)|youtu\.be/)([A-Za-z0-9_-]{11})")
_MAX_BYTES = 3_000_000


class BlockedURL(ValueError):
    pass


def youtube_id(url: str) -> str | None:
    m = _YT.search(url)
    return m.group(1) if m else None


def check_public(url: str) -> None:
    """只准 http(s) 連到公開網路上的主機。"""
    p = urlparse(url)
    if p.scheme not in ("http", "https") or not p.hostname:
        raise BlockedURL("只能開 http/https 網址")
    try:
        infos = socket.getaddrinfo(p.hostname, p.port or (443 if p.scheme == "https" else 80))
    except socket.gaierror as e:
        raise BlockedURL(f"找不到這個網站（{p.hostname}）") from e
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if not ip.is_global:
            raise BlockedURL("不能開內部網路的網址")


def clip(text: str, limit: int) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit] + "…（後面省略）"


def sample(parts: list[str], limit: int) -> str:
    """很長的字幕：平均抽樣各段，讓開頭、中間、結尾都看得到。"""
    full = " ".join(parts)
    if len(full) <= limit:
        return full
    n = len(parts)
    chunks, size = 6, max(1, limit // 6)
    out = []
    for i in range(chunks):
        seg = " ".join(parts[i * n // chunks:(i + 1) * n // chunks])
        out.append(seg[:size])
    return " … ".join(out)


class WebTools:
    def __init__(self, searxng_url: str, *, result_chars: int = 5000, timeout: float = 15,
                 transcript_langs: tuple[str, ...] = ("zh-TW", "zh-Hant", "zh", "zh-Hans", "zh-CN", "en", "ja", "ko")):
        self.searxng_url = searxng_url.rstrip("/")
        self.result_chars = result_chars
        self.transcript_langs = transcript_langs
        self._http = httpx.AsyncClient(timeout=timeout, headers={"User-Agent": _UA},
                                       follow_redirects=False)

    async def aclose(self) -> None:
        await self._http.aclose()

    async def call(self, name: str, args: dict) -> str:
        """執行工具，回傳給 LLM 看的文字。失敗也回文字（讓 Yuki 可以說查不到），不丟例外。"""
        try:
            if name == "web_search":
                return await self.search(str(args.get("query") or ""))
            if name == "open_url":
                return await self.open_url(str(args.get("url") or ""))
            return f"沒有這個工具：{name}"
        except BlockedURL as e:
            return f"打不開：{e}"
        except httpx.HTTPError as e:
            log.warning("工具 %s 失敗：%r", name, e)
            return f"連線失敗：{type(e).__name__}"
        except Exception as e:  # 工具壞掉不能讓整個回覆失敗
            log.exception("工具 %s 出錯", name)
            return f"工具出錯：{type(e).__name__}"

    # ---------- 搜尋 ----------
    async def search(self, query: str, n: int = 6) -> str:
        query = query.strip()
        if not query:
            return "沒有給搜尋關鍵字"
        try:
            resp = await self._http.get(f"{self.searxng_url}/search",
                                        params={"q": query, "format": "json"})
            resp.raise_for_status()
        except httpx.HTTPError as e:
            log.warning("SearXNG 搜尋失敗（%s 有開嗎？）：%r", self.searxng_url, e)
            return "搜尋服務沒有回應"
        results = resp.json().get("results", [])[:n]
        if not results:
            return f"搜尋「{query}」沒有結果"
        lines = [f"{i}. {r.get('title', '').strip()}\n   {r.get('url', '')}\n   {(r.get('content') or '').strip()[:300]}"
                 for i, r in enumerate(results, 1)]
        return clip(f"搜尋「{query}」的結果：\n" + "\n".join(lines), self.result_chars)

    # ---------- 打開網址 ----------
    async def open_url(self, url: str) -> str:
        url = url.strip().strip("<>")
        if vid := youtube_id(url):
            return await self.youtube(vid)
        html, final = await self._get(url)
        text = await asyncio.to_thread(_extract, html, final)
        if not text:
            return f"{final} 讀不到內容（可能要登入或是純圖片／影片）"
        return clip(f"{final} 的內容：\n{text}", self.result_chars)

    async def _get(self, url: str) -> tuple[str, str]:
        for _ in range(5):  # 自己跟轉址，每一跳都檢查是不是內網
            await asyncio.to_thread(check_public, url)
            async with self._http.stream("GET", url) as resp:
                if resp.is_redirect:
                    url = urljoin(url, resp.headers.get("location", ""))
                    continue
                resp.raise_for_status()
                ctype = resp.headers.get("content-type", "")
                if "html" not in ctype and "text" not in ctype:
                    raise BlockedURL(f"不是網頁（{ctype.split(';')[0] or '未知格式'}）")
                body = bytearray()
                async for chunk in resp.aiter_bytes():
                    body += chunk
                    if len(body) > _MAX_BYTES:
                        break
                return body.decode(resp.encoding or "utf-8", errors="replace"), str(resp.url)
        raise BlockedURL("轉址太多次")

    # ---------- YouTube ----------
    async def youtube(self, vid: str) -> str:
        url = f"https://www.youtube.com/watch?v={vid}"
        title = author = ""
        try:
            r = await self._http.get("https://www.youtube.com/oembed", params={"url": url, "format": "json"})
            if r.status_code == 200:
                meta = r.json()
                title, author = meta.get("title", ""), meta.get("author_name", "")
        except httpx.HTTPError:
            pass
        head = f"YouTube 影片：{title or vid}" + (f"（頻道：{author}）" if author else "")
        lang, parts = await asyncio.to_thread(self._transcript, vid)
        if not parts:
            return head + "\n這部影片沒有字幕，看不到內容，只知道標題。"
        budget = self.result_chars - len(head) - 50
        return f"{head}\n字幕（{lang}）：\n{sample(parts, budget)}"

    def _transcript(self, vid: str) -> tuple[str, list[str]]:
        from youtube_transcript_api import YouTubeTranscriptApi
        try:
            listing = YouTubeTranscriptApi().list(vid)
            try:
                t = listing.find_transcript(list(self.transcript_langs))
            except Exception:
                t = next(iter(listing))  # 沒有偏好的語言就拿第一個（通常是自動產生的）
            fetched = t.fetch()
            return t.language_code, [s.text.replace("\n", " ") for s in fetched]
        except Exception as e:
            log.info("YouTube %s 沒有字幕：%s", vid, type(e).__name__)
            return "", []


def _extract(html: str, url: str) -> str:
    import trafilatura
    text = trafilatura.extract(html, url=url, include_comments=False, include_tables=True) or ""
    if not text:  # 抽不到正文就至少給標題和描述
        title = re.search(r"<title[^>]*>(.*?)</title>", html, re.I | re.S)
        desc = re.search(r'<meta[^>]+(?:name|property)="(?:og:)?description"[^>]+content="([^"]*)"', html, re.I)
        text = "\n".join(x.group(1).strip() for x in (title, desc) if x)
    return text
