"""看圖片（yuki/vision.py）：不需要網路、模型、Discord。"""
import asyncio
import base64
import io
import json
from types import SimpleNamespace as NS

import httpx
from PIL import Image

from yuki import trace, vision
from yuki.llm import LMStudioClient
from yuki.memory.writer import ChatLine
from yuki.rag.decompose import Person

CDN = "https://cdn.discordapp.com/attachments/1/2/"
XIAO = Person("1", "小傑")


def run(coro):
    return asyncio.run(coro)


def png(w, h, mode="RGB", fmt="PNG") -> bytes:
    buf = io.BytesIO()
    Image.new(mode, (w, h), (200, 30, 30) if mode == "RGB" else (200, 30, 30, 0)).save(buf, fmt)
    return buf.getvalue()


def client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def decode(data_url: str) -> Image.Image:
    assert data_url.startswith("data:image/jpeg;base64,")
    return Image.open(io.BytesIO(base64.b64decode(data_url.split(",", 1)[1])))


# ---- URL 篩選 ----
def test_image_urls_attachments_and_embeds():
    msg = NS(attachments=[NS(url=CDN + "a.png", content_type="image/png", filename="a.png"),
                          NS(url=CDN + "b.jpg", content_type=None, filename="b.JPG"),
                          NS(url=CDN + "c.pdf", content_type="application/pdf", filename="c.pdf")],
             embeds=[NS(image=NS(url="https://media.discordapp.net/x.png"), thumbnail=NS(url=CDN + "a.png")),
                     NS(image=None, thumbnail=None)])
    assert vision.image_urls(msg) == [CDN + "a.png", CDN + "b.jpg", "https://media.discordapp.net/x.png"]


def test_image_urls_blocks_other_domains():
    msg = NS(attachments=[NS(url="https://example.com/a.png", content_type="image/png", filename="a.png"),
                          NS(url="http://cdn.discordapp.com/a.png", content_type="image/png", filename="a.png"),
                          NS(url="https://cdn.discordapp.com.evil.com/a.png", content_type="image/png",
                             filename="a.png")],
             embeds=[NS(image=NS(url="https://i.imgur.com/z.png"), thumbnail=None)])
    assert vision.image_urls(msg) == []


# ---- 下載與縮圖 ----
def test_fetch_resizes_long_side_and_never_upscales():
    c = client(lambda r: httpx.Response(200, content=png(2000, 1000)))
    im = decode(run(vision.fetch_image(CDN + "a.png", 1280, client=c)))
    assert im.size == (1280, 640)
    c = client(lambda r: httpx.Response(200, content=png(300, 200)))
    assert decode(run(vision.fetch_image(CDN + "a.png", 1280, client=c))).size == (300, 200)


def test_fetch_gif_and_alpha_become_jpeg():
    buf = io.BytesIO()
    frames = [Image.new("RGB", (100, 80), c) for c in ("red", "blue")]
    frames[0].save(buf, "GIF", save_all=True, append_images=frames[1:])
    c = client(lambda r: httpx.Response(200, content=buf.getvalue()))
    assert decode(run(vision.fetch_image(CDN + "a.gif", client=c))).size == (100, 80)
    c = client(lambda r: httpx.Response(200, content=png(50, 50, "RGBA")))
    assert decode(run(vision.fetch_image(CDN + "a.png", client=c))).mode == "RGB"


def test_fetch_failures_return_none():
    assert run(vision.fetch_image(CDN + "a.png", client=client(lambda r: httpx.Response(404)))) is None
    assert run(vision.fetch_image(CDN + "a.png", client=client(
        lambda r: httpx.Response(200, content=b"not an image")))) is None

    def boom(r):
        raise httpx.ConnectTimeout("t")
    assert run(vision.fetch_image(CDN + "a.png", client=client(boom))) is None
    # 太大
    c = client(lambda r: httpx.Response(200, content=b"x" * 2000))
    assert run(vision.fetch_image(CDN + "a.png", client=c, max_bytes=1000)) is None


def test_fetch_refuses_other_domain_without_requesting():
    hits = []
    c = client(lambda r: hits.append(r) or httpx.Response(200, content=png(10, 10)))
    assert run(vision.fetch_image("https://example.com/a.png", client=c)) is None
    assert hits == []


# ---- 挑圖、附圖 ----
def history(n_images):
    out = []
    for i, k in enumerate(n_images):
        out.append(ChatLine(XIAO, f"msg{i}", images=[f"{CDN}{i}_{j}.png" for j in range(k)], msg_id=str(i)))
    return out


def test_pick_newest_first_max_five():
    picked = vision.pick(history([2, 0, 3, 2]), 5)
    assert [u.rsplit("/", 1)[1] for _, u in picked] == ["3_1.png", "3_0.png", "2_2.png", "2_1.png", "2_0.png"]
    assert len(vision.pick(history([4, 4]), 5)) == 5
    # 只看最近幾則
    assert vision.pick(history([1] + [0] * 10), 5, recent=10) == []


def test_collect_chronological_and_skips_failures():
    def handler(r):
        return httpx.Response(404) if r.url.path.endswith("1_0.png") else httpx.Response(200, content=png(40, 40))
    got = run(vision.collect(history([0, 1, 1, 1]), n=5, client=client(handler)))
    assert [l.msg_id for l, _ in got] == ["2", "3"]


def test_glance_images_uses_small_side_and_limit():
    c = client(lambda r: httpx.Response(200, content=png(2000, 1000)))
    out = run(vision.glance_images(history([2, 2]), client=c))
    assert len(out) == 3 and all(decode(u).size == (512, 256) for u in out)


def test_attach_format_on_last_user_message():
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "[小傑] 看這個"},
            {"role": "assistant", "content": "好"}, {"role": "user", "content": "[小傑] 這是什麼"}]
    items = [(ChatLine(XIAO, "看這個"), "data:image/jpeg;base64,AA"), (ChatLine(XIAO, ""), "data:image/jpeg;base64,BB")]
    out = vision.attach(msgs, items)
    assert out[1] == msgs[1] and msgs[3]["content"] == "[小傑] 這是什麼"  # 原本的不變
    parts = out[3]["content"]
    assert parts[0] == {"type": "text", "text": "[小傑] 這是什麼"}
    assert [p["type"] for p in parts] == ["text", "text", "image_url", "text", "image_url"]
    assert "小傑" in parts[1]["text"] and "看這個" in parts[1]["text"]
    assert parts[2]["image_url"]["url"].endswith("AA")
    assert vision.attach(msgs, []) is msgs


# ---- 追蹤紀錄 ----
def test_trace_has_no_base64(tmp_path):
    trace.reset()
    trace.configure(tmp_path)
    secret = "data:image/jpeg;base64," + "QUJD" * 50
    msgs = vision.attach([{"role": "user", "content": "看"}], [(ChatLine(XIAO, "x"), secret)])

    def handler(r):
        return httpx.Response(200, json={"choices": [{"message": {"content": "好"}, "finish_reason": "stop"}]})

    async def go():
        llm = LMStudioClient("http://x/v1", "m")
        llm._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        async with trace.start("reply", guild="g", channel="c") as tr:
            tr.images = [{"msg_id": "7"}]
            await llm.chat(msgs)
        await llm.aclose()
    run(go())
    trace.reset()
    raw = "".join(p.read_text(encoding="utf-8") for p in tmp_path.glob("ai-*.jsonl"))
    assert "QUJD" not in raw and "base64" not in raw
    rec = json.loads(raw.splitlines()[0])
    assert rec["images"] == 1 and rec["image_sources"] == [{"msg_id": "7"}]
    content = rec["attempts"][0]["messages"][0]["content"]
    assert {"type": "text", "text": "[圖片]"} in content


def test_brain_collect_images_end_to_end():
    """回歸：B1 實測時 brain 傳了 collect 不收的參數，整段被 except 吃掉、0 張圖。這裡走 brain 那一層。"""
    from yuki.brain import Brain
    settings = NS(vision_enabled=True, vision_max_images=5, vision_reply_long_side=1280,
                  vision_recent_lines=10, vision_max_mb=10)
    fake = NS(settings=settings, _img_http=client(lambda req: httpx.Response(200, content=png(800, 600))))
    items = run(Brain.collect_images(fake, history([1, 1])))
    assert len(items) == 2 and all(d.startswith("data:image/jpeg;base64,") for _, d in items)
    assert run(Brain.collect_images(fake, history([1, 1]), spoken=True)) == []


def test_link_preview_uses_discord_proxy():
    """YouTube 等連結預覽圖：用 Discord 代理的 proxy_url（images-ext-N.discordapp.net），原始網址不抓。"""
    thumb = NS(url="https://i.ytimg.com/vi/abc/hqdefault.jpg",
               proxy_url="https://images-ext-1.discordapp.net/external/xyz/https/i.ytimg.com/vi/abc/hqdefault.jpg")
    msg = NS(attachments=[], embeds=[NS(image=None, thumbnail=thumb)])
    assert vision.image_urls(msg) == [thumb.proxy_url]
    assert vision.is_allowed_url("https://images-ext-12.discordapp.net/external/a.png")
    for bad in ("https://images-ext-1.discordapp.net.evil.com/a.png", "https://images-ext-x.discordapp.net/a.png",
                "http://images-ext-1.discordapp.net/a.png", "https://i.ytimg.com/vi/abc/hqdefault.jpg"):
        assert not vision.is_allowed_url(bad), bad
