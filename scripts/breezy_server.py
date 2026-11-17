"""BreezyVoice 語音合成服務（開發規格 6.0）：模型一直載著，Yuki 用 HTTP 叫它念。

BreezyVoice 的套件很舊（torch、numpy 1.x、Linux 才有的 ttsfrd），跟 Yuki 的環境衝突，
所以裝在 WSL 的 ~/BreezyVoice/.venv，在 WSL 裡跑：

    cd ~/BreezyVoice && . .venv/bin/activate && PYTHONUTF8=1 python \
        /mnt/f/PythonProject/Ai_chat_bot/Ai_chat_bot/scripts/breezy_server.py

POST /tts {"text": "..."} → audio/wav（22050Hz 單聲道）。GET /health。
音色 = 參考錄音（data/voice/yuki_ref.wav + yuki_ref.txt 逐字稿）。換聲音只要換這兩個檔再重開。
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import threading
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
BREEZY = Path(os.environ.get("BREEZY_DIR", Path.home() / "BreezyVoice"))
os.chdir(BREEZY)
sys.path[:0] = [str(BREEZY), str(BREEZY / "third_party" / "Matcha-TTS")]

# 顯存瘦身：onnxruntime 的部分（g2pw 注音、語音 tokenizer、說話人特徵）全部改用 CPU。
# g2pw 沒指定 provider，裝了 onnxruntime-gpu 就自己跑上 GPU；tokenizer 寫死用 CUDA。
# 這些只做注音和「參考聲音」的一次性處理，CPU 很夠，GPU 上卻各自佔一大塊記憶體。
import onnxruntime  # noqa: E402

_Session = onnxruntime.InferenceSession


class _CpuSession(_Session):
    def __init__(self, path, sess_options=None, providers=None, **kw):
        super().__init__(path, sess_options=sess_options, providers=["CPUExecutionProvider"], **kw)


onnxruntime.InferenceSession = _CpuSession

import re  # noqa: E402

import torch  # noqa: E402
import torchaudio  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import Response  # noqa: E402
from g2pw import G2PWConverter  # noqa: E402
from pydantic import BaseModel  # noqa: E402
from single_inference import CustomCosyVoice, get_bopomofo_rare, load_wav  # noqa: E402

SR = 22050


class Req(BaseModel):
    text: str


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=str(PROJECT / "data/voice/yuki_ref.wav"))
    ap.add_argument("--ref-text", default=str(PROJECT / "data/voice/yuki_ref.txt"))
    ap.add_argument("--model", default="MediaTek-Research/BreezyVoice-300M")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8890)
    args = ap.parse_args()

    t = time.time()
    tts = CustomCosyVoice(args.model)
    g2p = G2PWConverter()
    prompt_speech = load_wav(args.ref, 16000)
    prompt_text = get_bopomofo_rare(
        tts.frontend.text_normalize_new(Path(args.ref_text).read_text(encoding="utf-8").strip(), split=False), g2p)
    lock = threading.Lock()
    # 參考聲音的特徵（speech token、說話人向量、mel）只算一次，每句只換要念的文字
    base = tts.frontend.frontend_zero_shot("暖機。", prompt_text, prompt_speech)

    def synth(text: str) -> bytes:
        with lock, torch.inference_mode():
            content = get_bopomofo_rare(tts.frontend.text_normalize_new(text, split=False), g2p)
            parts = []
            for seg in re.split(r"(?<=[？！。.?!])\s*", content):  # 跟原本 inference_zero_shot 一樣逐句
                if not seg:
                    continue
                tok, tok_len = tts.frontend._extract_text_token(seg)
                parts.append(tts.model.inference(**{**base, "text": tok, "text_len": tok_len})["tts_speech"].cpu())
            speech = torch.concat(parts, dim=1)
            torch.cuda.empty_cache()  # 生成完把暫存的顯存還回去
        buf = io.BytesIO()
        torchaudio.save(buf, speech, SR, format="wav", encoding="PCM_S", bits_per_sample=16)
        return buf.getvalue()

    synth("暖機一下。")  # 第一次推論特別慢（CUDA 初始化），先跑掉
    print(f"BreezyVoice 就緒（{time.time() - t:.1f} 秒），參考聲音 {args.ref}", flush=True)

    app = FastAPI()

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.post("/tts")
    def tts_endpoint(req: Req):  # 一般 def：FastAPI 會丟到執行緒池，不卡住
        text = req.text.strip()
        if not text:
            raise HTTPException(400, "empty text")
        t0 = time.time()
        wav = synth(text)
        print(f"{time.time() - t0:.2f}s「{text}」", flush=True)
        return Response(wav, media_type="audio/wav")

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
