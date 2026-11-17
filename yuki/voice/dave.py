"""讓 discord-ext-voice-recv 收得到 DAVE（Discord 語音端對端加密）的聲音。

voice-recv 只解開傳輸層加密；DAVE 啟用時，解出來的仍是端對端加密過的 opus，
要再用 discord.py 的 davey session 解一次。這裡把解密包進 reader 的 decrypt_rtp。
"""
from __future__ import annotations

import logging

import discord.opus

log = logging.getLogger(__name__)

try:
    import davey
except ImportError:  # 沒裝 davey 時 discord.py 也連不上強制 DAVE 的頻道
    davey = None

OPUS_SILENCE = bytes([0xF8, 0xFF, 0xFE])
PCM_SILENCE = bytes(discord.opus.Decoder.FRAME_SIZE)  # 20ms、48kHz 雙聲道 16-bit


def e2ee_decrypt(state, user_id: int | None, data: bytes, why: dict | None = None) -> bytes | None:
    """state：discord.py 的 VoiceConnectionState。回傳 opus，或 None（這個封包丟掉）。
    why：給除錯用，記下每種結果的次數。"""
    def note(k):
        if why is not None:
            why[k] = why.get(k, 0) + 1

    session = getattr(state, "dave_session", None)
    if davey is None or not getattr(state, "dave_protocol_version", 0) or session is None:
        note("no_dave")
        return data  # 沒有啟用 DAVE
    if user_id is None:
        note("unknown_ssrc")
        return None  # 還不知道這個 ssrc 是誰（speaking 事件還沒到），沒辦法解
    # 不看 session.ready：實測 MLS welcome 處理完後 ready 仍是 False，但已經能解密
    try:
        out = session.decrypt(user_id, davey.MediaType.audio, data)
        note("decrypted")
        return out
    except Exception as e:
        if session.can_passthrough(user_id):
            note("passthrough")
            return data
        note(f"error:{type(e).__name__}:{str(e)[:40]}|{session.status}|ready={session.ready}")
        return None


def _make_decoder_safe() -> None:
    """解碼器碰到壞封包會丟例外，整個收音執行緒就停了：改成輸出一格靜音。"""
    D = discord.opus.Decoder
    if getattr(D, "_yuki_safe", False):
        return
    raw = D.decode

    def safe_decode(self, data, *, fec=False):
        try:
            return raw(self, data, fec=fec)
        except discord.opus.OpusError:
            return PCM_SILENCE

    D.decode = safe_decode
    D._yuki_safe = True


def _patch_gateway() -> None:
    """discord.py 處理完 MLS welcome/commit 後沒有呼叫 prepare/execute_transition，
    davey 就不會建立「解別人聲音」用的解密器（解密時出現 NoDecryptorForUser）。這裡補上。"""
    import struct

    from discord.gateway import DiscordVoiceWebSocket as WS
    if getattr(WS, "_yuki_dave", False):
        return
    raw_binary, raw_json = WS.received_binary_message, WS.received_message

    async def received_binary_message(self, msg: bytes) -> None:
        await raw_binary(self, msg)
        op = msg[2] if len(msg) > 2 else None
        session = getattr(self._connection, "dave_session", None)
        if session is None or op not in (WS.MLS_ANNOUNCE_COMMIT_TRANSITION, WS.MLS_WELCOME):
            return
        tid = struct.unpack_from(">H", msg, 3)[0]
        try:
            session.prepare_transition(tid, self._connection.dave_protocol_version)
            if tid == 0:  # 0 = 立即生效，不會再收到 execute
                session.execute_transition(0)
        except Exception:
            log.exception("DAVE prepare_transition(%s) 失敗", tid)

    async def received_message(self, msg) -> None:
        await raw_json(self, msg)
        session = getattr(self._connection, "dave_session", None)
        if session is not None and msg.get("op") == WS.DAVE_EXECUTE_TRANSITION:
            try:
                session.execute_transition(msg["d"]["transition_id"])
            except Exception:
                log.exception("DAVE execute_transition 失敗")

    WS.received_binary_message = received_binary_message
    WS.received_message = received_message
    WS._yuki_dave = True


def install() -> None:
    """連語音之前呼叫一次。"""
    _patch_gateway()
    _make_decoder_safe()


def patch_reader(vc) -> None:
    """vc.listen(...) 之後馬上呼叫。"""
    install()
    decryptor = vc._reader.decryptor
    original = decryptor.decrypt_rtp
    why: dict = {}

    def decrypt_rtp(packet):
        data = original(packet)
        out = e2ee_decrypt(vc._connection, vc._get_id_from_ssrc(packet.ssrc), data, why)
        return OPUS_SILENCE if out is None else out  # 解不開就給靜音框，解碼器照常前進

    decryptor.decrypt_rtp = decrypt_rtp
    vc._dave_stats = why
