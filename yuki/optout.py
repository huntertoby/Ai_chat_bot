"""/不要記我（以人為單位）：不管在哪個伺服器或私訊打，所有伺服器關於這個人的資料都刪掉，之後也不再收。

借資料的地方（測試伺服器借主伺服器的封存區和人物檔案、私訊借主伺服器的檔案）各自再查一次退出名單，
這裡只負責「刪」和「記下來」。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import legacy_memory


@dataclass
class Forgotten:
    messages: int       # 封存的聊天訊息
    profiles: int       # 人物檔案（.md，每個伺服器一份）
    facts: int          # 整理人物檔案用的一條條資訊
    memories: int       # 舊版 RAG 的事實記憶（已不再使用，資料還在硬碟上）
    voice: int          # 語音頻道進出紀錄


def forget_everywhere(user_id: str, where: str, *, archive, sync, voice, profile_db, profiles,
                      legacy_db: Path | str) -> Forgotten:
    """where：在哪裡打的（伺服器 ID 或 dm-<id>），只拿來記錄。"""
    n = archive.opt_out(where, user_id)
    sync.forget_skip()  # 每個伺服器的「不收誰」都要重讀
    return Forgotten(messages=n, profiles=profiles.delete_everywhere(user_id),
                     facts=profile_db.forget(None, user_id), memories=legacy_memory.forget_person(legacy_db, user_id),
                     voice=voice.forget_user(None, user_id))
