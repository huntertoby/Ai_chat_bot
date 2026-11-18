"""push 前的個資掃描（pm-workflow 的 push 檢查會跑）：repo 是公開的，朋友的名字、Discord ID、token 不能進去。

比對 HEAD 裡所有被追蹤的文字檔，找到就列出來、結束碼 1（push 會被擋下）。
要比對的字從兩個地方來，兩個都不進 repo：
  1. `.claude/private-words.txt`：一行一個（朋友的名字、學校、身分組名稱…）。# 開頭是註解。
  2. `.env`：DISCORD_TOKEN、所有 *_ID／*_IDS／*_USERS／*_GUILD／*_CHANNEL 的值（逗號分隔）。
     熱詞（VOICE_ASR_HOTWORDS）大多是一般詞，不比對；裡面有朋友名字的話寫進 private-words.txt。
另外任何 17～20 位數的數字（Discord ID 的長度）也會被擋，除非寫在 `.claude/private-allow.txt`。

用法：python -m scripts.check_private
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GIT = ["git", "-c", f"safe.directory={ROOT.as_posix()}"]
SNOWFLAKE = re.compile(r"(?<!\d)\d{17,20}(?!\d)")
ENV_KEYS = re.compile(r"^(DISCORD_TOKEN|\w+_(IDS?|USERS|GUILD|CHANNEL))$")
BINARY = (".pdf", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".wav", ".mp3", ".db", ".zip")


def lines(path: Path) -> list[str]:
    if not path.is_file():
        return []
    return [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]


def private_words() -> set[str]:
    words = set(lines(ROOT / ".claude" / "private-words.txt"))
    for l in lines(ROOT / ".env"):
        key, _, val = l.partition("=")
        if ENV_KEYS.match(key.strip()):
            words |= {v.strip().strip('"\'') for v in val.split(",")}
    return {w for w in words if len(w) >= 2}


def tracked_text() -> dict[str, str]:
    names = subprocess.run(GIT + ["ls-tree", "-r", "-z", "--name-only", "HEAD"], cwd=ROOT,
                           capture_output=True, check=True).stdout.decode("utf-8").split("\0")
    out = {}
    for n in filter(None, names):
        if n.lower().endswith(BINARY):
            continue
        blob = subprocess.run(GIT + ["show", f"HEAD:{n}"], cwd=ROOT, capture_output=True).stdout
        if b"\0" in blob[:8000]:
            continue  # 二進位檔
        out[n] = blob.decode("utf-8", "replace")
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # hook 跑的時候主控台是 Big5，⚠ 之類的字會讓 print 出錯
    words = private_words()
    allow = set(lines(ROOT / ".claude" / "private-allow.txt"))
    if not (ROOT / ".claude" / "private-words.txt").is_file():
        print("⚠ 沒有 .claude/private-words.txt：只比對 .env 的值和 Discord ID 長度的數字")
    found = []
    for name, text in tracked_text().items():
        low = text.lower()
        for w in words:
            if w.lower() in low:
                found.append((name, w))
        for m in SNOWFLAKE.findall(text):
            if m not in allow and m not in words:
                found.append((name, m))
    if found:
        print("個資掃描沒過：以下內容會被 push 到公開 repo")
        for name, w in sorted(set(found)):
            print(f"  {name}: {w[:2]}…（{len(w)} 字）")  # 不把個資本身印進 log
        print("換成佔位（例如 阿明、<主伺服器ID>）或從 repo 移除後再 commit、push。")
        return 1
    print(f"個資掃描通過：{len(words)} 個字詞、HEAD 裡的文字檔都沒有")
    return 0


if __name__ == "__main__":
    sys.exit(main())
