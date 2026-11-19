"""push 前的個資掃描（pm-workflow 的 push 檢查會跑）：repo 是公開的，朋友的名字、Discord ID、token 不能進去。

比對 HEAD 裡所有被追蹤的文字檔，**加上這次要 push 的每一個 commit**（預設 `@{u}..HEAD`：每個 commit 新增或改到的
檔案、和 commit 訊息）。中間某個 commit 放了名字、後面又刪掉，HEAD 雖然乾淨，名字還是會跟著歷史公開，所以也要擋。
找到就列出來、結束碼 1（push 會被擋下）。
要比對的字從兩個地方來，兩個都不進 repo：
  1. `.claude/private-words.txt`：一行一個（朋友的名字、學校、身分組名稱…）。# 開頭是註解。
  2. `.env`：DISCORD_TOKEN、所有 *_ID／*_IDS／*_USERS／*_GUILD／*_CHANNEL 的值（逗號分隔）。
     熱詞（VOICE_ASR_HOTWORDS）大多是一般詞，不比對；裡面有朋友名字的話寫進 private-words.txt。
另外任何 17～20 位數的數字（Discord ID 的長度）也會被擋，除非寫在 `.claude/private-allow.txt`。

用法：python -m scripts.check_private          # HEAD + @{u}..HEAD（沒有上游分支就只看 HEAD）
      python -m scripts.check_private A..B     # HEAD + 指定範圍的每個 commit（例如 hook 給的 remote..local）
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
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


def _git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-c", f"safe.directory={root.as_posix()}", *args], cwd=root, capture_output=True)


def _blob_texts(root: Path, rev: str, names: list[str], label: str) -> dict[str, str]:
    out = {}
    for n in filter(None, names):
        if n.lower().endswith(BINARY):
            continue
        blob = _git(root, "show", f"{rev}:{n}").stdout
        if b"\0" in blob[:8000]:
            continue  # 二進位檔
        out[f"{label}{n}"] = blob.decode("utf-8", "replace")
    return out


def tracked_text(root: Path = ROOT) -> dict[str, str]:
    r = _git(root, "ls-tree", "-r", "-z", "--name-only", "HEAD")
    r.check_returncode()
    return _blob_texts(root, "HEAD", r.stdout.decode("utf-8").split("\0"), "")


def default_range(root: Path = ROOT) -> str | None:
    """還沒 push 的 commit：有上游分支就是 @{u}..HEAD；沒有就只看 HEAD。"""
    return "@{u}..HEAD" if _git(root, "rev-parse", "--verify", "-q", "@{u}").returncode == 0 else None


def history_text(rev_range: str, root: Path = ROOT) -> dict[str, str]:
    """範圍內每個 commit：它新增或修改的檔案（當時的內容）和 commit 訊息。key 前面標 commit 前 8 碼。"""
    r = _git(root, "rev-list", rev_range)
    r.check_returncode()
    out: dict[str, str] = {}
    for sha in r.stdout.decode().split():
        label = f"{sha[:8]}:"
        out[f"{label}(commit 訊息)"] = _git(root, "log", "-1", "--format=%B", sha).stdout.decode("utf-8", "replace")
        names = _git(root, "diff-tree", "--root", "--no-commit-id", "-r", "-z", "--name-only",
                     "--diff-filter=AMRC", sha).stdout.decode("utf-8").split("\0")
        out.update(_blob_texts(root, sha, names, label))
    return out


def scan(texts: dict[str, str], words: set[str], allow: set[str]) -> list[tuple[str, str]]:
    found = []
    for name, text in texts.items():
        low = text.lower()
        for w in words:
            if w.lower() in low:
                found.append((name, w))
        for m in SNOWFLAKE.findall(text):
            if m not in allow and m not in words:
                found.append((name, m))
    return found


def main(argv: list[str] | None = None, root: Path = ROOT, words: set[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # hook 跑的時候主控台是 Big5，⚠ 之類的字會讓 print 出錯
    argv = sys.argv[1:] if argv is None else argv
    if words is None:
        words = private_words()
        if not (ROOT / ".claude" / "private-words.txt").is_file():
            print("⚠ 沒有 .claude/private-words.txt：只比對 .env 的值和 Discord ID 長度的數字")
    allow = set(lines(ROOT / ".claude" / "private-allow.txt"))
    rev_range = argv[0] if argv else default_range(root)
    texts = tracked_text(root)
    if rev_range:
        texts.update(history_text(rev_range, root))
    found = scan(texts, words, allow)
    if found:
        print("個資掃描沒過：以下內容會被 push 到公開 repo")
        for name, w in sorted(set(found)):
            print(f"  {name}: {w[:2]}…（{len(w)} 字）")  # 不把個資本身印進 log
        print("換成佔位（例如 阿明、<主伺服器ID>）或從 repo 移除後再 commit、push。"
              "已經 commit 的要改寫那個 commit（不能只在後面的 commit 刪掉）。")
        return 1
    where = f"HEAD 和 {rev_range} 的每個 commit" if rev_range else "HEAD"
    print(f"個資掃描通過：{len(words)} 個字詞、{where}裡的文字檔都沒有")
    return 0


if __name__ == "__main__":
    sys.exit(main())
