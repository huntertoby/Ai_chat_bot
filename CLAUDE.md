# Yuki 專案守則

Discord 聊天機器人 Yuki，全部在本機跑（LM Studio、Qwen3-ASR、BreezyVoice、SearXNG）。
這個專案用 pm-workflow 流程；共用守則由 plugin 自動注入，這裡只放本專案特有的。

## 溝通
- 一律用**中文（台灣繁體）**回覆使用者。使用者不是工程師，結論先講、少用術語。

## 文件位置（既有專案，不照預設的 docs/ 結構搬檔）
| 文件 | 位置 |
|---|---|
| 專案計畫：目標、不做的事、**階段 A～D 和優先順序** | `企劃.md` |
| 各功能的規格、模型選擇、實測數字、驗收標準（第 6 節是待開發） | `開發規格.md` |
| 決定紀錄 | `docs/決定紀錄.md` |
| 交接文件 | `docs/handoff/<角色>.md` |
| 階段規格／報告 | `docs/stages/` |
| 上手筆記 | `docs/notes/` |
| 安裝、指令、架構 | `README.md` |

`開發規格.md` 很長（700 行），只讀需要的那一節（用 `grep -n "^### "` 找節號）。

## 環境
- Python 在上一層：`..\.venv\Scripts\python.exe`（不要用系統的 python）。
- 測試：`..\.venv\Scripts\python.exe -m pytest -q`（不需要模型或網路）。
- git 有 dubious ownership：每個 git 指令都加 `-c safe.directory=F:/PythonProject/Ai_chat_bot/Ai_chat_bot`。**不要改全域 git 設定。**
- 執行 Yuki：`..\.venv\Scripts\python.exe -u main.py >> logs/yuki.log 2>&1`（背景跑；log 看 `logs/yuki.log`）。
- BreezyVoice 在 WSL：`~/BreezyVoice/.venv`，跑 `scripts/breezy_server.py`，埠 8890。
- LM Studio 埠 8001，**一次只能處理一個請求**；跑評估時 Yuki 也在用它，會互相拖慢。
- 只結束自己啟動的 Yuki／BreezyVoice 程序。
- 開機自動啟動（啟動資料夾捷徑，D-15/D-16）：登入後自動開 LM Studio Server＋Yuki 的模型、Docker＋SearXNG、BreezyVoice、Yuki。
  腳本在 `scripts/autostart/`，log 在 `logs/autostart.log`；`stop.ps1` 不關 LM Studio／Docker（共用）。
  自動開的 Yuki／BreezyVoice 算使用者的；開發時要自己開 Yuki，先跑 `stop.ps1`，測完再 `start.ps1` 還原。

## 隱私（repo 是公開的：github.com/huntertoby/Ai_chat_bot）
- **不能進 repo**：朋友的真名、暱稱、學校、個人經歷、Discord ID、`.env`、`data/`、`logs/`、
  `Luna對話分析.md`、`eval/archive_cases.jsonl`、`searxng/config/settings.yml`。
- 文件和程式裡要舉例時用佔位：阿明、小傑、小美、阿凱；ID 寫 `<主伺服器ID>`、`<你的使用者ID>`。使用者自己的帳號 hunterchiao 可以出現。
- push 前 hook 會跑 `scripts/check_private.py`：名單在 `.claude/private-words.txt`（本機），加上 `.env` 的 ID 和 token。
  被擋下就把內容換成佔位，**不要把字從名單拿掉來過關**。新認識的朋友名字要加進名單。
- Yuki 本身：可以拿成員名字上網查（D-27，取消舊規則）；語音只聽 `VOICE_LISTEN_USERS`。
- Luna 的 repo 沒有授權條款：只能參考做法，不能複製程式碼。

## 寫 prompt 的經驗（這個模型）
- **不要放會被照抄的範例句**：模型會原封不動拿去用在每一題（搜尋關鍵字、兩岸問題都發生過）。描述「怎麼做」，不給「說什麼」。
- 改了 prompt 要用腳本實測（同一組問題跑一次以上，看不同題目的回答），不要只看一題。
- 原模型是中國訓練的：政治立場、簡體字、中國用語都要靠人設壓住（開發規格 3.3）。

## 評估
- 檢索：`python -m scripts.eval_agentic`（合成 200 題）、`--cases eval/archive_cases.jsonl --real`（真實 80 題）、
  `--cases eval/chitchat_cases.jsonl --real --no-judge`（閒聊 30 句）。一輪約 15 分鐘。
- 改了檢索或 prompt，三組都要重跑，跟開發規格 6.4 的上一版比較。
