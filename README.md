# Yuki - AI Chat Bot

Yuki 是一隻住在 Discord 群組裡的聊天機器人，定位是**群組裡的一個成員**，不是放在 Discord 裡的 ChatGPT：
記得大家、知道群組最近發生什麼事、脾氣好接得住鬧，也能進語音頻道一起聊。

全部在本機跑：LLM 用 LM Studio，語音辨識和語音合成用本機 GPU，搜尋用自架的 SearXNG。

> 設計理由與優先順序：[企劃.md](企劃.md)；每個功能的實際規格、模型選擇、實測數字與待開發項目：[開發規格.md](開發規格.md)。

## 功能

| 功能 | 說明 |
|---|---|
| 聊天 | 被 @、叫名字、回覆她、私訊時回應，之後 3 分鐘內持續注意頻道。人設寫的是「群組氛圍」和「態度」，不是固定的個性清單 |
| 查群組的事（Agentic RAG） | 不用 embedding：Yuki 需要時自己用工具搜聊天紀錄（`search_chat`、`read_chat`）和人物檔案（`get_profile`）。底層是 SQLite FTS5 全文檢索 |
| 人物檔案 | 從聊天紀錄整理每個人的 `.md` 檔案，閒置時自動增量更新；當成「印象」而不是事實 |
| 上網 | `web_search`（自架 SearXNG）、`open_url`（網頁正文、YouTube 字幕）。不准連內網、不准拿成員名字去搜 |
| 啟動補抓 | 每次上線先把關機期間的公開頻道訊息補進封存區 |
| 語音頻道狀態 | 知道誰在哪個語音頻道、誰剛進出、今天待了多久 |
| 語音對話 | `/來語音` 叫她進頻道：收音 → 切句 → ASR → 回覆 → TTS 念出來；講到一半被插話會停 |
| 測試伺服器 | 可以指定測試用伺服器：不封存、不整理記憶，人物檔案和聊天紀錄借主伺服器的 |

## 架構

```
文字訊息 / 語音（ASR 轉文字）
   │
   ▼
回覆（LM Studio，tool calling）
   system prompt = 人設 + 群組近況 + 語音頻道現況 + 發言者的人物檔案
   │
   ├─ 需要時查：search_chat / read_chat / get_profile ──→ archive.db（SQLite + FTS5）
   └─ 需要時上網：web_search / open_url ───────────────→ SearXNG（Docker，只綁本機）
   │
   ▼
文字回覆 ──（語音模式）──→ BreezyVoice（WSL，HTTP）→ Discord 語音

背景：公開頻道訊息即時封存（同時建全文索引）；閒置時增量更新人物檔案
```

重點設計：

- **不先檢索**：平常聊天只帶發言者的人物檔案和最近對話，回覆約 0.3 秒；問到群組的事才查（實測 30 句閒聊只查了 1 次）。
- **全文檢索的細節**：jieba 斷繁體很不準，索引和查詢都先轉簡體再斷詞；依訊息長度和新舊調整排序；
  時間或人的限制下找不到時自動放寬再找一次；關鍵字是成員名字時改成「他說的話加分」。
- **工具次數用完還想查**時，要求模型用手上的資料直接回答，不會把 `<tool_call>` 文字當成回覆送出。
- **隱私**：只封存公開頻道；`/不要記我` 刪掉自己的封存訊息與人物檔案；搜尋結果只給模型看。
- **LM Studio 一次只處理一個請求**（MTP 推測解碼不支援並行），聊天和背景工作共用同一個佇列。
- **完全非同步**：HTTP 都用 httpx.AsyncClient，模型推論放在 thread，不會卡住 Discord。

## 專案結構

```
main.py                    入口
yuki/
  config.py                所有設定（從 .env 讀）
  bot.py                   Discord 事件、注意力機制、斜線指令、語音指令
  brain.py                 回覆流程（agentic / 舊的 embedding RAG）
  llm.py                   LM Studio async client、tool calling
  prompts.py               人設與所有 prompt
  archive.py               封存區（SQLite）+ 全文索引
  archive_sync.py          啟動補抓、即時封存
  voice_presence.py        語音頻道在場狀態
  tools/archive.py         search_chat / read_chat / get_profile
  tools/web.py             web_search / open_url
  profile/                 人物檔案：抽取、合併、閒置時自動更新
  voice/                   語音對話：DAVE 解密、切句（silero-vad）、ASR、TTS、對話流程
  memory/、rag/、embedding.py   舊的事實記憶 + embedding RAG（RETRIEVAL=embedding 時才用）
scripts/                   封存抓取、建索引、評估、語音測試、BreezyVoice 服務
searxng/                   SearXNG 的 docker compose 與設定範例
tests/                     單元測試（不需要模型或網路）
eval/                      評估題目（合成資料）
```

## 安裝

需求：NVIDIA 顯卡（實測 RTX 5090 32GB）、Windows + WSL2、Docker Desktop、ffmpeg。

1. 套件（有 NVIDIA 顯卡請先裝 CUDA 版 PyTorch）

   ```bash
   pip install -r requirements.txt
   ```

2. 複製 `.env.example` 成 `.env`，填 `DISCORD_TOKEN`，確認 `LMSTUDIO_BASE_URL`。
   Discord Developer Portal 要開 **Message Content Intent** 和 **Server Members Intent**。

3. LM Studio 載入聊天模型、啟動 Local Server（模型選擇與設定見開發規格第 2 節）。`LLM_MODEL` 留空會用目前載入的模型。

4. SearXNG（上網用）：

   ```bash
   cp searxng/config/settings.example.yml searxng/config/settings.yml   # 把 secret_key 換成隨機字串
   docker compose -f searxng/docker-compose.yml up -d
   ```

5. 封存區與全文索引（第一次）：

   ```bash
   python -m scripts.archive_fetch --guild-id <伺服器ID> --role <身分組名稱>
   python -m scripts.build_fts
   ```

6. 啟動

   ```bash
   python main.py
   ```

### 開機自動啟動（Windows，選用）

用「啟動資料夾捷徑」在登入 Windows 後自動開 BreezyVoice（WSL）和 Yuki；掛掉會 10 秒後重開（1 小時內重開超過 5 次就停，看 log）。
LM Studio 和 SearXNG 不包含在內，要自己設定開機啟動。

```powershell
powershell -ExecutionPolicy Bypass -File scripts\autostart\install.ps1     # 在啟動資料夾建立捷徑
powershell -ExecutionPolicy Bypass -File scripts\autostart\uninstall.ps1   # 移除捷徑
powershell -ExecutionPolicy Bypass -File scripts\autostart\start.ps1       # 手動啟動（不跳視窗；已在跑的不會重複開）
powershell -ExecutionPolicy Bypass -File scripts\autostart\stop.ps1        # 手動停止（只結束腳本自己開的程序）
```

- log：`logs/autostart.log`（看顧程式）、`logs/yuki.log`、`logs/breezy.log`；程序編號記在 `logs/autostart/*.pid`。
- 要自己開 Yuki 做開發前，先跑 `stop.ps1`，做完再 `start.ps1`。
- BreezyVoice 的 WSL 發行版（預設 `swift`）在 `scripts/autostart/common.ps1` 最上面改。

### 語音（選用）

- `.env` 設 `VOICE_ENABLED=true`、`VOICE_GUILD_IDS`、`VOICE_LISTEN_USERS`（只聽這些人，其他人的聲音不轉文字也不存）。
- Discord 語音強制端對端加密（DAVE），需要 `davey`；`discord-ext-voice-recv` 不會解 DAVE，`yuki/voice/dave.py` 補上了。
- 語音辨識：Qwen3-ASR（`VOICE_ASR_MODEL`，0.6B 約 1.9GB、1.7B 約 4.7GB 顯存）。
- 語音合成：`VOICE_TTS=edge`（免安裝，但要念的文字會送到微軟）或 `VOICE_TTS=breezy`（BreezyVoice，台灣腔、本機）。
  BreezyVoice 的套件太舊，裝在 WSL 的獨立環境，用 `scripts/breezy_server.py` 常駐；安裝步驟和要改的套件版本見開發規格 6.0。
  服務沒開時會自動改用 Edge。
- 測試收音／播放：`python -m scripts.voice_probe <伺服器ID> [秒數]`。

## 指令與對話方式

| 指令 | 說明 |
|---|---|
| `/記憶 關鍵字:...` | 搜聊天紀錄（跟 Yuki 的 `search_chat` 同一套） |
| `/檔案 對象:@某人` | 看 Yuki 整理的人物檔案（只有你看得到） |
| `/不要記我` | 刪掉你的人物檔案、封存訊息和語音進出紀錄，之後也不再整理你 |
| `/白板` | 群組近況白板 |
| `/來語音`、`/離開語音` | 叫 Yuki 進出你所在的語音頻道 |

- @Yuki、訊息含「yuki」、回覆 Yuki 或私訊 → 回應，並在 3 分鐘內持續注意這個頻道；說「掰掰」「退下」結束。
- 語音裡只有你和她時每句都回；有其他人在時要叫她名字。

## 測試與評估

```bash
python -m pytest -q
```

Agentic 檢索評估（讓 Yuki 用工具自己查，LLM 判斷回答對不對）：

```bash
python -m scripts.eval_agentic                                  # 合成記憶 200 題
python -m scripts.make_archive_cases --n 80                     # 從自己的封存區出題（含真實聊天，不要外流）
python -m scripts.eval_agentic --cases eval/archive_cases.jsonl --real
```

| 測試 | 結果 |
|---|---|
| 真實封存區 80 題（一半刻意換句話說） | 答對 85% |
| 合成記憶 200 題 | 命中 77%，答對 77%（有查的題目 82%） |
| 閒聊不亂查 | 28/30 |
| 不用查的回覆 | 中位數 0.33 秒 |

### 舊的 embedding RAG（`RETRIEVAL=embedding`）

查詢分解 → dense（bge-m3）+ BM25 → RRF → cross-encoder rerank → 時間感知 → LLM 重排。
在 18,270 筆合成記憶上 Hit@6 84%，但每句都要先花約 1 秒檢索、多佔 2.8GB 顯存，現在預設改用 agentic。
評估：`python -m scripts.eval_retrieval --cases eval/synthetic_cases.jsonl --db data/synthetic.db`。

## 授權條款

本專案採用 [MIT License](LICENSE) 授權。
