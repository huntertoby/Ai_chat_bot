# Yuki - AI Chat Bot 🤖

Yuki 是一個具備**長期記憶能力**與**主被動觀察機制**的 Discord 聊天機器人。透過連接本地端 LLM（例如 LM Studio），Yuki 不僅能與使用者流暢對話，還能記住群組中的大小事、個人偏好，甚至在背景默默觀察並自動更新情報。

---

## ✨ 核心特色功能

### 1. 🧠 雙層記憶機制 (Dual-Layer Memory)
- **深層記憶片段 (Vector Database)**：採用 ChromaDB 向量資料庫，記錄每個使用者的具體瑣事、個人偏好與對話細節。當你提起相關話題時，Yuki 能透過語意搜尋 (Cosine Similarity) 喚醒相關記憶。
- **核心白板 (Global Whiteboard)**：記錄群組的全局狀態與重大事件，讓 Yuki 隨時掌握群組的最新動態。

### 2. 🕵️ 主被動觀察機制 (Active & Passive Observation)
- **主動對話模式**：當你 Tag 或是呼叫 `Yuki` 時，她會啟動持續注意力 (Attention Timeout 預設為 3 分鐘)，在這段期間內你可以像跟真人一樣連續對話，無須一直 Tag。
- **被動背景萃取**：即使你沒有呼叫 Yuki，她也會在背景「默默聽」。當緩衝池 (Buffer) 滿了，或是她聽到與舊記憶高度相關的話題時，就會觸發背景 LLM 進行情報萃取，自動新增、更新或刪除記憶。

### 3. 🔌 本地 LLM 無縫串接
- 專為本地端大型語言模型設計。目前預設支援串接 **LM Studio** 的 Chat 與 Embeddings API。
- 使用自訂的 Prompt 搭配 JSON 格式輸出，讓 LLM 自主決定何時該執行記憶的 CRUD (增刪改查)。

---

## 📁 專案架構

- `main.py`: 機器人的主程式入口，負責處理 Discord API 事件、指令，以及注意力機制的判斷。
- `modules/llm_engine.py`: 核心大腦。負責與本地端 LLM 溝通、組合 System Prompt、解析 JSON 回應，並執行主動對話與被動萃取的邏輯。
- `memory/memory_db.py`: 記憶庫管理。封裝了 ChromaDB 的操作，負責向量記憶的寫入、更新、刪除與相似度搜尋。
- `chroma_memory/`: (執行後自動產生) 存放 ChromaDB 本地端資料庫檔案。

---

## 🚀 快速開始

### 1. 環境準備
確保已安裝必要的 Python 套件：
```bash
pip install discord.py requests python-dotenv chromadb
```

### 2. 環境變數設定
在專案根目錄建立 `.env` 檔案，並填入你的 Discord Bot Token：
```env
DISCORD_TOKEN=你的_DISCORD_BOT_TOKEN_放這裡
```

### 3. 啟動本地 LLM 服務
請開啟 LM Studio，並啟動 Local Server。確保以下 API 皆可正常運作：
- Chat Completions API: `http://<你的IP>:1234/v1/chat/completions`
- Embeddings API: `http://<你的IP>:1234/v1/embeddings`
*(若 IP 或 Port 不同，請至 `modules/llm_engine.py` 中修改 `LM_STUDIO_API_URL` 與 `LM_STUDIO_EMBEDDING_URL`)*

### 4. 啟動 Yuki
```bash
python main.py
```

---

## 📜 指令列表

- `!記憶` (或 `!memories`, `!allinfo`): 顯示 Yuki 目前在群組中記住的所有八卦與情報。

---

## 📜 授權條款

本專案採用 [MIT License](LICENSE) 授權。
