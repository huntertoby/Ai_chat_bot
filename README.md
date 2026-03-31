# Yuki - AI Chat Bot 🤖

Yuki 是一個具備長期記憶能力的 Discord 聊天機器人，透過連接本地端 LLM（例如 LM Studio）來提供智能對話服務。

## 📁 專案架構

- `main.py`: 機器人的主程式入口，負責啟動與處理 Discord API 互動。
- `memory/`: 存放各使用者的專屬記憶檔案，實現長期對話記憶功能。
- `modules/`: 存放核心功能模組，例如 API 串接、記憶管理與資料處理等。

## ✨ 核心功能

- **長期對話記憶**: 為每位使用者獨立保存對話上下文，讓機器人記得過去的聊天內容。
- **本地 LLM 支援**: 支援無縫串接 LM Studio 等本地大型語言模型服務。
- **Discord 整合**: 提供順暢且即時的 Discord 伺服器互動體驗。

## 🚀 快速開始

1. 確保已安裝必要的 Python 套件（例如 `discord.py`, `requests` 等）。
   ```bash
   pip install -r requirements.txt
   ```
2. 設定您的環境變數或配置檔（包含 Discord Bot Token 與 API Endpoint）。
3. 確保本地端的 LLM（如 LM Studio）已啟動並在指定的 Port 監聽。
4. 啟動機器人：
   ```bash
   python main.py
   ```

## 📜 授權條款

本專案採用 [MIT License](LICENSE) 授權。
