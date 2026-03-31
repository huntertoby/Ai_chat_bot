# modules/llm_engine.py
import requests
import json
import re

from Ai_chat_bot.memory import memory_db

# ================= ⚙️ 全域設定區 =================
BACKEND_TYPE = "lm_studio"
LM_STUDIO_API_URL = "http://192.168.0.75:1234/v1/chat/completions"
LM_STUDIO_MODEL = "local-model"
MAX_TOKENS = 4096
MAX_HISTORY_ROUNDS = 20

LM_STUDIO_EMBEDDING_URL = "http://192.168.0.75:1234/v1/embeddings"
LM_STUDIO_EMBEDDING_MODEL = "text-embedding-qwen3-embedding-4b"

# ================= 🎭 JSON 專用 Prompt =================
BASE_SYSTEM_PROMPT = """
你是一個 Discord 群組助手，名字叫 Yuki。請用繁體中文與大家對話。

【雙層記憶機制與管理】
系統提供兩種記憶讓你參考：
1. 【核心白板 (memory.md)】：記錄群組的全局狀態、核心人物設定與重大事件。
2. 【深層記憶片段 (資料庫)】：記錄過去發生的具體瑣事、個人偏好與對話細節（附有 [記憶ID] 與 [記錄時間]）。

請比較目前的對話與歷史記憶。若發現資訊有變、舊情報失效，或有全新情報，請進行記憶管理。

⚠️ 嚴格規定：在寫入或更新記憶時，**絕對不能使用「玩家」、「他」、「使用者」等代名詞**，必須明確寫出該使用者的「真實名字」。

💡 記憶擴充：寫入新記憶時，請利用你的知識庫為主詞補充「簡短背景」，讓記憶更具體。
例如：玩家說「喜歡玩APEX」，請寫入「(玩家名) 喜歡玩 APEX Legends (一款大逃殺射擊遊戲)」。

🗣️ 主動追問：若玩家提到你不熟悉的事物（未知遊戲、模糊事件），或資訊過於簡略，請自然地追問細節，引導玩家多說一點，以便建立更完整的記憶。

請務必以 **JSON 格式** 回傳，不要包含 Markdown 程式碼區塊標記。
JSON 格式必須包含：
1. "reply": 你的回應內容。
2. "memory_ops": 一個操作陣列。處理「個人偏好、瑣碎情報、特定事件」等深層碎片記憶。支援三種動作："add" (新增), "update" (更新舊記憶), "delete" (刪除)。
3. "update_whiteboard": 字串格式。當發生「群組重大進展」、「核心狀態改變」或「重要人物設定」時，請將新情報濃縮成一句話寫在這裡，系統會將其更新到核心白板 (memory.md) 中。若無重大變更，請務必留空字串 ""。

【memory_ops 範例】(處理日常瑣事與特定情報)
- 若發現新情報：[{"action": "add", "text": "Hunter 明天要考微積分 (大學數學科目)"}]
- 若玩家改口：[{"action": "update", "id": "舊記憶的ID", "text": "Toby 現在喜歡喝白色的 Monster Energy (一種無糖能量飲料)"}]
- 若無變動：請回傳空陣列 []

【update_whiteboard 範例】(處理全局與核心情報)
- "Toby (大四資訊管理系學生) 目前正在積極開發名為 Yuki 的 Discord 機器人。"
- "群組最近非常熱衷於討論 AI 圖像生成與 ComfyUI 的技術與節點報錯問題。"
- "" (如果只是一般日常閒聊，請保持空字串)
"""
user_histories = {}

def get_dynamic_system_prompt(relevant_memories: list):
    """把從資料庫撈出來的帶有 ID 與時間的記憶組合成 Prompt"""
    if not relevant_memories:
        memory_context = "（目前沒有找到相關的歷史記憶）"
    else:
        # 格式化範例： [ID: 8f2a1b] (2023-10-25 14:00) 玩家喜歡吃蘋果
        memory_context = "\n".join([f"- [ID: {m['id']}] ({m['timestamp']}) {m['text']}" for m in relevant_memories])

    return f"{BASE_SYSTEM_PROMPT}\n\n【檢索到的相關歷史記憶 (越上面的越新)】：\n{memory_context}"


def extract_json_from_response(text: str):
    """抓最後一個可 json.loads 的物件，並抽 reply/memo"""
    if not text:
        return "", None

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.replace("<think>", "").replace("</think>", "").strip()

    decoder = json.JSONDecoder()
    last_obj = None
    idx = 0

    # ✅ 關鍵修復：同時支援物件 '{' 和陣列 '['
    while idx < len(text):
        if text[idx] in ['{', '[']:
            try:
                # 嘗試從目前的 { 或 [ 開始解析 JSON
                obj, end_offset = decoder.raw_decode(text[idx:])

                if isinstance(obj, dict):
                    last_obj = obj
                elif isinstance(obj, list):
                    # 如果抓到的是純陣列，手動幫它包裝成預期的格式
                    last_obj = {"memory_ops": obj}

                idx += end_offset
                continue
            except Exception:
                pass
        idx += 1

    if not last_obj:
        print(f"⚠️ JSON 解析失敗，原始回應前 200 字: {text[:200]}")
        return "", []

    reply = last_obj.get("reply", "")
    memory_ops = last_obj.get("memory_ops", [])

    if not isinstance(reply, str):
        reply = str(reply)
    if not isinstance(memory_ops, list):
        memory_ops = []

    return reply, memory_ops

def call_api(messages_list):
    """呼叫 LM Studio API"""
    payload = {
        "model": LM_STUDIO_MODEL,
        "messages": messages_list,
        "max_tokens": MAX_TOKENS,
        "temperature": 0.7,
        "stream": False,
    }
    try:
        response = requests.post(
            LM_STUDIO_API_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=120
        )
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"].get("content", "")
    except requests.exceptions.RequestException as e:
        print(f"❌ API 連線失敗: {e}")
        return ""

def chat(user_id, user_name, user_input):
    global user_histories

    if user_id not in user_histories:
        user_histories[user_id] = []

    search_query = f"{user_name} 問：{user_input}"
    # 🔍 1. 將使用者的輸入轉成向量，並搜尋關聯記憶
    input_vector = get_embedding(search_query)

    relevant_memories = []
    if input_vector:
        relevant_memories = memory_db.search_memory(input_vector, top_k=5, threshold=0.45)

    # 2. 準備 Prompt (帶入撈出來的記憶)
    current_system_msg = {"role": "system", "content": get_dynamic_system_prompt(relevant_memories)}
    tagged_input = f"[{user_name}] 說：{user_input}"
    user_histories[user_id].append({"role": "user", "content": tagged_input})

    # (歷史長度控制保持不變...)
    if len(user_histories[user_id]) > MAX_HISTORY_ROUNDS * 2:
        user_histories[user_id] = user_histories[user_id][-(MAX_HISTORY_ROUNDS * 2):]

    messages_to_send = [current_system_msg] + user_histories[user_id]

    try:
        # 🌟 【美化 Log】傳送前的 Request 狀態
        print("\n" + "═" * 60)
        print(f"🗣️ [主動對話] 收到來自 {user_name} 的訊息")
        print(f"🤔 思考中... (參考了 {len(relevant_memories)} 筆記憶)")
        print("─" * 60)
        print("📤 [準備傳送給 LLM 的 Context (節錄)]:")
        for msg in messages_to_send:
            role = msg.get('role', 'Unknown').upper()
            content = msg.get('content', '')
            # System Prompt 通常太長，我們只印前 100 字方便確認
            if role == "SYSTEM":
                print(f"[{role}]: {content}... (略)")
            else:
                print(f"[{role}]: {content}")
        print("═" * 60 + "\n")

        raw_response = call_api(messages_to_send)

        # 🌟 【美化 Log】收到原始 Response
        print("\n" + "═" * 60)
        print("📥 [LLM 原始回覆 (Raw Response)]:")
        print(raw_response)
        print("═" * 60 + "\n")

        if not raw_response:
            return "糟糕，我跟 LM Studio 的連線好像斷了..."

        # 3. JSON 解析：取得回應文字與「新事實」
        tts_text, memory_ops = extract_json_from_response(raw_response)

        # 🌟 【美化 Log】解析結果
        print("\n" + "─" * 60)
        print(f"💬 [準備回覆給玩家]: {tts_text}")
        print(f"🧠 [解析出的記憶操作]:\n{json.dumps(memory_ops, ensure_ascii=False, indent=2)}")
        print("─" * 60 + "\n")

        # 💾 4. 根據 LLM 的判斷執行記憶的增、刪、改
        if memory_ops:
            for op in memory_ops:
                action = op.get("action")

                if action == "add" and "text" in op:
                    vector = get_embedding(op["text"])
                    if vector:
                        memory_db.add_memory(user_id, op["text"], vector)

                elif action == "update" and "id" in op and "text" in op:
                    vector = get_embedding(op["text"])
                    if vector:
                        memory_db.update_memory(user_id, op["id"], op["text"], vector)

                elif action == "delete" and "id" in op:
                    memory_db.delete_memory(op["id"])

        user_histories[user_id].append({"role": "assistant", "content": raw_response})
        return tts_text

    except Exception as e:
        print(f"❌ 系統錯誤: {e}")
        return "腦袋打結了..."


# ================= 🕵️ 被動觀察（背景記憶萃取）=================
PASSIVE_SYSTEM_PROMPT = """
你是一個負責整理情報的幕後 AI。請分析以下的 Discord 群組聊天紀錄，並萃取出參與者的個人特徵、偏好或重要情報。

系統會提供「過去的相關記憶」給你參考。請比較最新的聊天紀錄與歷史記憶，執行新增、更新或刪除。

⚠️ 嚴格規定：在萃取記憶時，**絕對不能使用「玩家」、「他」等代名詞**，必須明確寫出發言者的「統一稱呼」或「真實名字」。

💡 記憶擴充：萃取新記憶時，請盡可能為主詞補充「簡短背景」，讓記憶更具體且有利於未來檢索。
例如：聽到「喜歡玩APEX」，請寫入「(玩家名) 喜歡玩 APEX Legends (一款大逃殺射擊遊戲)」。

請務必以 **JSON 格式** 回傳，不要包含 Markdown 程式碼區塊標記。
JSON 格式必須包含：
"memory_ops": 一個操作陣列。支援 "add", "update", "delete"。
範例：
{
  "memory_ops": [
    {"action": "add", "text": "焦廣恆 喜歡玩 Delta Force (一款第一人稱戰術射擊遊戲)"},
    {"action": "update", "id": "舊記憶的ID", "text": "焦廣恆 其實不喜歡上學 (學生生活)"},
    {"action": "delete", "id": "舊記憶的ID"}
  ]
}
如果沒有任何新情報或需要修改的記憶，請回傳 {"memory_ops": []}。
"""


def extract_passive_memory(user_id, user_name, chat_log):
    """回傳 memory_ops 並執行 CRUD (增刪改查)"""

    # 🔍 1. 先用近期的對話紀錄去搜尋資料庫，看看有沒有相關的舊記憶
    log_vector = get_embedding(chat_log)
    relevant_memories = []
    if log_vector:
        relevant_memories = memory_db.search_memory(log_vector, top_k=5, threshold=0.45)

    # 2. 組合背景記憶上下文
    if not relevant_memories:
        memory_context = "（目前沒有找到相關的歷史記憶）"
    else:
        memory_context = "\n".join([f"- [ID: {m['id']}] ({m['timestamp']}) {m['text']}" for m in relevant_memories])

    # 3. 準備送給 LLM 的訊息
    messages_to_send = [
        {"role": "system", "content": PASSIVE_SYSTEM_PROMPT},
        {"role": "user",
         "content": f"【檢索到的相關歷史記憶】：\n{memory_context}\n\n【近期對話紀錄】（{user_name} 說）：\n{chat_log}"}
    ]

    try:
        # 🌟 【美化 Log】背景觀察觸發
        print("\n" + "═" * 60)
        print(f"🕵️ [背景觀察觸發] 開始分析 {user_name} 的近期閒聊...")
        print("📤 [餵給 LLM 的對話紀錄]:\n" + chat_log)
        print("═" * 60 + "\n")

        raw_response = call_api(messages_to_send)
        if not raw_response: return

        # 🌟 【美化 Log】背景觀察的原始輸出
        print("\n" + "═" * 60)
        print("📥 [背景觀察 LLM 原始回覆]:")
        print(raw_response)
        print("═" * 60 + "\n")

        # 4. 抓取 JSON
        _, memory_ops = extract_json_from_response(raw_response)

        # 💾 5. 全面支援增、刪、改
        if memory_ops:
            # 🌟 【美化 Log】漂亮的 JSON 輸出
            print("\n" + "─" * 60)
            print(f"🕵️ [背景觀察] 決定執行以下記憶操作：\n{json.dumps(memory_ops, ensure_ascii=False, indent=2)}")
            print("─" * 60 + "\n")

            for op in memory_ops:
                action = op.get("action")

                if action == "add" and "text" in op:
                    fact_vector = get_embedding(op["text"])
                    if fact_vector:
                        memory_db.add_memory(user_id, op["text"], fact_vector)

                elif action == "update" and "id" in op and "text" in op:
                    fact_vector = get_embedding(op["text"])
                    if fact_vector:
                        memory_db.update_memory(user_id, op["id"], op["text"], fact_vector)

                elif action == "delete" and "id" in op:
                    memory_db.delete_memory(op["id"])
        else:
            print("🕵️ [背景觀察] 判定為無意義閒聊，沒有更新任何情報。")

    except Exception as e:
        print(f"❌ 背景萃取錯誤: {e}")

def get_embedding(text: str):
    """
    呼叫 LM Studio Embeddings API，將輸入的文字轉換為向量 (Vector)。
    回傳: list[float] (若失敗則回傳 None)
    """
    if not text or not text.strip():
        print("⚠️ 輸入文字為空，無法產生 Embedding")
        return None

    payload = {
        "model": LM_STUDIO_EMBEDDING_MODEL,
        "input": text
    }

    try:
        response = requests.post(
            LM_STUDIO_EMBEDDING_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=60  # Embedding 通常很快，60秒很夠了
        )
        response.raise_for_status()
        result = response.json()

        # 根據 OpenAI 格式，向量會放在 data 陣列的第一個物件的 "embedding" 屬性中
        embedding_vector = result["data"][0]["embedding"]
        return embedding_vector

    except requests.exceptions.RequestException as e:
        print(f"❌ Embedding API 連線失敗: {e}")
        return None
    except KeyError as e:
        print(f"❌ Embedding API 回傳格式錯誤，找不到鍵值: {e}")
        return None