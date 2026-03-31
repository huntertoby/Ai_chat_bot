# memory_db.py
import chromadb
import uuid
from datetime import datetime

# 1. 初始化 ChromaDB
client = chromadb.PersistentClient(path="./chroma_memory")

# 2. 建立或讀取集合
collection = client.get_or_create_collection(
    name="yuki_user_memories",
    metadata={"hnsw:space": "cosine"}
)


def get_current_time():
    """取得當前時間字串，方便 LLM 閱讀"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ✅ 補回遺失的 add_memory
def add_memory(user_id: str, fact_text: str, vector: list):
    """將單一事實與向量存入資料庫，並加上時間戳記"""
    mem_id = str(uuid.uuid4())[:8]  # 截斷 UUID 讓 LLM 比較好抄寫，節省 Token
    timestamp = get_current_time()

    collection.add(
        embeddings=[vector],
        documents=[fact_text],
        metadatas=[{"user_id": user_id, "timestamp": timestamp}],
        ids=[mem_id]
    )
    print(f"💾 [記憶新增] {user_id}: {fact_text} (ID: {mem_id})")


def update_memory(user_id: str, mem_id: str, new_text: str, new_vector: list):
    """更新現有記憶"""
    timestamp = get_current_time()
    collection.update(
        ids=[mem_id],
        embeddings=[new_vector],
        documents=[new_text],
        metadatas=[{"user_id": user_id, "timestamp": timestamp}]
    )
    print(f"🔄 [記憶更新] {user_id} - ID:{mem_id} -> {new_text}")


def delete_memory(mem_id: str):
    """刪除指定記憶"""
    collection.delete(ids=[mem_id])
    print(f"🗑️ [記憶刪除] 移除了 ID: {mem_id}")


# 修改 memory_db.py 中的 search_memory 函數
def search_memory(query_vector: list, top_k: int = 5, threshold: float = 0.5):
    """根據向量進行全域搜尋，並過濾掉距離大於 threshold 的無關記憶"""
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        # ✅ 關鍵：加上 "distances" 讓 ChromaDB 回傳距離分數
        include=["documents", "metadatas", "distances"]
    )

    formatted_memories = []
    if results and results["documents"] and len(results["documents"][0]) > 0:
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        ids = results["ids"][0]
        distances = results["distances"][0]

        for i in range(len(docs)):
            # ✅ 核心邏輯：距離小於門檻才算數 (數字越小越相似)
            if distances[i] <= threshold:
                formatted_memories.append({
                    "id": ids[i],
                    "text": docs[i],
                    "timestamp": metas[i].get("timestamp", "未知時間"),
                    "distance": distances[i] # 順便記錄距離方便 Debug
                })

        # 根據時間排序 (最新的在前面)
        formatted_memories.sort(key=lambda x: x["timestamp"], reverse=True)

    return formatted_memories


def get_all_memories():
    """取得群組的所有記憶，並依時間由新到舊排序"""
    results = collection.get(
        include=["documents", "metadatas"]
    )

    formatted_memories = []
    if results and results["documents"]:
        docs = results["documents"]
        metas = results["metadatas"]
        ids = results["ids"]

        for i in range(len(docs)):
            formatted_memories.append({
                "id": ids[i],
                "text": docs[i],
                "timestamp": metas[i].get("timestamp", "未知時間")
            })

        # 根據時間排序 (最新的在前面)
        formatted_memories.sort(key=lambda x: x["timestamp"], reverse=True)

    return formatted_memories