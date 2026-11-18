"""所有設定都從 .env / 環境變數讀取，程式裡不再寫死 IP 或模型名稱。"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _str(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _int(name: str, default: int) -> int:
    return int(_str(name, str(default)))


def _float(name: str, default: float) -> float:
    return float(_str(name, str(default)))


def _bool(name: str, default: bool) -> bool:
    return _str(name, str(default)).lower() in ("1", "true", "yes", "on")


def _opt_float(name: str) -> float | None:
    v = _str(name)
    return float(v) if v else None


def _list(name: str, default: str) -> list[str]:
    return [s.strip() for s in _str(name, default).split(",") if s.strip()]


@dataclass(frozen=True)
class Settings:
    discord_token: str = field(default_factory=lambda: _str("DISCORD_TOKEN"))
    bot_name: str = field(default_factory=lambda: _str("BOT_NAME", "Yuki"))
    # 訊息裡出現這些字就算叫到 Yuki（不分大小寫）
    wake_words: list[str] = field(default_factory=lambda: _list("WAKE_WORDS", "yuki"))
    exit_words: list[str] = field(default_factory=lambda: _list("EXIT_WORDS", "掰掰,再見,退下,去睡覺,閉嘴"))

    # ---- LM Studio（OpenAI 相容 API）----
    lmstudio_base_url: str = field(default_factory=lambda: _str("LMSTUDIO_BASE_URL", "http://192.168.0.75:1234/v1"))
    # 留空 = 自動抓 LM Studio 目前第一個非 embedding 模型
    llm_model: str = field(default_factory=lambda: _str("LLM_MODEL"))
    llm_temperature: float = field(default_factory=lambda: _float("LLM_TEMPERATURE", 0.7))
    llm_max_tokens: int = field(default_factory=lambda: _int("LLM_MAX_TOKENS", 2048))
    llm_timeout: float = field(default_factory=lambda: _float("LLM_TIMEOUT", 180))
    # 同時送給 LM Studio 的請求上限。MTP / speculative decoding 模型同時收到兩個請求時
    # LM Studio 會出錯（speculative batch index ... not inside the current sub-batch），所以預設 1
    llm_max_concurrency: int = field(default_factory=lambda: _int("LLM_MAX_CONCURRENCY", 1))
    # 推理模型（Qwen3 系列等）會先想再答，想太久會很慢甚至逾時。
    # none / low / medium / high；留空則不送這個參數（非推理模型或其他後端）
    llm_reasoning_effort: str = field(default_factory=lambda: _str("LLM_REASONING_EFFORT", "low"))
    llm_json_reasoning_effort: str = field(default_factory=lambda: _str("LLM_JSON_REASONING_EFFORT", "low"))

    # ---- Embedding ----
    # local = sentence-transformers 在本機跑（預設 bge-m3，中文最好）
    # lmstudio = 呼叫 LM Studio 的 /v1/embeddings
    embed_backend: str = field(default_factory=lambda: _str("EMBED_BACKEND", "local"))
    embed_model: str = field(default_factory=lambda: _str("EMBED_MODEL", "BAAI/bge-m3"))
    # 留空時依模型名稱自動決定（nomic / qwen3 需要前綴，bge-m3 不用）
    embed_query_prefix: str | None = field(default_factory=lambda: os.getenv("EMBED_QUERY_PREFIX"))
    embed_doc_prefix: str | None = field(default_factory=lambda: os.getenv("EMBED_DOC_PREFIX"))

    # ---- Rerank ----
    rerank_enabled: bool = field(default_factory=lambda: _bool("RERANK_ENABLED", True))
    rerank_model: str = field(default_factory=lambda: _str("RERANK_MODEL", "BAAI/bge-reranker-v2-m3"))
    # 4bit / 8bit（需要 bitsandbytes）/ none；主要給 Qwen3-Reranker 這類大模型用
    rerank_quantize: str = field(default_factory=lambda: _str("RERANK_QUANTIZE", "none"))
    # 給 Qwen3-Reranker 的任務說明（英文效果較好）；留空用預設；bge-reranker 不使用
    rerank_instruction: str = field(default_factory=lambda: _str("RERANK_INSTRUCTION", ""))
    # rerank 分數（sigmoid 後 0~1）低於此值視為不相關，直接丟掉
    rerank_min_score: float = field(default_factory=lambda: _float("RERANK_MIN_SCORE", 0.05))
    time_boost: float = field(default_factory=lambda: _float("TIME_BOOST", 0.1))
    rerank_focus: bool = field(default_factory=lambda: _bool("RERANK_FOCUS", True))
    rerank_fallback: int = field(default_factory=lambda: _int("RERANK_FALLBACK", 2))
    order_top_n: int = field(default_factory=lambda: _int("ORDER_TOP_N", 60))
    device: str = field(default_factory=lambda: _str("DEVICE", "auto"))
    # embedding / reranker 用 fp16 載入：顯存省一半（LM Studio 在同一張卡上），檢索結果幾乎一樣
    model_half: bool = field(default_factory=lambda: _bool("MODEL_HALF", True))

    # ---- 檢索 ----
    retrieve_top_n: int = field(default_factory=lambda: _int("RETRIEVE_TOP_N", 20))
    retrieve_final_k: int = field(default_factory=lambda: _int("RETRIEVE_FINAL_K", 6))
    rrf_k: int = field(default_factory=lambda: _int("RRF_K", 60))
    decompose_with_llm: bool = field(default_factory=lambda: _bool("DECOMPOSE_WITH_LLM", True))
    max_subqueries: int = field(default_factory=lambda: _int("MAX_SUBQUERIES", 4))
    query_rewrite: bool = field(default_factory=lambda: _bool("QUERY_REWRITE", False))
    # time = 只在問題有時間說法時用 LLM 重排；always / off；舊設定 true / false 視同 always / off
    llm_rerank: str = field(default_factory=lambda: {"true": "always", "false": "off"}.get(
        _str("LLM_RERANK", "time").lower(), _str("LLM_RERANK", "time").lower()))
    llm_rerank_top: int = field(default_factory=lambda: _int("LLM_RERANK_TOP", 10))

    # ---- 記憶寫入 ----
    dedup_similarity: float = field(default_factory=lambda: _float("DEDUP_SIMILARITY", 0.90))
    max_ops_per_batch: int = field(default_factory=lambda: _int("MAX_OPS_PER_BATCH", 5))
    max_deletes_per_batch: int = field(default_factory=lambda: _int("MAX_DELETES_PER_BATCH", 2))
    whiteboard_max_items: int = field(default_factory=lambda: _int("WHITEBOARD_MAX_ITEMS", 20))

    # ---- 對話 ----
    attention_timeout: int = field(default_factory=lambda: _int("ATTENTION_TIMEOUT", 180))
    history_messages: int = field(default_factory=lambda: _int("HISTORY_MESSAGES", 20))
    passive_enabled: bool = field(default_factory=lambda: _bool("PASSIVE_ENABLED", True))
    passive_batch: int = field(default_factory=lambda: _int("PASSIVE_BATCH", 8))
    passive_idle_seconds: int = field(default_factory=lambda: _int("PASSIVE_IDLE_SECONDS", 90))

    # ---- 人物檔案（scripts/build_profiles.py 產生，回覆時放進 prompt）----
    profile_enabled: bool = field(default_factory=lambda: _bool("PROFILE_ENABLED", True))
    profile_dir: Path = field(default_factory=lambda: ROOT / _str("PROFILE_DIR", "data/profiles"))
    profile_db_path: Path = field(default_factory=lambda: ROOT / _str("PROFILE_DB_PATH", "data/profiles.db"))
    # 一次回覆最多放幾個人的檔案（發言者、被提到的人、最近在聊天的人），每份最多幾字
    profile_max_people: int = field(default_factory=lambda: _int("PROFILE_MAX_PEOPLE", 4))
    profile_prompt_chars: int = field(default_factory=lambda: _int("PROFILE_PROMPT_CHARS", 1500))
    # 建檔用：每段聊天紀錄的字數（32k 上下文的模型 9000 字左右剛好）、每份檔案的字數上限、推理強度
    profile_segment_chars: int = field(default_factory=lambda: _int("PROFILE_SEGMENT_CHARS", 9000))
    profile_max_chars: int = field(default_factory=lambda: _int("PROFILE_MAX_CHARS", 2500))
    # 自動增量更新：所有頻道閒置幾分鐘、且封存區累積多少則新訊息才跑；有人講話就停
    profile_auto_update: bool = field(default_factory=lambda: _bool("PROFILE_AUTO_UPDATE", True))
    profile_idle_minutes: int = field(default_factory=lambda: _int("PROFILE_IDLE_MINUTES", 5))
    profile_min_new: int = field(default_factory=lambda: _int("PROFILE_MIN_NEW", 200))
    profile_reasoning_effort: str = field(default_factory=lambda: _str("PROFILE_REASONING_EFFORT", "none"))

    # ---- 上網（自架 SearXNG：docker compose -f searxng/docker-compose.yml up -d）----
    web_enabled: bool = field(default_factory=lambda: _bool("WEB_ENABLED", True))
    searxng_url: str = field(default_factory=lambda: _str("SEARXNG_URL", "http://127.0.0.1:8888"))
    # 一次回覆最多幾輪工具呼叫（搜尋 → 打開網頁 → 回答）；每輪多花幾秒
    web_max_rounds: int = field(default_factory=lambda: _int("WEB_MAX_ROUNDS", 3))
    # 每個工具結果最多給模型看幾個字（太長會擠掉記憶和對話）
    web_result_chars: int = field(default_factory=lambda: _int("WEB_RESULT_CHARS", 5000))

    # ---- 啟動補抓（Yuki 不會一直開著：每次上線先補齊上次關機到現在的訊息）----
    catchup_enabled: bool = field(default_factory=lambda: _bool("CATCHUP_ENABLED", True))
    # 「公開頻道」= 這個身分組看得到的頻道（跟 archive_fetch --role 一樣）；留空 = @everyone
    archive_role: str = field(default_factory=lambda: _str("ARCHIVE_ROLE"))
    archive_exclude: list[str] = field(default_factory=lambda: _list("ARCHIVE_EXCLUDE", ""))
    # 補抓到的訊息要不要整理成記憶；每個頻道最多整理最近幾則（離線太久時避免一上線就卡住 LM Studio）
    catchup_memory: bool = field(default_factory=lambda: _bool("CATCHUP_MEMORY", True))
    catchup_memory_max: int = field(default_factory=lambda: _int("CATCHUP_MEMORY_MAX", 300))
    # 補抓的對話中間隔超過幾分鐘就切成不同段落整理
    catchup_gap_minutes: int = field(default_factory=lambda: _int("CATCHUP_GAP_MINUTES", 30))

    # 主人（做這隻 bot 的人）：他交代的事比較會照做，但聊天不偏袒他（prompts.OWNER_RULES）
    owner_user_ids: list[str] = field(default_factory=lambda: _list("OWNER_USER_IDS", ""))

    # ---- 檢索方式（開發規格 6.4）----
    # agentic：Yuki 自己用工具查封存區（不載入 embedding / reranker，省 2.8 GB 顯存）
    # embedding：舊的每句先檢索事實記憶（bge-m3 + reranker）
    retrieval: str = field(default_factory=lambda: _str("RETRIEVAL", "embedding"))
    # 語音一次回覆最多幾輪工具呼叫（每輪 +0.6～1 秒）
    voice_max_rounds: int = field(default_factory=lambda: _int("VOICE_MAX_ROUNDS", 1))

    # ---- 測試伺服器（開發規格 6.0）----
    # 測試伺服器不跑人物檔案自動更新（測試對話不該變成誰的檔案）
    test_guild_ids: list[str] = field(default_factory=lambda: _list("TEST_GUILD_IDS", ""))
    # 找不到某人的人物檔案時，借用這個伺服器的（測試時她才認得你）
    profile_fallback_guild: str = field(default_factory=lambda: _str("PROFILE_FALLBACK_GUILD"))

    # ---- 語音對話（開發規格 6.0）----
    voice_enabled: bool = field(default_factory=lambda: _bool("VOICE_ENABLED", False))
    voice_guild_ids: list[str] = field(default_factory=lambda: _list("VOICE_GUILD_IDS", ""))
    # 只聽這些 user ID；其他人的聲音直接丟掉，不轉文字、不存
    voice_listen_users: list[str] = field(default_factory=lambda: _list("VOICE_LISTEN_USERS", ""))
    voice_log_channel: str = field(default_factory=lambda: _str("VOICE_LOG_CHANNEL"))
    voice_asr_model: str = field(default_factory=lambda: _str("VOICE_ASR_MODEL", "Qwen/Qwen3-ASR-0.6B"))
    # 給 ASR 的常用詞提示（人名會自動加）
    voice_asr_hotwords: list[str] = field(default_factory=lambda: _list(
        "VOICE_ASR_HOTWORDS", "LoL,ADC,打野,開黑,Luna,Jinx,Ezreal,patch,system prompt"))
    voice_tts: str = field(default_factory=lambda: _str("VOICE_TTS", "edge"))
    voice_tts_voice: str = field(default_factory=lambda: _str("VOICE_TTS_VOICE", "zh-TW-HsiaoChenNeural"))
    # VOICE_TTS=breezy：WSL 裡的 scripts/breezy_server.py；連不上時自動改用 Edge
    voice_tts_url: str = field(default_factory=lambda: _str("VOICE_TTS_URL", "http://127.0.0.1:8890"))
    voice_silence_ms: int = field(default_factory=lambda: _int("VOICE_SILENCE_MS", 900))
    voice_alone_leave_seconds: int = field(default_factory=lambda: _int("VOICE_ALONE_LEAVE_SECONDS", 120))
    # 有其他人在時：叫她名字後幾秒內，名單上的人說的話都回
    voice_attention_seconds: int = field(default_factory=lambda: _int("VOICE_ATTENTION_SECONDS", 60))

    # ---- 維運（開發規格 6.12）：AI 追蹤紀錄、出錯私訊 OWNER_USER_IDS ----
    trace_dir: Path = field(default_factory=lambda: ROOT / _str("TRACE_DIR", "data/logs/ai"))
    trace_max_days: int = field(default_factory=lambda: _int("TRACE_MAX_DAYS", 14))
    trace_max_mb: int = field(default_factory=lambda: _int("TRACE_MAX_MB", 200))
    alert_dedupe_min: int = field(default_factory=lambda: _int("ALERT_DEDUPE_MIN", 30))
    alert_max_per_hour: int = field(default_factory=lambda: _int("ALERT_MAX_PER_HOUR", 10))
    alert_health_sec: int = field(default_factory=lambda: _int("ALERT_HEALTH_SEC", 60))

    # ---- 看圖片（開發規格 6.9）----
    vision_enabled: bool = field(default_factory=lambda: _bool("VISION_ENABLED", True))
    vision_reply_long_side: int = field(default_factory=lambda: _int("VISION_REPLY_LONG_SIDE", 1280))
    vision_glance_long_side: int = field(default_factory=lambda: _int("VISION_GLANCE_LONG_SIDE", 512))
    vision_max_images: int = field(default_factory=lambda: _int("VISION_MAX_IMAGES", 5))
    vision_recent_lines: int = field(default_factory=lambda: _int("VISION_RECENT_LINES", 10))
    vision_max_mb: int = field(default_factory=lambda: _int("VISION_MAX_MB", 10))

    # ---- 防洗版與節流（開發規格 6.2，yuki/throttle.py）----
    throttle_cooldown_sec: float = field(default_factory=lambda: _float("THROTTLE_COOLDOWN_SEC", 5))
    throttle_tag_limit: int = field(default_factory=lambda: _int("THROTTLE_TAG_LIMIT", 3))
    throttle_tag_window_sec: float = field(default_factory=lambda: _float("THROTTLE_TAG_WINDOW_SEC", 60))
    throttle_tag_cooldown_sec: float = field(default_factory=lambda: _float("THROTTLE_TAG_COOLDOWN_SEC", 120))
    # 雜訊：去掉表情標點後 ≤ 這麼多字的訊息不判斷；大量重複：同字元連續幾次、同一人被 @ 幾次
    noise_max_chars: int = field(default_factory=lambda: _int("NOISE_MAX_CHARS", 2))
    flood_repeat_chars: int = field(default_factory=lambda: _int("FLOOD_REPEAT_CHARS", 10))
    flood_repeat_mentions: int = field(default_factory=lambda: _int("FLOOD_REPEAT_MENTIONS", 3))

    # ---- 插話判斷（開發規格 6.1，yuki/chime.py）----
    # 0 = 只有被叫才回；1 = 門檻 0.7、同頻道每 10 分鐘最多 1 次；2 = 0.6、5 分鐘；3 = 0.5、2 分鐘
    chime_in: int = field(default_factory=lambda: _int("CHIME_IN", 1))
    # 哪些伺服器開插話判斷；留空＝只有 TEST_GUILD_IDS（主伺服器等階段 D 再開）
    chime_guild_ids: list[str] = field(default_factory=lambda: _list("CHIME_GUILD_IDS", ""))
    # 留空 = 依 CHIME_IN 的等級；填了就蓋過等級的門檻
    chime_threshold: float | None = field(default_factory=lambda: _opt_float("CHIME_THRESHOLD"))
    chime_debounce_sec: float = field(default_factory=lambda: _float("CHIME_DEBOUNCE_SEC", 4))
    chime_context_lines: int = field(default_factory=lambda: _int("CHIME_CONTEXT_LINES", 15))
    chime_max_images: int = field(default_factory=lambda: _int("CHIME_MAX_IMAGES", 3))
    chime_max_tokens: int = field(default_factory=lambda: _int("CHIME_MAX_TOKENS", 160))
    # 被叫「閉嘴」「沒問你」之後，這個頻道多久不主動插話
    chime_mute_minutes: float = field(default_factory=lambda: _float("CHIME_MUTE_MINUTES", 30))
    # 按表情：每個頻道每 N 分鐘最多幾個（跟插話次數分開算）
    chime_react_max: int = field(default_factory=lambda: _int("CHIME_REACT_MAX", 3))
    chime_react_window_min: float = field(default_factory=lambda: _float("CHIME_REACT_WINDOW_MIN", 10))

    db_path: Path = field(default_factory=lambda: ROOT / _str("DB_PATH", "data/yuki.db"))
    log_level: str = field(default_factory=lambda: _str("LOG_LEVEL", "INFO"))


def load_settings() -> Settings:
    return Settings()
