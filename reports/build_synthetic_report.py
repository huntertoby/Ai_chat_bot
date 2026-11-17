"""產生「5 年假記憶模擬測試」PDF 報告：圖表（matplotlib）+ 排版（reportlab）。

數字全部來自 data/synthetic/ 的評估紀錄與這次實驗過程，寫死在 DATA 區，方便對照修改。
需要 reportlab、matplotlib（不在專案 requirements 裡，另外的 venv 跑即可）。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_CENTER  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.pdfbase import pdfmetrics  # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont  # noqa: E402
from reportlab.platypus import (Image, KeepTogether, Paragraph, SimpleDocTemplate,  # noqa: E402
                                Spacer, Table, TableStyle)

OUT = Path(__file__).resolve().parent
FIG = OUT / "figs"
FIG.mkdir(exist_ok=True)

# ---------- 配色（dataviz 參考調色盤：前三個類別色可兩兩區分，含色盲） ----------
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

# ---------- 字型 ----------
FONT_DIR = Path("C:/Windows/Fonts")
pdfmetrics.registerFont(TTFont("JH", str(FONT_DIR / "msjh.ttc"), subfontIndex=0))
pdfmetrics.registerFont(TTFont("JHB", str(FONT_DIR / "msjhbd.ttc"), subfontIndex=0))
font_manager.fontManager.addfont(str(FONT_DIR / "msjh.ttc"))
plt.rcParams.update({
    "font.family": font_manager.FontProperties(fname=str(FONT_DIR / "msjh.ttc")).get_name(),
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white",
    "axes.facecolor": "white", "font.size": 10,
})

# ================= DATA =================
PIPELINE = [  # (設定, Hit@6 %, 檢索延遲中位數)
    ("混合檢索（dense + BM25 + RRF）", 69.0, "0.06 秒"),
    ("+ 時間感知", 76.0, "0.08 秒"),
    ("+ cross-encoder rerank", 78.5, "0.11 秒"),
    ("+ rerank + 時間感知", 82.0, "0.15 秒"),
    ("+ LLM 重排（目前預設）", 84.0, "1.0 秒"),
    ("+ LLM 重排 + 查詢改寫", 84.0, "2.8 秒"),
]
TIME_SPLIT = {  # 硬過濾 vs 不用時間（有 rerank）
    "時間說對了（99 題）": (89, 83),
    "時間說錯了（38 題）": (53, 79),
}
POOL = [(20, 88, 82), (50, 91, 82), (100, 92, 82)]  # (候選池大小, 正解在池內 %, 最後進前 6 %)


# ================= 圖表 =================
def fig_pipeline() -> Path:
    labels = [p[0] for p in PIPELINE][::-1]
    vals = [p[1] for p in PIPELINE][::-1]
    lat = [p[2] for p in PIPELINE][::-1]
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    bars = ax.barh(labels, vals, color=BLUE, height=0.55)
    for b, v, l in zip(bars, vals, lat):
        ax.text(v + 0.6, b.get_y() + b.get_height() / 2, f"{v:g}%   （{l}）", va="center",
                color=INK, fontsize=9)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Hit@6（200 題，前 6 筆含正確記憶的比例）")
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    path = FIG / "pipeline.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def fig_time_split() -> Path:
    cats = list(TIME_SPLIT)
    hard = [TIME_SPLIT[c][0] for c in cats]
    none = [TIME_SPLIT[c][1] for c in cats]
    x = range(len(cats))
    w = 0.34
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    b1 = ax.bar([i - w / 2 - 0.01 for i in x], hard, w, color=BLUE, label="硬過濾（只搜那段期間）")
    b2 = ax.bar([i + w / 2 + 0.01 for i in x], none, w, color=ORANGE, label="不用時間")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, f"{b.get_height():g}%",
                    ha="center", color=INK, fontsize=9)
    ax.set_xticks(list(x), cats)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Hit@6 (%)")
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=2, fontsize=9)
    fig.tight_layout()
    path = FIG / "time_split.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def fig_pool() -> Path:
    xs = [p[0] for p in POOL]
    fig, ax = plt.subplots(figsize=(6.0, 2.8))
    ax.plot(xs, [p[1] for p in POOL], color=BLUE, linewidth=2, marker="o", markersize=7,
            label="正解在候選池內")
    ax.plot(xs, [p[2] for p in POOL], color=ORANGE, linewidth=2, marker="o", markersize=7,
            label="最後進到前 6 名")
    for x, a, b in POOL:
        ax.text(x, a + 1.2, f"{a}%", ha="center", color=INK, fontsize=9)
        ax.text(x, b - 3.2, f"{b}%", ha="center", color=INK, fontsize=9)
    ax.set_xticks(xs, [f"top-{x}" for x in xs])
    ax.set_ylim(70, 100)
    ax.set_ylabel("%")
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    fig.tight_layout()
    path = FIG / "pool.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


# ================= 排版 =================
S = {
    "title": ParagraphStyle("title", wordWrap="CJK", fontName="JHB", fontSize=22, leading=30, textColor=INK, spaceAfter=4),
    "subtitle": ParagraphStyle("subtitle", wordWrap="CJK", fontName="JH", fontSize=11, leading=16, textColor=INK2,
                               spaceAfter=14),
    "h1": ParagraphStyle("h1", wordWrap="CJK", keepWithNext=1, fontName="JHB", fontSize=15, leading=22, textColor=INK, spaceBefore=14,
                         spaceAfter=6),
    "h2": ParagraphStyle("h2", wordWrap="CJK", keepWithNext=1, fontName="JHB", fontSize=12, leading=18, textColor=INK, spaceBefore=8,
                         spaceAfter=4),
    "body": ParagraphStyle("body", wordWrap="CJK", fontName="JH", fontSize=10, leading=16.5, textColor=INK, spaceAfter=5),
    "bullet": ParagraphStyle("bullet", wordWrap="CJK", fontName="JH", fontSize=10, leading=16, textColor=INK,
                             leftIndent=12, bulletIndent=2, spaceAfter=2),
    "caption": ParagraphStyle("caption", wordWrap="CJK", fontName="JH", fontSize=8.5, leading=13, textColor=INK2,
                              alignment=TA_CENTER, spaceAfter=10),
    "cell": ParagraphStyle("cell", wordWrap="CJK", fontName="JH", fontSize=9, leading=13, textColor=INK),
    "cellb": ParagraphStyle("cellb", wordWrap="CJK", fontName="JHB", fontSize=9, leading=13, textColor=INK),
    "callout": ParagraphStyle("callout", wordWrap="CJK", fontName="JH", fontSize=10, leading=16, textColor=INK),
}


def P(text, style="body"):
    return Paragraph(text, S[style])


def bullets(items):
    return [Paragraph(t, S["bullet"], bulletText="•") for t in items]


def table(rows, widths, header=True, highlight_row=None):
    data = [[P(c, "cellb" if (header and i == 0) else "cell") for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    style = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.HexColor(INK2)),
        ("LINEBELOW", (0, 1), (-1, -1), 0.4, colors.HexColor(GRID)),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f0ec")))
    if highlight_row is not None:
        style.append(("BACKGROUND", (0, highlight_row), (-1, highlight_row), colors.HexColor("#eaf2fc")))
    t.setStyle(TableStyle(style))
    return t


def callout(text):
    t = Table([[P(text, "callout")]], colWidths=[170 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f4f8fd")),
        ("LINEBEFORE", (0, 0), (0, -1), 3, colors.HexColor(BLUE)),
        ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    return t


def figure(path, width_mm, caption):
    img = Image(str(path))
    ratio = img.imageHeight / img.imageWidth
    img.drawWidth, img.drawHeight = width_mm * mm, width_mm * mm * ratio
    return KeepTogether([img, P(caption, "caption")])


def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("JH", 8)
    canvas.setFillColor(colors.HexColor(INK2))
    canvas.drawString(20 * mm, 12 * mm, "Yuki 長期記憶 RAG：5 年假記憶模擬測試報告")
    canvas.drawRightString(190 * mm, 12 * mm, f"{doc.page}")
    canvas.restoreState()


def build() -> Path:
    f_pipe, f_time, f_pool = fig_pipeline(), fig_time_split(), fig_pool()
    out = OUT / "yuki_synthetic_memory_report.pdf"
    doc = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm,
                            topMargin=18 * mm, bottomMargin=20 * mm,
                            title="Yuki 長期記憶 RAG：5 年假記憶模擬測試報告", author="Yuki 專案")
    st = []

    # ---------- 封面摘要 ----------
    st += [P("Yuki 長期記憶 RAG<br/>5 年假記憶模擬測試報告", "title"),
           P("測試日期 2026-10-29 ～ 10-30　｜　資料：10 位虛構成員 × 5 年 × 每天一筆 = 18,270 筆記憶", "subtitle")]
    # ---------- 1. 目標 ----------
    st.append(P("1. 目標", "h1"))
    st.append(P("原本的目標是直接用真實群組的資料，測試 Yuki 的長期記憶檢索在實際使用下的表現。"
                "但真實資料不適合拿來做第一輪評估，最後決定先用<b>模擬資料</b>測試："))
    st += bullets([
        "<b>沒有標準答案</b>：真實聊天沒有「這個問題應該找到哪一筆記憶」的標註，無法自動計算命中率，只能逐題人工判斷。",
        "<b>原始群組資料不夠嚴謹</b>：群組裡大多是打屁哈啦，很多內容無從考究，難以判斷檢索結果是對是錯。",
        "<b>時間情境無法控制</b>：要測「昨天」「上個月」這類問題，需要每天都有紀錄，而且確切知道每筆發生在哪一天。",
        "<b>隱私</b>：拿成員的真實對話反覆出題、跑實驗，需要先取得大家同意。",
    ])
    st.append(P("模擬資料可以一次解決這些問題：10 位虛構成員 × 5 年 × 每天一筆記憶，每一筆都知道主角和日期，"
                "可以自動出題、量化比較。這次測試要回答三個問題："))
    st += bullets([
        "記憶累積到 1.8 萬筆時，檢索還找得到正確的記憶嗎？rerank、時間感知、LLM 重排各自貢獻多少？",
        "延遲能不能接受？",
        "接上 Discord 實際對話時，Yuki 的回答是否正確？",
    ])

    # ---------- 2. 系統架構 ----------
    st.append(P("2. 受測系統", "h1"))
    st.append(P("Yuki 是有長期記憶的 Discord 聊天機器人。回覆前先從記憶庫檢索相關記憶，再交給本地 LLM 生成回覆。"))
    st.append(table([
        ["元件", "使用的模型／做法"],
        ["聊天與記憶整理 LLM", "swift-1.5-qwen3.8-27b（GGUF Q4_K_XL，MTP），LM Studio，同時只能處理 1 個請求"],
        ["假資料產生 LLM", "Swift-Qwen3.8-27B NVFP4，vLLM（WSL），最多 8 個請求並行"],
        ["Embedding", "BAAI/bge-m3（本機 GPU）"],
        ["Rerank", "BAAI/bge-reranker-v2-m3（cross-encoder，本機 GPU）"],
        ["稀疏檢索", "BM25 + jieba 斷詞"],
        ["儲存", "SQLite + 記憶體內向量矩陣（numpy 暴力計算餘弦相似度）"],
    ], [45 * mm, 125 * mm]))
    st.append(Spacer(1, 6))
    st.append(P("檢索流程", "h2"))
    st += bullets([
        "<b>查詢分解</b>：依人物拆成子查詢（「大熊跟 Momo 昨天在幹嘛」→ 兩個子查詢），每個子查詢之後各自保有名額。",
        "<b>混合檢索</b>：每個子查詢 dense 與 BM25 各取前 20，用 RRF（k=60）合併。",
        "<b>時間感知</b>：規則式解析時間說法，日期吻合的記憶在候選與 rerank 階段加分（見第 5 節）。",
        "<b>Rerank</b>：cross-encoder 打分，低於 0.05 的丟掉。",
        "<b>LLM 重排</b>：前 10 筆連同日期與今天日期交給 LLM 挑選、排序（約 +1 秒）。",
        "<b>輪流取</b>：各子查詢輪流取，共 6 筆，放進回覆的 prompt，每筆附日期與「距今多久」。",
    ])

    # ---------- 3. 假資料 ----------
    st.append(P("3. 假資料產生", "h1"))
    st.append(P("用 LLM 產生 10 位虛構成員（深夜泡麵王、阿寶、Momo、小綠、老K、雪見、大熊、Luna、阿健、小喵），"
                "期間 2021-10-29 ～ 2026-10-29，每人每天一筆記憶，共 <b>10 × 1,827 = 18,270 筆</b>，沒有缺任何一天。"))
    st += bullets([
        "<b>三層產生</b>：成員設定 → 每人每年 6 個主線事件（例如搬家、受傷、出國）→ 每人每月一次呼叫產生整個月的每日記憶，"
        "並帶入上個月最後幾筆以保持連貫。",
        "<b>品質把關</b>：一次呼叫少於該月天數的 80% 就重試；記憶沒有人名開頭就自動補上。",
        "<b>已知瑕疵</b>：逐月產生看不到完整歷史，所以有少數矛盾（深夜泡麵王「搬到內湖」出現兩次）"
        "和完全相同的句子。真實聊天由記憶寫入器去重、合併，不會有這個問題。",
    ])
    st.append(table([
        ["產生方式", "速度", "說明"],
        ["LM Studio（MTP，一次 1 個請求）", "約 94 筆／分鐘", "每次呼叫約 18 秒，10 人排隊"],
        ["vLLM NVFP4（8 個請求並行）", "約 300 筆／分鐘", "單次 12.7 秒；8 個並行共 17.7 秒"],
    ], [62 * mm, 32 * mm, 76 * mm]))
    st.append(Spacer(1, 4))
    st.append(P("前 7,633 筆用 LM Studio 產生，剩下約 1.06 萬筆改用 vLLM，花了 33 分鐘。"
                "匯入（bge-m3 算向量）18,270 筆花了 86 秒。", "body"))

    # ---------- 4. 評估方法 ----------
    st.append(P("4. 評估方法", "h1"))
    st += bullets([
        "從記憶庫隨機抽 200 筆，請 LLM 以另一位成員的身分寫一個口語問題，這筆記憶就是標準答案。",
        "每題設定一個「發問日期」（記憶發生後 1～300 天），讓「昨天」「上個月」等說法有對應的日期。",
        "指標 <b>Hit@6</b>：最後放進 prompt 的 6 筆記憶中，有沒有包含標準答案。",
        "200 題的抽樣誤差約 <b>±3 個百分點</b>，小於這個幅度的差異不算可靠的結論。",
    ])
    st.append(P("第一版題目沒有設定發問日期，「昨天」會被當成實際的今天（2026-10-29）解讀，"
                "所以不能拿來評估時間功能。後來重新產生了帶發問日期的題目，本報告的數字都來自新題目。"))


    # ---------- 5. 結果 ----------
    st.append(P("5. 結果", "h1"))
    st.append(figure(f_pipe, 165, "圖 1　各設定的 Hit@6 與檢索延遲（中位數，不含生成回覆）"))
    st.append(table([
        ["設定", "全部 200 題", "有時間說法（137 題）", "沒有時間說法（63 題）", "檢索延遲"],
        ["rerank + 時間感知", "82%", "87%", "71%", "0.15 秒"],
        ["+ 查詢改寫", "82%", "86%", "75%", "1.7 秒"],
        ["+ LLM 重排（預設）", "84%", "90%", "71%", "1.0 秒"],
        ["+ 兩個都開", "84%", "89%", "73%", "2.8 秒"],
    ], [42 * mm, 26 * mm, 36 * mm, 36 * mm, 30 * mm], highlight_row=3))
    st.append(Spacer(1, 4))
    st.append(P("逐題比對「只開 LLM 重排」和「兩個都開」：各有 4 題只有其中一邊答對，等於打平。"
                "查詢改寫多花 1.8 秒卻沒有幫助，所以預設關閉。"))

    st.append(P("5.1 時間感知：加分，不要過濾", "h2"))
    st.append(P("第一版做法是硬過濾：問題說「昨天」就只搜昨天的記憶。結果整體反而從 78% 降到 76%。"
                "拆開來看，137 題有時間說法的題目中，有 38 題（28%）講的時間跟記憶實際日期對不上"
                "（例如 6 天前的事說成「前天」）。"))
    st.append(figure(f_time, 140, "圖 2　硬過濾在時間說對與說錯時的表現（有 rerank）"))
    st.append(P("改成加分制之後：日期吻合的記憶在候選階段多一份排名、rerank 分數 +0.1，但期間外的記憶不會被排除。"
                "Hit@6 從 78.5% 升到 82%。加分強度試過 0.05～0.3，0.1 以上結果相同，取最保守的 0.1。"))

    st.append(P("5.2 瓶頸診斷：排序，而不是召回", "h2"))
    st.append(figure(f_pool, 130, "圖 3　放大候選池：召回提升，但最後結果不變"))
    st.append(P("剩下 18% 沒答對的題目分成兩種：約 6% 正解有進候選池但排不進前 6（同一個人每天做很像的事），"
                "約 8～12% 根本沒撈到（問題太口語、太模糊）。前者正是 LLM 重排要解決的；後者只靠放大候選池沒有用。"))

    # ---------- 6. 評估中發現的問題 ----------
    st.append(P("6. 評估中發現並修正的問題", "h1"))
    st.append(table([
        ["問題", "症狀", "修正"],
        ["推理模型把 token 用光", "查詢改寫 200 次有 174 次沒有輸出；數字跟沒開一樣，卻多花 4.7 秒",
         "簡單的結構化任務（改寫、重排、查詢分解）一律關閉思考"],
        ["編號從 0 還是 1", "模型習慣從 1 開始編號，程式從 0 開始讀：改寫被丟掉、重排挑錯記憶",
         "prompt 改成從 1 開始編號"],
        ["MTP 模型不能並行", "同時送多個請求時 LM Studio 回傳 400 錯誤，產生資料中斷",
         "LLM 客戶端加上並行上限（預設 1）"],
        ["上萬筆時寫入變慢", "每次寫入重建索引要重新斷詞，約 3.8 秒", "斷詞結果快取，重建降到 0.08 秒"],
    ], [38 * mm, 70 * mm, 62 * mm]))

    # ---------- 7. Discord 實測 ----------
    st.append(P("7. Discord 實測", "h1"))
    st.append(P("把 18,270 筆假記憶複製到獨立的測試資料庫，指定給一個測試伺服器，讓 Yuki 用真實流程回答（2026-10-30）。"))
    st.append(table([
        ["問題", "Yuki 的回答（摘要）", "結果"],
        ["大熊昨天在幹嘛", "在台中家附近的公園散步，打算週末帶媽媽去花市", "答對"],
        ["Momo 昨天做了什麼", "在家組了一台新電腦，邊裝邊看科技 YouTube", "答對"],
        ["小喵 2023年3月在做什麼音樂", "忙《靜默聲學》EP，確認母帶細節，三月底發行", "答對"],
        ["老K 最近頸椎還好嗎", "上次物理治療時說有明顯改善（實際是 4 個月前）", "部分"],
        ["大熊跟 Momo 昨天各自在幹嘛", "兩人都有回答到（各子查詢保有名額）", "答對"],
        ["深夜泡麵王搬到哪了", "內湖，板南線內湖站附近的迷你公寓", "答對"],
    ], [52 * mm, 103 * mm, 15 * mm]))
    st.append(Spacer(1, 4))
    st.append(P("第 4 題的內容正確，但把 4 個月前的事講得像最近發生。修正方式：prompt 中每筆記憶附上程式算好的"
                "「距今多久」（例如「約 4 個月前」），並要求超過一個月的事要講出大概時間。"))

    # ---------- 8. 限制與下一步 ----------
    st.append(P("8. 已知限制", "h1"))
    st += bullets([
        "<b>模糊問題</b>：沒有時間說法的題目卡在 71～75%，是目前最弱的類型。",
        "<b>「最早」「第一次」類問題會查不到</b>：例如「老K 頸椎病最早是什麼時候」，cross-encoder 給所有記憶的分數都低於門檻"
        "（最高 0.037），結果 0 筆。答案是記憶的日期，而日期不在內文裡。需要另外的處理：rerank 時去掉問法只留主題，"
        "或偵測這類問題後依日期排序。",
        "<b>計數類問題</b>（「XXX 說我壞話幾次」）不能靠前幾筆檢索，需要對所有相關訊息逐則判斷、由程式計數。",
        "<b>評估規模</b>：200 題的誤差約 ±3 點，要比較更細的改動需要 500 題以上。",
        "<b>題目來源偏差</b>：題目由 LLM 依記憶內容寫成，用字常跟記憶很接近，真實使用者的問法可能更難。",
    ])


    # ---------- 9. 結論 ----------
    st.append(P("9. 結論", "h1"))
    st.append(callout(
        "在 18,270 筆記憶中，最終管線（混合檢索 + rerank + 時間感知 + LLM 重排）"
        "前 6 筆找到正確記憶的比例為 <b>84%</b>，起點的純混合檢索是 69%。檢索本身只要約 0.15 秒，"
        "LLM 重排再加約 1 秒。Discord 實測 6 題中 5 題答對、1 題部分正確（已修正）。"))
    st.append(Spacer(1, 8))
    st += bullets([
        "<b>時間感知</b>（「昨天」「上個月」等）是單項最大的進步，但必須用「加分」而不是「過濾」："
        "28% 的問題講的時間其實不精確，硬過濾會把正解直接排除。",
        "<b>瓶頸在排序，不在召回</b>：候選池從 20 放大到 100，正解在池內的比例從 88% 升到 92%，"
        "最後結果卻完全沒變（82%）。同一個人每天做很像的事，cross-encoder 看不懂日期，分不出是哪一天。",
        "<b>LLM 重排</b>（把前 10 筆連同日期交給聊天模型挑）專門解決上一點，有時間說法的題目從 87% 提升到 90%。"
        "<b>查詢改寫</b>跟它一起開沒有額外提升，預設關閉。",
        "評估過程抓到兩個會讓功能完全失效的 bug：推理模型把 token 用光、模型從 1 開始編號。修正前的結果不能採信。",
    ])

    st.append(P("模擬資料證明了檢索管線在 1.8 萬筆規模下可用，也找出了只有在大量、每天都很相似的資料裡才會出現的問題。"
                "這些結論還需要用真實群組的資料驗證。"))

    doc.build(st, onFirstPage=on_page, onLaterPages=on_page)
    return out


if __name__ == "__main__":
    print(build())
