"""產生「Yuki 長期記憶檢索架構」最終研究報告 PDF：只談架構、模型與設計依據，不談開發過程。

版面元件、配色、字型與結果圖沿用 build_synthetic_report.py。需要 reportlab、matplotlib。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import KeepTogether, SimpleDocTemplate, Spacer

import build_synthetic_report as base
from build_synthetic_report import (AQUA, BLUE, FIG, INK, INK2, ORANGE, OUT, P, bullets, callout, figure,
                                    table)

TITLE = "Yuki：群組聊天機器人的長期記憶檢索架構"


def fig_architecture() -> Path:
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    ax.set_xlim(0, 7.6)
    ax.set_ylim(0, 4.3)
    ax.axis("off")

    def box(x, y, w, h, text, edge, fill):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                    linewidth=1.2, edgecolor=edge, facecolor=fill))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.2, color=INK,
                linespacing=1.35)
        return x, y, w, h

    def arrow(p, q, style="-|>", ls="-"):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=10, linewidth=1.1,
                                     color=INK2, linestyle=ls, shrinkA=2, shrinkB=2))

    # ---- 回覆路徑 ----
    ax.text(0.15, 4.08, "回覆路徑（被呼叫時，同步）", fontsize=9.5, color=BLUE, fontweight="bold")
    steps = ["使用者\n訊息", "查詢分解\n（依人物）", "混合檢索\ndense + BM25\nRRF", "時間感知\n加分",
             "Cross-encoder\nrerank", "LLM 重排\n（時間題）", "名額輪流\n取 6 筆", "生成回覆\n（LLM）"]
    w, h, gap, y1 = 0.80, 0.86, 0.12, 2.95
    tops = []
    for i, s in enumerate(steps):
        x = 0.15 + i * (w + gap)
        fill = "#eaf2fc" if 0 < i < 7 else "#f1f0ec"
        tops.append(box(x, y1, w, h, s, BLUE if 0 < i < 7 else INK2, fill))
        if i:
            px = tops[i - 1]
            arrow((px[0] + w, y1 + h / 2), (x, y1 + h / 2))

    # ---- 記憶寫入路徑 ----
    ax.text(0.15, 1.78, "記憶寫入（背景，非同步）", fontsize=9.5, color=ORANGE, fontweight="bold")
    y2, h2 = 0.55, 0.92
    store = box(1.95, y2, 1.45, h2, "記憶庫\nSQLite + 向量矩陣\n+ BM25 索引", INK2, "#f1f0ec")
    guard = box(3.7, y2, 1.25, h2, "守門規則\nID 白名單・刪除上限\n去重・主詞補名", ORANGE, "#fdf1eb")
    llm = box(5.15, y2, 1.2, h2, "LLM 整理\n新增／更新／刪除\n（JSON schema）", ORANGE, "#fdf1eb")
    buf = box(6.5, y2, 0.98, h2, "對話緩衝\n8 則或\n閒置 90 秒", ORANGE, "#fdf1eb")
    arrow((buf[0], y2 + h2 / 2), (llm[0] + llm[2], y2 + h2 / 2))
    arrow((llm[0], y2 + h2 / 2), (guard[0] + guard[2], y2 + h2 / 2))
    arrow((guard[0], y2 + h2 / 2), (store[0] + store[2], y2 + h2 / 2))

    # 記憶庫供檢索；聊天訊息進入緩衝；寫入前先查相關舊記憶
    hybrid = tops[2]
    arrow((store[0] + 0.45, y2 + h2), (hybrid[0] + hybrid[2] / 2, y1))
    msg = tops[0]
    arrow((msg[0] + msg[2] / 2, y1), (buf[0] + buf[2] / 2, y2 + h2), ls="--")
    ax.text(0.2, 2.17, "所有聊天訊息（包含沒叫 Yuki 的）也進入緩衝", fontsize=7.5, color=INK2)
    ax.text(3.7, 0.27, "整理前先從記憶庫檢索相關舊記憶，作為更新／刪除的依據", fontsize=7.5, color=INK2)

    fig.tight_layout(pad=0.2)
    path = FIG / "architecture.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def build() -> Path:
    f_arch = fig_architecture()
    base.PIPELINE = base.PIPELINE[:4] + [
        ("+ LLM 重排（每一題）", 84.0, "1.0 秒"),
        ("本架構（LLM 重排只用於時間題）", 82.0, "0.8 秒"),
    ]
    f_pipe, f_time, f_pool = base.fig_pipeline(), base.fig_time_split(), base.fig_pool()
    out = OUT / "yuki_memory_rag_research_report.pdf"
    doc = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm,
                            topMargin=18 * mm, bottomMargin=20 * mm, title=TITLE, author="Yuki 專案")
    st = []

    # ---------- 標題與摘要 ----------
    st += [P("Yuki：群組聊天機器人的<br/>長期記憶檢索架構", "title"),
           P("最終研究報告　｜　2026-11", "subtitle")]
    st.append(callout(
        "<b>摘要</b>：本研究提出一套給 Discord 群組聊天機器人使用的長期記憶檢索架構，結合查詢分解、"
        "dense 與 BM25 混合檢索、時間感知加分、cross-encoder rerank 與帶日期的 LLM 重排，並以每個子查詢"
        "保留名額的方式組合結果。在 18,270 筆、涵蓋 5 年的記憶庫上，前 6 筆找到正確記憶的比例為 "
        "<b>82%</b>（僅混合檢索為 69%），檢索延遲中位數 0.8 秒；若每一題都做 LLM 重排可達 84%，延遲約 1.0 秒。"))

    # ---------- 1. 問題定義 ----------
    st.append(P("1. 問題定義", "h1"))
    st.append(P("群組聊天機器人的長期記憶，跟一對一助理或文件問答的檢索有幾個不同的難處："))
    st += bullets([
        "<b>多人</b>：同一個問題可能涉及好幾個人（「A 跟 B 最近在忙什麼」），任何一人的記憶都不能被另一人擠掉。",
        "<b>內容高度相似</b>：同一個人每天做差不多的事，記憶文字很像，只靠語意相似度分不出是哪一天。",
        "<b>問題常帶時間</b>：「昨天」「上個月」「最近」很常見，但說法常不精確；而且答案的時間資訊在記憶的日期裡，不在內文裡。",
        "<b>寫入沒有人審核</b>：記憶由模型從閒聊中自動整理，必須防止錯誤覆寫、大量刪除與惡意指令。",
        "<b>本地部署</b>：所有模型跑在單張消費級顯卡上，延遲和並行能力有限。",
    ])

    # ---------- 2. 系統架構 ----------
    st.append(P("2. 系統架構", "h1"))
    st.append(P("系統分成兩條路徑：使用者呼叫 Yuki 時同步執行的<b>回覆路徑</b>，以及在背景整理對話的<b>記憶寫入路徑</b>。"
                "兩者共用同一個記憶庫，記憶依伺服器隔離。"))
    st.append(figure(f_arch, 155,"圖 1　系統架構：回覆路徑（上）與記憶寫入路徑（下）"))

    st.append(P("2.1 回覆路徑", "h2"))
    st += bullets([
        "<b>查詢分解</b>：以規則找出問題提到的人（成員名單、@提及、「我」換成發問者），每人一個子查詢；"
        "只有「單一人物、多個面向」的問題才呼叫 LLM 拆分。",
        "<b>混合檢索</b>：每個子查詢分別做 dense（bge-m3 餘弦相似度）與 BM25（jieba 斷詞）檢索，各取前 20，"
        "以 RRF（k=60）合併。RRF 只看名次，兩種分數尺度不同也能合併；BM25 補足人名與專有名詞。",
        "<b>時間感知</b>：以規則解析「昨天」「上週五」「上個月」「2023年3月」「最近」等說法成日期區間。"
        "區間內的記憶在候選階段多一份排名，rerank 後再加 0.1 分，但<b>不排除</b>區間外的記憶。",
        "<b>Cross-encoder rerank</b>：對（主題, 記憶）成對打分，低於 0.05 的視為不相關並丟棄。打分前先拿掉時間說法與"
        "「什麼時候」「是不是」等問法，只留主題：問「什麼時候」時答案在記憶的日期裡，用原句打分會讓每筆記憶都被判定為答不了。"
        "若問題已指定某人、卻全部低於門檻，保留分數最高的 2 筆。",
        "<b>LLM 重排</b>：問題含時間說法時，將前 10 筆連同各自日期與今天日期交給聊天模型挑選、排序；"
        "未被選中的依原順序接在後面，不會遺失。沒有時間說法的問題不重排（見 5.1）。",
        "<b>時間序問題</b>：「最早」「第一次」「最後一次」類問題的答案在日期，候選數放大到 60，相關記憶依日期排序。",
        "<b>名額輪流</b>：各子查詢輪流取，共 6 筆放進 prompt。每筆附上日期與程式預先算好的「距今多久」。",
    ])

    st.append(P("2.2 記憶寫入路徑", "h2"))
    st += bullets([
        "<b>觸發</b>：頻道累積 8 則人類訊息或閒置 90 秒，就把這段對話交給 LLM 整理，不影響回覆速度。",
        "<b>操作</b>：LLM 依 JSON schema 輸出新增、更新、刪除三種操作。同一件事的變化必須「更新並合併」舊記憶，而不是新增一筆。",
        "<b>守門規則</b>：更新與刪除只能使用這次檢索到的記憶 ID；每批最多 5 個操作、2 個刪除；新增前與既有記憶相似度達 0.90 以上視為重複；"
        "主詞一定補上人名；對話中要求「刪掉所有記憶」之類的指令不會被執行。",
        "<b>白板</b>：另存最多 20 條的群組近況，超過時由 LLM 壓縮。",
    ])

    st.append(P("2.3 儲存", "h2"))
    st.append(P("記憶存在 SQLite，向量以 BLOB 形式一併儲存；查詢時在記憶體中以 numpy 做暴力餘弦相似度計算，"
                "並維護 BM25 索引與斷詞快取。1.8 萬筆時 dense 查詢約 5 毫秒、BM25 約 12 毫秒。"
                "資料庫記錄 embedding 模型名稱，模型更換時拒絕混用並提示重新計算向量。"))

    # ---------- 3. 模型 ----------
    st.append(P("3. 模型選擇", "h1"))
    st.append(table([
        ["用途", "模型", "選擇理由"],
        ["Embedding", "BAAI/bge-m3", "多語言，中文表現佳；不需要 query／document 前綴；本機 GPU 執行"],
        ["Rerank", "BAAI/bge-reranker-v2-m3", "多語言 cross-encoder，成對打分比向量相似度精確；評估中優於更大的 Qwen3-Reranker-4B"],
        ["稀疏檢索", "BM25 + jieba", "補足人名、專有名詞等 dense 容易忽略的字面比對"],
        ["聊天 LLM", "Swift-Qwen3.8-27B", "回覆、查詢分解、LLM 重排、記憶整理共用同一個模型"],
    ], [26 * mm, 46 * mm, 98 * mm]))
    st.append(Spacer(1, 6))
    st.append(P("Rerank 模型比較", "h2"))
    st.append(table([
        ["", "bge-reranker-v2-m3（採用）", "Qwen3-Reranker-4B（4-bit）"],
        ["參數量", "約 0.57B", "4B"],
        ["做法", "傳統 cross-encoder，輸出一個分數", "LLM 判斷 yes／no，可加任務說明"],
        ["Hit@6（全部）", "82%", "80%"],
        ["沒有時間說法的題目", "68%", "62%"],
        ["延遲中位數", "0.8 秒", "1.0 秒"],
    ], [42 * mm, 64 * mm, 64 * mm], highlight_row=None))
    st.append(Spacer(1, 4))
    st.append(P("兩者在相同架構下比較。這份資料由很短的中文句子組成，較大的 LLM 式 reranker 沒有帶來提升，"
                "差距集中在沒有時間說法、最依賴語意判斷的題目；4-bit 量化與分數門檻也可能造成影響。"))
    st.append(P("LLM 的兩種部署", "h2"))
    st.append(table([
        ["", "GGUF Q4_K_XL + MTP（LM Studio）", "NVFP4（vLLM）"],
        ["單一請求速度", "快：MTP 推測解碼一次猜多個 token", "相近：同為約 4-bit，受顯存頻寬限制"],
        ["並行", "不支援，同時只能處理 1 個請求", "支援，8 個並行時吞吐量約 5.7 倍"],
        ["長 prompt", "一般", "較快：Blackwell 原生 FP4 運算"],
        ["適合", "日常回覆（一次回一個人）", "大量資料處理、多人同時使用"],
    ], [30 * mm, 70 * mm, 70 * mm]))
    st.append(Spacer(1, 4))
    st.append(P("推理模型用在查詢分解、LLM 重排這類短的結構化任務時，一律關閉思考：任務本身不需要推理，"
                "開啟後思考內容會耗盡輸出長度上限，導致沒有答案。"))

    st.append(P("3.1 主要參數", "h2"))
    st.append(table([
        ["參數", "值", "參數", "值"],
        ["每路檢索候選數", "20", "RRF k", "60"],
        ["rerank 門檻", "0.05", "時間加分", "0.1"],
        ["LLM 重排範圍", "前 10 筆", "最後放進 prompt", "6 筆"],
        ["LLM 重排時機", "問題含時間說法", "時間序問題候選數", "60"],
        ["rerank 保底筆數", "2", "rerank 打分對象", "去掉問法的主題"],
        ["子查詢上限", "4", "去重相似度", "0.90"],
        ["每批操作／刪除上限", "5 ／ 2", "白板上限", "20 條"],
    ], [50 * mm, 35 * mm, 50 * mm, 35 * mm]))

    # ---------- 4. 實驗設定 ----------
    st.append(P("4. 實驗設定", "h1"))
    st.append(P("真實群組聊天沒有標準答案、內容多為閒聊難以考究，也牽涉隱私，因此以模擬資料評估："))
    st += bullets([
        "<b>資料集</b>：由 LLM 產生 10 位虛構成員、2021-10-29 ～ 2026-10-29 每人每天一筆記憶，共 18,270 筆；"
        "每人每年有 6 個主線事件，讓記憶前後連貫。",
        "<b>題目</b>：隨機抽 200 筆記憶，由 LLM 以另一位成員的身分寫成口語問題；每題設定發問日期"
        "（記憶發生後 1～300 天），讓時間說法有對應日期。137 題含時間說法。",
        "<b>指標</b>：Hit@6，即最後放進 prompt 的 6 筆記憶中是否包含標準答案。200 題的抽樣誤差約 ±3 個百分點。",
    ])

    # ---------- 5. 結果 ----------
    st.append(KeepTogether([P("5. 結果", "h1"), P("5.1 消融實驗", "h2"),
                            figure(f_pipe, 165, "圖 2　逐步加入各元件的 Hit@6 與檢索延遲（中位數，不含生成回覆）")]))
    st.append(table([
        ["設定", "全部 200 題", "有時間說法（137 題）", "沒有時間說法（63 題）", "檢索延遲"],
        ["rerank + 時間感知", "82%", "87%", "71%", "0.15 秒"],
        ["+ 查詢改寫", "82%", "86%", "75%", "1.7 秒"],
        ["+ LLM 重排（每一題）", "84%", "90%", "71%", "1.0 秒"],
        ["+ LLM 重排 + 查詢改寫", "84%", "89%", "73%", "2.8 秒"],
        ["本架構（重排只用於時間題）", "82%", "88%", "68%", "0.8 秒"],
        ["本架構，去掉主題打分與保底", "82%", "89%", "68%", "0.8 秒"],
    ], [48 * mm, 24 * mm, 34 * mm, 34 * mm, 30 * mm], highlight_row=5))
    st.append(Spacer(1, 4))
    st += bullets([
        "<b>LLM 重排只改善有時間說法的題目</b>（87% → 90%），沒有時間說法的題目重排前後都是 71%。"
        "因此本架構只在問題含時間說法時重排，延遲中位數從 1.0 秒降到 0.8 秒；整體 82% 與 84% 的差距在 ±3 點誤差內。"
        "需要最高命中率時可改為每一題都重排。",
        "<b>LLM 查詢改寫</b>（把口語問題改寫成記憶的寫法）與 LLM 重排並用時沒有額外提升（逐題比對各有 4 題只有一方答對），"
        "卻多 1.8 秒，因此不列入本架構。",
        "<b>主題打分與保底</b>對這批題目沒有影響（82% 對 82%），保留它們是為了「最早／什麼時候」類問題："
        "這類問題用原句打分時，候選記憶的分數全部低於門檻，結果為 0 筆。",
    ])

    st.append(P("5.2 時間：加分而非過濾", "h2"))
    st.append(P("若把時間說法當成硬過濾條件，在含時間說法的 137 題中，有 38 題（28%）的說法與記憶實際日期不符"
                "（例如 6 天前的事說成「前天」），硬過濾會把正解直接排除："))
    st.append(figure(f_time, 135, "圖 3　硬過濾在時間說法正確與不精確時的表現（有 rerank）"))
    st.append(P("因此本架構採用加分制：時間吻合者優先，但不排除其他記憶。加分強度在 0.05～0.3 之間結果穩定（82%）。"))

    st.append(P("5.3 瓶頸在排序", "h2"))
    st.append(figure(f_pool, 125, "圖 4　擴大候選池提高召回，但最終命中率不變"))
    st.append(P("候選池從 20 擴大到 100，正解在池內的比例從 88% 升到 92%，最終結果維持 82%。"
                "未命中的題目中，約 6% 是正解在池內卻排不進前 6（內容相似、只差日期），"
                "這正是帶日期的 LLM 重排要解決的部分；其餘約 8～12% 是口語、模糊的問題根本沒有被撈到。"))

    st.append(P("5.4 定性觀察", "h2"))
    st.append(P("將模擬記憶接上 Discord 實際對話測試 6 題，涵蓋指定日期、指定月份、兩人並問與模糊問題，"
                "檢索皆把正確記憶排在第一位。唯一的問題是把 4 個月前的事描述得像最近發生；"
                "在 prompt 中為每筆記憶附上「距今多久」後改善。"))

    # ---------- 6. 限制 ----------
    st.append(P("6. 限制與未來方向", "h1"))
    st += bullets([
        "<b>模擬資料偏差</b>：題目由 LLM 依記憶內容寫成，用字常與記憶相近；真實使用者的問法可能更難，結論需要以真實資料驗證。",
        "<b>模糊問題</b>：沒有時間說法的題目停在 68～75%，是最弱的類型。",
        "<b>時間序問題</b>：「最早」「第一次」「最後一次」已改為偵測後依日期排序，但評估題目中沒有這類問題，效果尚未量化。",
        "<b>計數問題</b>：「某人說過幾次」需要對所有相關訊息逐一判斷、由程式計數，無法以前 k 筆檢索回答。",
        "<b>評估規模</b>：200 題的誤差約 ±3 點，82% 與 84% 的差異只能視為方向。",
    ])

    # ---------- 7. 結論 ----------
    st.append(P("7. 結論", "h1"))
    st += bullets([
        "在群組聊天的長期記憶中，<b>時間</b>是最關鍵的訊號，但應作為排序加分而非過濾條件。",
        "記憶量大且內容相似時，<b>瓶頸在排序而非召回</b>；能讀懂日期的 LLM 重排比擴大候選池有效。",
        "以子查詢為單位<b>保留名額</b>，讓多人問題中每個人都有記憶進入 prompt。",
        "<b>rerank 模型不是越大越好</b>：4B 的 Qwen3-Reranker 沒有勝過約 0.57B 的 bge-reranker。",
        "整體架構在單張消費級顯卡上，於 1.8 萬筆記憶中以中位數 0.8 秒的檢索延遲達到 82% 的 Hit@6；"
        "每一題都做 LLM 重排可達 84%（約 1.0 秒）。",
    ])

    doc.build(st, onFirstPage=on_page, onLaterPages=on_page)
    return out


def on_page(canvas, doc):
    from reportlab.lib import colors
    canvas.saveState()
    canvas.setFont("JH", 8)
    canvas.setFillColor(colors.HexColor(INK2))
    canvas.drawString(20 * mm, 12 * mm, TITLE)
    canvas.drawRightString(190 * mm, 12 * mm, f"{doc.page}")
    canvas.restoreState()


if __name__ == "__main__":
    print(build())
