# 開發 PM 交接文件

> 新人只讀這一份就能接手。交件、壓縮對話前更新「現況」並 commit。

## 角色
- 負責：把 `企劃.md` 階段表裡的項目寫成階段規格（`docs/stages/`）、派 `pm-workflow:implementer`、驗收、commit、push；改了檢索或 prompt 要跑評估。
- 不負責（交給誰）：決定做哪一項、改使用者定過的規則（總 PM 問使用者）。

## 流程
- 每一項照 `/pm-workflow:stage`：影響盤點 → 小規格 → 派實作 → 自己抽查 → 跑檢查 → push → 回報總 PM。
- 階段規格檔名：`docs/stages/A1-規格.md`、報告 `docs/stages/A1-報告.md`（用企劃的編號）。
- 驗收標準先看 `開發規格.md` 對應的節（例：6.12 有寫「關掉 LM Studio，5 秒內收到私訊」）。
- push 前 hook 會跑：單元測試、個資掃描。被擋下照 `CLAUDE.md` 的隱私規則處理。
- 要測 Discord 上的實際行為，在**測試伺服器**；主伺服器不要拿來試。
- 只結束自己啟動的 Yuki／BreezyVoice 程序（Yuki 可能是使用者自己開的）。

## 現況（最後更新：2026-11-18）
- 進行中：A1 第二版（含 LM Studio、SearXNG，D-16）已 push。等使用者跑 scripts/autostart/install.ps1、重開機登入、測試伺服器 /來語音 驗收；LM Studio 實際載入模型也在那時才會驗到（見 docs/stages/A1-報告.md）。
- 最近完成：A1 腳本兩版（scripts/autostart/：四個看顧＋start/stop/install）。
- 已知：顯卡被別的工具的 gemma（16GB）佔著時，Yuki 的模型載不進去。Docker Desktop 開時會跳視窗。Docker＋SearXNG 目前開著（實作測試時開的）。

## 待辦（階段 A，照順序）
- [ ] A1（等使用者驗收）開機自動啟動 Yuki 和 BreezyVoice（啟動資料夾捷徑，D-15；BreezyVoice 要等 WSL 起來；Yuki 掛掉要自動重開）。只動啟動資料夾，其他系統設定要改先回報。
- [ ] A2 維運：AI 追蹤紀錄、出錯私訊擁有者（開發規格 6.12）
- [ ] A3 語音實測收尾（開發規格 6.0 的 5 項實測；要使用者一起進語音）
- [ ] A4 小修：
  - 人物檔案自動更新每分鐘空跑（`yuki/profile/auto.py`：發言太少、不建檔的人一直被 `has_unmerged` 當成還沒合併）→ log 只在真的有更新時寫
  - 量「查不到時說不知道」（開發規格 6.4 表格最後一列）
  - 開發規格第 7、8 節照現況更新（git、SearXNG、README 那幾條已處理）

## 待決事項（要誰決定什麼）
| 事項 | 選項 | 建議 | 等誰 |
|---|---|---|---|

## 要讀的文件
- `CLAUDE.md`（環境、隱私、寫 prompt 的經驗）
- `企劃.md` 的「優先順序」節
- `開發規格.md` 對應的節（6.0、6.4、6.12…）
- `README.md`（架構、專案結構）
