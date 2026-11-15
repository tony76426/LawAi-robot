# LawAI 法律研究工作臺（前端與 CORS）

本資料夾是 v2 LawAi-robot 的獨立前端。參考原 LawAI AI Robot 的深青藍配色、常見問題、聊天與麥克風入口，改為「智能問答 / 語意資料檢索 / 模型觀點對照」三模式；使用純 HTML + CSS + JavaScript，不需要額外 npm 套件。

## 已完成

- RWD：桌面側邊導覽 + 右側來源、平板與手機直式工作區
- /api/health 顯示實際後端連線、FAQ 數量、索引狀態、人工核對來源數
- /api/generate 舊 FAQ 回覆（有拒答及 FAQ 性質聲明）
- /api/search Top-5 候選與匹配分數、來源及摘要，手機也會在主要對話顯示
- /api/compare 的模型 A / 模型 B / 第三審閱結果（受控、預設關閉）
- 安全文字渲染，不用 innerHTML 插入使用者輸入；對話不存本機資料庫；不在 GitHub 放 API 金鑰
- 語音轉文字（僅在瀏覽器支援且由使用者點擊時啟動）
- 跨網域可用 CORS 白名單，不設定 Access-Control-Allow-Origin: *

## 同網域測試

在儲存庫根目錄：

1. python -m pip install -r requirements.txt
2. python app.py
3. 開啟 http://127.0.0.1:10000/（Flask 導向 /web/）

只要沒有建立過真正 Embedding 索引，健康檢查會顯示尚未就緒；智能問答與資料搜尋會明確以**關鍵字模式**運作。沒有人工核實的法律資料時，模型比較不可啟動。

## 分開部署／CORS CROSS

例如將 web/ 目錄當成 Cloudflare Pages 的靜態資源根目錄，與 API 分別部署：

1. 在 web/config.js 把 window.LAWAI_API_BASE_URL 改成實際 API HTTPS 來源，例如 https://your-api.example.com（**不要寫 token**）。
2. 在後端 Render 環境變數 LAWAI_ALLOWED_ORIGINS 設定實際前端完整來源，例如 https://lab.example.com 。多個網域用逗號隔開。不可設 *。
3. 確認 API 所在 Render 服務已啟用；若仍 Suspended，前端只能顯示離線狀態。
4. 使用瀏覽器實測 OPTIONS preflight、POST /api/search 和 /api/generate、手機版與連線錯誤。
5. 若 LAWAI_API_TOKEN 已設定，前端只能在「連線設定」手動輸入；Token **只存頁面記憶體**。正式公開環境應另建登入與後端代理，不要公開分享 Token。

## 注意

- 舊 airobt.html 保留於主儲存庫，方便對照；不修改正式 lawfavor.com 的 AI Robot。
- 已實作的前端**不是**正式法律研究結果，沒有自動法條／裁判同步。
- 模型比較功能有多次 API 呼叫成本，後端預設關閉且需要已核實來源。
- 當 Render 為 Suspended，本頁無法進行線上 API 諮詢；GitHub 合併不會自行解除暫停。
