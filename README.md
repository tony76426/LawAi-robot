# LawAi-robot｜法律語意檢索實驗版 v2

> 這是獨立實驗儲存庫，**不是正式 LawAI 的 AI Robot / AI Report 後端**。Render LawAi-robot 目前仍暫停；修改 GitHub 不等於啟動部署。

## 修復了哪些問題？

舊程式採用已過時的 Embedding 呼叫方式，且無論查到什麼都強制回傳相似度最高的答案。原 vector_database.json 有 30 筆 QA，當中的 1536 維向量全部為正、數值分布像隨機測試資料。**新版只使用原始 FAQ 的問題與答案文字，完全忽略舊 vector 欄位。** 原始檔保留備份。

| 功能 | 現況 |
| --- | --- |
| 30 筆既有 FAQ、關鍵字 Top-5 檢索、相似度不足時拒答 | 已實作 |
| OpenAI 真實 Embedding 離線建立與索引驗證 | 已實作程式；尚未呼叫付費 API 生成索引 |
| 有效索引時向量＋關鍵字混合檢索 | 已實作；索引未建立時安全退回關鍵字 |
| 台灣法條／法院裁判來源格式與人工審核欄位 | 已實作；目前 legal_sources.json 為空 |
| 支持、反方、審閱三階段模型分析 | 已實作受控 API，預設停用 |
| 來源 ID 查驗 | 僅能檢查 [S1] 等 ID，**不能認證法律來源及推論的真實正確性** |
| 安全顯示聊天文字、分頁內聊天紀錄 | 已實作 |
| 法務部與司法院 API 自動同步及法規效力查核 | **尚未實作** |

## 本機啟動與測試

Python 3.11 以上：

~~~bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
python app.py
~~~

開啟 http://127.0.0.1:10000 。沒有真實向量索引時，介面會顯示「關鍵字檢索」，不會冒稱已完成語意搜尋。

### 建立真實語意向量（明確啟動才會產生 API 費用）

先在本機環境變數設定 OPENAI_API_KEY，然後執行：

~~~bash
python scripts/build_index.py --model text-embedding-3-small
python app.py
~~~

產生的 data/embedding_index.json 會被 .gitignore 排除，不會自動提交 GitHub。它包含模型名稱、資料內容指紋、維度與單位向量驗證。資料變更或模型變更後必須重建並重啟。若正確索引存在且金鑰有效，搜尋時會再向 OpenAI 生成使用者問題的 Embedding；API 失敗則明確退回關鍵字檢索。

Render 的檔案系統可能不持久，重新部署時必須先規劃可持久存放索引的方式。**程式不會偷偷在正式網站上重建向量或啟動付費模型。**

可選環境變數：LAWAI_INDEX_PATH、LAWAI_EMBEDDING_MODEL（預設 text-embedding-3-small）。

## API

- POST /api/search：JSON 格式 {"question":"我要離婚"}，回傳候選、分數、來源、審核狀態及真正檢索模式。
- POST /api/generate：相容原本的聊天介面，回傳 question、answer、score；FAQ 回覆標示非個案法律意見，沒有相關答案就拒答。
- GET /api/health：查詢 FAQ 數量、人工審核來源數量、索引狀態。
- POST /api/compare：**預設關閉**；必須設定 LAWAI_ENABLE_COMPARISON=1、LAWAI_API_TOKEN、OPENAI_API_KEY、有效真向量索引，且檢索得到經人工審核的法律來源，才能進行三次模型呼叫。

模型 A／B／審閱者預設分別為 gpt-4o-mini、gpt-4o、gpt-4o-mini，可由 LAWAI_MODEL_A / LAWAI_MODEL_B / LAWAI_MODEL_REVIEWER 指定。必須核對實際模型權限與費率。模型比較必須以 Authorization: Bearer <LAWAI_API_TOKEN> 提交，**不可把 Token 放進公開前端 JavaScript**。如在環境中設定 LAWAI_API_TOKEN，所有 POST 端點也都要求這個 Bearer Token；正式對外使用需要後端登入代理。

## 法律資料匯入

knowledge/legal_sources.json 初始值為空陣列 []，不能臆造法律依據。人工取得並核對合法來源後，可新增 JSON 物件，必要欄位：

- id：唯一識別碼
- kind：law 或 judgment
- title：已核對法條或裁判名稱
- content：合法取得的原文或片段
- source_url：有效 HTTPS 原始網址
- review_status：人工確認後才填 approved
- reviewed_at：實際人工核對日期
- reviewed_by：實際核對者識別

格式完整**不等於**程式確認資料正確。必須由律師核對法規有效日期、裁判原文、法律見解及撤下／修正機制。請勿將私人卷證、個人識別資料或密鑰上傳公開 GitHub。

## 限制、資安與驗收

- 不再使用 innerHTML 插值呈現使用者／模型文字；聊天記錄暫存在同分頁 sessionStorage，並移除舊 localStorage HTML 記錄。
- 有輸入長度限制、簡易每 IP 頻率限制；但正式多機部署仍須 WAF、共享頻率限制、權限驗證、API 費用預算及安全稽核。
- 三模型結果不代表三人律師審查。引用 ID 比對只確認輸出使用了目前來源清單，不代表法條有效或推論正確。
- tests/test_rag.py 及 GitHub Actions 只做離線工程測試，沒有宣稱法律正確率。正式部署前應有至少 100～200 個律師標記問題，測量 Recall@5、引用正確率、拒答率、成本、延遲。
- 此儲存庫沒有自動接通司法院、法務部或正式 LawAI backend。正式整合需先做合法資料取得、測試環境、律師審核與功能回歸測試。

## 實際導入建議

先在本機建立 30 筆真向量並跑評估；再建立經人工審查的台灣法條／裁判資料集；接著測試雙模型比較及成本／風險；最後才考慮整合正式 LawAI。保持 Render 暫停，避免立即增加月租。
