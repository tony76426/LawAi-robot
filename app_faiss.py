
from flask import Flask, request, jsonify
import openai
import numpy as np
import faiss
import json
import os

app = Flask(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY")  # 請在 Render 上設好環境變數

# 載入 Faiss index 與 metadata
index = faiss.read_index("lawai_faiss.index")
with open("lawai_metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

# 載入 prompt 模板
BASE_PROMPT = "以下是使用者的問題：\n{}
請根據相似問題給出清楚明確的回答。"

@app.route("/api/generate", methods=["POST"])
def generate():
    user_input = request.json.get("input", "")
    if not user_input:
        return jsonify({"error": "未提供輸入"}), 400

    # 將輸入編碼為向量
    embedding_response = openai.Embedding.create(
        input=[user_input],
        model="text-embedding-ada-002"
    )
    user_embedding = np.array(embedding_response["data"][0]["embedding"]).astype("float32")

    # 查找最相似的三筆資料
    D, I = index.search(np.array([user_embedding]), k=3)
    similar = [metadata[i] for i in I[0] if i < len(metadata)]

    # 組合 prompt
    combined_qna = "\n\n".join([f"Q: {x['question']}\nA: {x['answer']}" for x in similar])
    prompt = BASE_PROMPT.format(combined_qna + f"\n\n使用者問題：{user_input}")

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "你是一位法律助理 AI，請簡明扼要協助回答問題"},
            {"role": "user", "content": prompt}
        ]
    )
    return jsonify({"output": response.choices[0].message.content.strip()})
