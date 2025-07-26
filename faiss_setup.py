
import json
import numpy as np
import faiss

with open("vector_database.json", "r", encoding="utf-8") as f:
    vector_data = json.load(f)

dimension = len(vector_data[0]["embedding"])
index = faiss.IndexFlatL2(dimension)
metadata = []

for entry in vector_data:
    vec = np.array(entry["embedding"]).astype("float32")
    index.add(np.expand_dims(vec, axis=0))
    metadata.append({
        "question": entry.get("question", ""),
        "answer": entry.get("answer", "")
    })

faiss.write_index(index, "lawai_faiss.index")
with open("lawai_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print("✅ Faiss index 建立完成")
