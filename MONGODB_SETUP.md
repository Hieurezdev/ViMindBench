# MongoDB Vector Search Setup Guide

## Overview

Pipeline đã được chuyển sang dùng **MongoDB Atlas** thay vì Chroma/BM25 local.

## Chiến lược Retrieval

Retriever sẽ tự động thử theo thứ tự:

1. **Atlas Vector Search** (nếu có vector index) → Chính xác nhất
2. **Text Search** (nếu có text index) → Fallback 1
3. **Keyword Search** (regex) → Fallback 2

## Bước 1: Import Data vào MongoDB

```bash
# Đã có sẵn
python import_json_to_mongodb.py
```

## Bước 2: Thêm Embeddings (Tuỳ chọn - cho Vector Search)

**Nếu bạn muốn dùng Vector Search (khuyến nghị):**

```bash
uv add tqdm  # Progress bar
uv run python add_embeddings.py
```

Script này sẽ:
- Đọc tất cả documents từ MongoDB
- Tạo embedding vector cho mỗi document (BGE-M3: 1024 dimensions)
- Lưu vào trường `embedding`

**Lưu ý:** Quá trình này có thể mất vài phút tuỳ số lượng documents.

## Bước 3: Tạo Vector Search Index trên Atlas

### 3.1. Truy cập MongoDB Atlas

1. Đăng nhập: https://cloud.mongodb.com
2. Chọn Cluster → Browse Collections
3. Database: `Data` → Collection: `mental`
4. Tab: **Search Indexes**
5. **Create Search Index** → **Atlas Vector Search**

### 3.2. Cấu hình Index

**Tên index:** `vector_index`

**JSON Configuration:**

```json
{
  "fields": [
    {
      "type": "vector",
      "path": "embedding",
      "numDimensions": 1024,
      "similarity": "cosine"
    },
    {
      "type": "filter",
      "path": "type"
    },
    {
      "type": "filter",
      "path": "keywords"
    }
  ]
}
```

**Lưu ý:**
- `numDimensions: 1024` cho `BAAI/bge-m3`
- `similarity: cosine` → Similarity metric
- `filter` fields → Để filter kết quả theo type/keywords

### 3.3. Đợi Index Build

- Thời gian: 5-30 phút tuỳ dataset size
- Check status trong Atlas UI

## Bước 4: (Tuỳ chọn) Tạo Text Search Index

Nếu không muốn dùng Vector Search, bạn có thể tạo Text Index:

**Tên index:** `text_search_index`

**Configuration:**

```json
{
  "mappings": {
    "dynamic": false,
    "fields": {
      "content": {
        "type": "string",
        "analyzer": "lucene.standard"
      },
      "summary": {
        "type": "string",
        "analyzer": "lucene.standard"
      },
      "keywords": {
        "type": "string"
      }
    }
  }
}
```

## Bước 5: Chạy Pipeline

```bash
uv run main.py
```

## Kiểm tra Retrieval Strategy

Khi chạy, console sẽ báo strategy nào đang được dùng:

```
✓ Connected to MongoDB: Data.mental
Vector search activated!  ← Vector Search đang dùng

# Hoặc
Vector search failed: ..., falling back to text search  ← Text Search
Text search failed: ..., falling back to keyword search  ← Keyword
```

## Troubleshooting

### Lỗi: "MONGO_URI not set"
→ Kiểm tra file `.env` có `MONGO_URI`

### Lỗi: "index not found: vector_index"  
→ Đợi Atlas build xong index, hoặc bỏ qua và dùng text search

### Lỗi: "field 'embedding' not found"
→ Chạy `uv run python add_embeddings.py`

### Retrieval quá chậm
→ Kiểm tra:
1. Atlas index đã build chưa
2. Network connection đến Atlas
3. Dataset size (có thể cần upgrade Atlas tier)

## Production Tips

1. **Batch Embeddings**: Script `add_embeddings.py` đã dùng batch, nhưng có thể optimize thêm
2. **Caching**: Có thể cache embeddings ở client side
3. **Hybrid Search**: Kết hợp vector + text search với weights
4. **Index maintenance**: Rebuild index định kỳ khi add data mới

## Khác biệt so với Chroma/BM25

| Feature | Chroma/BM25 (Cũ) | MongoDB (Mới) |
|---------|------------------|---------------|
| **Storage** | Local files | Cloud Atlas |
| **Scalability** | Limited | Unlimited |
| **Speed** | Fast (local) | Depends on network |
| **Setup** | Automatic | Requires index |
| **Cost** | Free | Free tier → Paid |
| **Backup** | Manual | Automatic |
| **Collaboration** | Difficult | Easy |

## Env Variables

```bash
# MongoDB (Required)
MONGO_URI=mongodb+srv://user:pass@cluster.mongodb.net/
MONGO_DB_NAME=Data
MONGO_COLLECTION_NAME=mental

# LLM
OPENAI_BASE_URL=http://localhost:8000/v1
OPENAI_API_KEY=EMPTY
MODEL_NAME=hoangchihien3011/VietMind

# Pipeline
NUM_QA_PAIRS=10
```

Xong! Pipeline giờ hoàn toàn dựa trên MongoDB.
