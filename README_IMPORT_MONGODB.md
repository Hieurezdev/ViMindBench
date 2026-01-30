# Import JSON to MongoDB

Script tự động import tất cả file JSON vào MongoDB.

## Yêu cầu

### Cài đặt dependencies

```bash
# Cài đặt pymongo
pip install pymongo python-dotenv
```

Hoặc thêm vào `pyproject.toml`:

```toml
[project]
dependencies = [
    "pymongo>=4.0.0",
    "python-dotenv>=1.0.0"
]
```

### Cài đặt MongoDB

Nếu chưa có MongoDB, cài đặt bằng một trong các cách sau:

**macOS (Homebrew):**
```bash
brew tap mongodb/brew
brew install mongodb-community
brew services start mongodb-community
```

**Docker:**
```bash
docker run -d -p 27017:27017 --name mongodb mongo:latest
```

## Cấu hình

1. Tạo file `.env` từ `.env.example`:
```bash
cp .env.example .env
```

2. Cập nhật các biến môi trường trong `.env`:

```bash
# MongoDB Configuration
MONGO_URI=mongodb://localhost:27017/          # URI kết nối MongoDB
MONGO_DB_NAME=finetuning_data                 # Tên database
MONGO_COLLECTION_NAME=psychology_data         # Tên collection

# Data Directory
DATA_DIR=data/formated_data                   # Thư mục chứa file JSON
```

### Các biến cấu hình:

- **MONGO_URI**: Đường dẫn kết nối MongoDB
  - Local: `mongodb://localhost:27017/`
  - MongoDB Atlas: `mongodb+srv://username:password@cluster.mongodb.net/`
  
- **MONGO_DB_NAME**: Tên database để lưu dữ liệu (mặc định: `finetuning_data`)

- **MONGO_COLLECTION_NAME**: Tên collection để lưu documents (mặc định: `psychology_data`)

- **DATA_DIR**: Thư mục chứa các file JSON cần import
  - `data/formated_data` - Dữ liệu tâm lý học
  - `data/formated_data_tam_li_voi_cuoc_song` - Dữ liệu tâm lý với cuộc sống

## Sử dụng

### Chạy script để import tất cả file JSON:

```bash
python import_json_to_mongodb.py
```

### Import từ thư mục khác:

Thay đổi `DATA_DIR` trong file `.env` hoặc:

```bash
DATA_DIR=data/formated_data_tam_li_voi_cuoc_song python import_json_to_mongodb.py
```

## Tính năng

✅ **Tự động tìm kiếm**: Tìm tất cả file `.json` trong thư mục và các thư mục con

✅ **Batch Processing**: Import theo batch để tối ưu hiệu suất

✅ **Metadata Tracking**: Tự động thêm thông tin về file nguồn vào mỗi document:
   - `_source_file`: Tên file
   - `_source_path`: Đường dẫn đầy đủ

✅ **Error Handling**: Xử lý lỗi khi parse JSON hoặc insert vào MongoDB

✅ **Logging**: Ghi log chi tiết quá trình import

✅ **Hỗ trợ nhiều định dạng**:
   - JSON object: `{...}` → Tự động wrap thành array
   - JSON array: `[{...}, {...}]` → Import trực tiếp

## Ví dụ Output

```
2026-01-29 15:20:10 - INFO - ✓ Connected to MongoDB at mongodb://localhost:27017/
2026-01-29 15:20:10 - INFO - ✓ Using database: finetuning_data
2026-01-29 15:20:10 - INFO - Found 77 JSON files in data/formated_data
2026-01-29 15:20:10 - INFO - Processing: Khoa Hoc Tam Ly_10271.json
2026-01-29 15:20:10 - INFO - ✓ Inserted 1 documents into 'psychology_data'
...
============================================================
Import Summary:
  Files processed: 77/77
  Documents inserted: 77
============================================================
2026-01-29 15:20:45 - INFO - ✓ Import completed successfully!
```

## Kiểm tra dữ liệu

### Sau khi import, kiểm tra dữ liệu trong MongoDB:

```bash
# Kết nối MongoDB shell
mongosh

# Chọn database
use finetuning_data

# Đếm số documents
db.psychology_data.countDocuments()

# Xem document mẫu
db.psychology_data.findOne()

# Xem tất cả source files
db.psychology_data.distinct("_source_file")
```

### Hoặc dùng Python:

```python
from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client["finetuning_data"]
collection = db["psychology_data"]

# Đếm documents
print(f"Total documents: {collection.count_documents({})}")

# Xem document đầu tiên
print(collection.find_one())
```

## Xử lý lỗi thường gặp

### 1. Lỗi kết nối MongoDB

```
✗ Failed to connect to MongoDB: [Errno 61] Connection refused
```

**Giải pháp**: Đảm bảo MongoDB đang chạy:
```bash
# macOS
brew services start mongodb-community

# hoặc kiểm tra status
brew services list
```

### 2. Lỗi parse JSON

```
✗ Failed to parse JSON file xxx.json: Expecting value: line 1 column 1 (char 0)
```

**Giải pháp**: File JSON không hợp lệ, kiểm tra nội dung file

### 3. Thư mục không tồn tại

```
Directory does not exist: data/formated_data
```

**Giải pháp**: Kiểm tra đường dẫn trong `DATA_DIR`

## Tùy chỉnh

### Thay đổi batch size:

Mở file `import_json_to_mongodb.py` và sửa trong hàm `main()`:

```python
stats = importer.import_directory(
    directory=DATA_DIR,
    collection_name=COLLECTION_NAME,
    batch_size=500  # Thay đổi từ 1000 thành 500
)
```

### Import vào collection khác:

```bash
MONGO_COLLECTION_NAME=my_custom_collection python import_json_to_mongodb.py
```

## Lưu ý

- Script sẽ **insert** dữ liệu mới, không xóa dữ liệu cũ
- Nếu muốn xóa dữ liệu cũ trước khi import:
  ```python
  # Thêm vào script trước khi import
  collection.drop()  # Xóa toàn bộ collection
  ```
- Mỗi document sẽ có thêm metadata `_source_file` và `_source_path`
