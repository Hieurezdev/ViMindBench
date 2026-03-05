"""
init_embedding.py
-----------------
Chạy file này TRƯỚC kaggle_pipeline.py để:
1. Cài embedding stack ổn định
2. Load viDense model lên GPU / RAM
3. Warm-up một lần encode thử
4. Lưu trạng thái sẵn sàng (optional flag file)

Thứ tự chạy trên Kaggle:
  [Cell 1] !python init_embedding.py
  [Cell 2] !python kaggle_pipeline.py  (hoặc run pipeline cell)
"""

# ── Install embedding stack ổn định ──────────────────────────────────────────
!uv pip install sentence-transformers==3.0.1 transformers==4.46.3

# ── Load và warm-up viDense ───────────────────────────────────────────────────
import os
import time

EMBEDDING_MODEL = "namdp-ptit/ViDense"
READY_FLAG = "/kaggle/working/.embedding_ready"

print(f"[init_embedding] Loading model: {EMBEDDING_MODEL}")
t0 = time.time()

try:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True)
    print(f"[init_embedding] ✓ Model loaded in {time.time() - t0:.1f}s")

    # Warm-up encode
    warmup_text = "Tâm lý học là gì?"
    vec = model.encode(warmup_text, normalize_embeddings=True)
    print(f"[init_embedding] ✓ Warm-up encode OK — vector dim: {len(vec)}")

    # Ghi flag để pipeline chính biết embedding đã sẵn sàng
    with open(READY_FLAG, "w") as f:
        f.write("ready")
    print(f"[init_embedding] ✓ Flag written: {READY_FLAG}")
    print("[init_embedding] ✓ viDense sẵn sàng. Bây giờ có thể chạy kaggle_pipeline.py")

except Exception as e:
    print(f"[init_embedding] ✗ ERROR: {e}")
    raise
