# Fashion Detection & Attribute Recognition (F&A Model)

Dự án phát triển hệ thống AI thông minh nhận diện đa đối tượng trong lĩnh vực thời trang. Model được thiết kế để không chỉ định vị chính xác trang phục cơ bản (quần, áo) mà còn tập trung chuyên sâu vào việc phát hiện các phụ kiện thời trang (Accessories) và trích xuất chi tiết các thuộc tính (Attributes) đi kèm như chất liệu, hoa văn, và kiểu dáng thiết kế.

## 🌟 Tính Năng Chính

-   **Nhận diện Đối tượng (Object Detection)**: Phát hiện và phân loại chính xác **47 nhóm đối tượng** thời trang bao gồm trang phục (Outerwear, Top, Bottom, Full body) và phụ kiện (Túi xách, Trang sức, Mũ, Kính, v.v.).
-   **Trích xuất Thuộc tính (Attribute Extraction)**: Phân loại đồng thời **294 thuộc tính** chi tiết cho từng vật thể (Màu sắc, Chất liệu, Hoa văn, Kiểu dáng cổ áo, v.v.).
-   **Kiến trúc Hiện đại**: Sử dụng **YOLOS (You Only Look at One Sequence)** dựa trên Vision Transformer (ViT) kết hợp cơ chế **Double Heads** (Detection Head & Attribute Head).
-   **Web Demo**: Giao diện trực quan cho phép upload ảnh và xem kết quả nhận diện thời gian thực.
-   **Pipeline Tự động**: Hệ thống training end-to-end từ tải dữ liệu, tiền xử lý, training đến đánh giá.

## 📂 Cấu Trúc Dự Án

```
.
├── full_pipeline.py        # All-in-one script: Tải data, xử lý, train và đánh giá model
├── app.py                  # Ứng dụng Web Demo (FastAPI/Uvicorn)
├── model.py                # Định nghĩa kiến trúc model (YOLOS + Attribute Head)
├── preprocessing.py        # Các hàm xử lý ảnh và augmentation
├── inference.py            # Code suy luận (Inference) cho production
├── config.py               # Cấu hình hệ thống (Nếu có tách riêng)
├── utils.py                # Các hàm tiện ích bổ trợ
├── label_descriptions.json # Danh sách nhãn (Classes & Attributes)
├── requirements.txt        # Các thư viện Python cần thiết
├── README.md               # Tài liệu dự án
└── static/                 # Tài nguyên Frontend cho Web app
```

## 🛠 Yêu Cầu & Cài Đặt

Dự án khuyến nghị sử dụng **uv** để quản lý gói (nhanh hơn và ổn định hơn pip).

### Yêu cầu tiên quyết
-   Python 3.8 trở lên
-   GPU (NVIDIA RTX series khuyến nghị) để training nhanh hơn.

### Cài đặt dependencies

```bash
# Sử dụng uv (Khuyến nghị)
uv sync

# Hoặc sử dụng pip truyền thống
pip install -r requirements.txt
```

## 🚀 Hướng Dẫn Sử Dụng

### 1. Training Model

Để bắt đầu quy trình training (bao gồm tự động tải dataset Fashionpedia ~20GB):

```bash
uv run full_pipeline.py
```

*Lưu ý: Quá trình này sẽ tải dữ liệu về thư mục `data/` và lưu checkpoints tại `checkpoints/`.*

### 2. Chạy Demo Web App

Sau khi có model (hoặc dùng checkpoint có sẵn), bạn có thể bật web server để trải nghiệm:

```bash
uv run uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Truy cập `http://localhost:8000` trên trình duyệt để sử dụng:
1.  Upload ảnh thời trang (JPG/PNG).
2.  Hệ thống sẽ hiển thị ảnh gốc bên cạnh ảnh đã detect (vẽ bounding box).
3.  Xem chi tiết JSON kết quả (Class, Confidence, Attributes) ở phía dưới.

## 📊 Kết Quả Đạt Được

Hệ thống đã đạt được những kết quả khả quan trên tập dữ liệu Fashionpedia đầy thách thức:

-   **Model**: YOLOS-Small (Transformer-based)
-   **Số lượng tham số**: ~31 Triệu (Nhẹ, tối ưu cho real-time)
-   **Validation Metrics**:
    -   **mAP@50**: ~31.6% (Độ chính xác tốt cho bài toán 47 classes)
    -   **Attribute Loss**: Rất thấp (~0.016), cho thấy khả năng học thuộc tính xuất sắc.

## 🔍 Phạm Vi Nhận Diện

Hệ thống hỗ trợ nhận diện toàn diện:

1.  **Trang phục**: Áo khoác, Sơ mi, Quần dài, Váy, Đầm, Jumpsuit...
2.  **Phụ kiện**: Túi xách, Ví, Đồng hồ, Mũ, Kính mắt, Thắt lưng, Giày, Tất...
3.  **Thuộc tính**:
    -   *Họa tiết*: Kẻ sọc, Chấm bi, Hoa văn...
    -   *Chất liệu*: Da, Len, Ren, Lụa...
    -   *Chi tiết*: Cổ chữ V, Tay ngắn, Có túi, Khóa kéo...

## 👥 Tác Giả & Tín Dụng

-   **Model Base**: [HuggingFace YOLOS](https://huggingface.co/hustvl/yolos-small)
-   **Dataset**: [Fashionpedia](https://fashionpedia.github.io/)
-   **Framework**: PyTorch, Transformers, Albumentations.