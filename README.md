# Báo Cáo Phân Tích & Hướng Dẫn Chia Dữ Liệu (Data Splitting Strategy)

Tài liệu này tổng hợp quá trình thử nghiệm, phân tích lý do thay đổi chiến lược chia tập dữ liệu (**Time-based Split**) cho dataset **CIC-IDS2017**, và hướng dẫn tích hợp giữa phần Tiền xử lý (Baseline ML) với phần Xây dựng Đồ thị (GNN).

---

## 1. Bối Cảnh & Quá Trình Thử Nghiệm

Bộ dữ liệu **CIC-IDS2017** có đặc thù: mỗi dạng tấn công chỉ diễn ra trong một khung giờ cố định ở từng file CSV riêng biệt. Việc áp dụng các kỹ thuật chia thời gian thông thường đã phát sinh các vấn đề nghiêm trọng:

### Cách 1: Chia theo mốc thời gian chung (`SPLIT_MODE = "global"`)

- **Hạn chế:**
  - Tập **Train** thiếu hoàn toàn các nhãn _Bot_, _DDoS_, và _PortScan_.
  - Tập **Validation** chỉ chứa 0.55% mẫu tấn công (chủ yếu là _Bot_ và _PortScan_ chưa từng xuất hiện ở Train). Early stopping của XGBoost/RF bị mất tác dụng.
  - Tập **Test** chứa 67% mẫu tấn công (_PortScan_, _DDoS_). Test chủ yếu đo khả năng phát hiện loại tấn công chưa từng học thay vì đánh giá mô hình tổng quát.

### Cách 2: Chia theo thời gian từng file (`SPLIT_MODE = "per_file"`)

- **Hạn chế:**
  - Do các đợt tấn công (_DoS Hulk_, _GoldenEye_, _Slowhttptest_, _FTP-Patator_, _Web Attack_) chỉ diễn ra trong vài chục phút ở khung giờ riêng, quy tắc 70% thời gian đầu file sẽ đẩy toàn bộ đợt tấn công vào **Train**.
  - Kết quả: Tập **Test** chỉ xuất hiện 4/14 loại tấn công (_Bot_, _DDoS_, _Heartbleed_, _PortScan_). Các loại tấn công lớn khác bị biến mất hoàn toàn khỏi **Val** và **Test**.

---

## 2. Phương Pháp Đề Xuất Chính: `per_file_label`

Để đáp ứng yêu cầu phân tích lỗi chi tiết từng loại tấn công (**Error Analysis - Mục 7.2**) và so sánh công bằng giữa Baseline ML (RF, XGBoost) với GraphSAGE, dự án thống nhất chọn chế độ: **`per_file_label`**.

### Nguyên lý thực hiện:

Với mỗi nhóm kết hợp giữa `(File CSV, Loại nhãn multi-class)`:

1. Sắp xếp toàn bộ flow theo thứ tự thời gian (`timestamp`).
2. **Train:** Lấy 70% số lượng flow đầu tiên.
3. **Validation:** Lấy 15% số lượng flow tiếp theo.
4. **Test:** Lấy 15% số lượng flow cuối cùng.

> **Không xáo trộn (No Shuffle):** Đảm bảo thứ tự thời gian trong từng loại tấn công/bình thường được giữ nguyên. Test luôn nằm ở giai đoạn sau của Train trong cùng một đợt tấn công.

### So sánh các chế độ chia:

| Chế độ chia                            | Ưu điểm                                                                                                  | Nhược điểm / Hạn chế                                                                                                       |
| :------------------------------------- | :------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------- |
| **`per_file_label`** _(Đề xuất chính)_ | + Tất cả 15 nhãn đều có mặt ở 3 tập (Train/Val/Test).<br>• Phù hợp làm Error Analysis & so sánh mô hình. | • Mốc thời gian giữa 3 tập bị chồng lấp (overlapping).<br> + Đồ thị cửa sổ Test có thể bị thiên lệch (ít flow nền BENIGN). |
| **`per_file`**                         | + Cửa sổ thời gian giữ đúng thành phần lưu lượng thực tế.                                                | + Test chỉ có 4/14 loại tấn công.<br>• Val gần như không có dữ liệu tấn công.                                              |
| **`global`** _(Thử nghiệm phụ)_        | + Tuân thủ đúng mốc thời gian thực tế toàn tuần.                                                         | + Train thiếu nhiều loại tấn công quan trọng.<br>• Mô hình bị đánh giá trên nhãn chưa từng học.                            |

---

## 3. Lưu Ý Kỹ Thuật Dành Cho Partner B (GNN & Graph Construction)

Để đảm bảo kết quả thực nghiệm đồng bộ và chính xác khi làm báo cáo, Partner B cần lưu ý các điểm sau:

### 1 Tính Chất Thời Gian (Timestamp Overlap)

- Khoảng thời gian (time-range) của 3 tập Train, Val, Test sẽ **chồng lên nhau** (cùng chạy từ 03/07 đến 07/07).
- Khi viết báo cáo/thuyết minh, hãy mô tả chính xác là: _"Chia theo thời gian trong từng nhóm (file × nhãn)"_, **không** viết _"Tập Test nằm hoàn toàn sau tập Train về mặt thời gian chung"_.

### 2. Cấu Trúc Cửa Sổ Thời Gian & Đồ Thị (Graph Windows)

- Vì cắt 15% cuối của đợt tấn công ra làm Test, một số cửa sổ thời gian trong tập Test có thể tập trung mật độ tấn công rất cao mà không có/ít có lưu lượng bình thường (`BENIGN`) làm nền.
- **Hệ quả:** Đồ thị ở tập Test có thể "sạch" hơn thực tế ngoài đời. Độ chính xác của GNN có thể cao hơn thực tế. **Đây là đặc điểm cần ghi rõ vào mục Hạn chế (Limitations) của báo cáo.**

### 3. Thống Kê Các Nhãn Mẫu Hiếm (Minority Classes)

- Các nhãn có số lượng cực ít (_Heartbleed_: 11 mẫu, _Infiltration_: 36 mẫu, _SQL Injection_: 21 mẫu) khi chia 70/15/15 chỉ còn khoảng 2–6 mẫu ở Val/Test.
- **Định hướng:** Chỉ số Recall/F1-score của các nhãn này không đủ độ tin cậy về mặt thống kê, chỉ nên dùng với mục đích minh họa thay vì đưa ra kết luận tổng quát.

---

## 4. Quy Chuẩn Sử Dụng Dữ Liệu Cho GNN

Partner B tiến hành dựng đồ thị và huấn luyện GNN dựa trên tập dữ liệu đã chuẩn hóa:

- **File dữ liệu đầu vào:**
  - Meta-data (dùng dựng cấu trúc đồ thị): `train_meta.parquet`, `val_meta.parquet`, `test_meta.parquet` (chứa `timestamp`, `src_ip`, `dst_ip`, `label_bin`, `label_multi`).
  - Feature-data (dùng làm Node/Edge Features): `train_features.parquet`, `val_features.parquet`, `test_features.parquet`.
- **Đồng bộ Preprocessor:** Sử dụng lại file scaler đã fit từ `preprocessor.joblib`. **Tuyệt đối không fit lại scaler** trên tập Val/Test.
- **Độc lập tập dữ liệu:** Xây dựng đồ thị hoàn toàn tách biệt trong nội bộ từng tập (Train/Val/Test), **không** nối cạnh (edge) giữa các flow thuộc khác tập với nhau.

---

_Nếu Partner B muốn đề xuất phương pháp chia khác hoặc có thắc mắc về cách đóng gói `.parquet`, hãy trao đổi trước khi tiến hành bước dựng Đồ thị (Graph Construction)._
