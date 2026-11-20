# Báo cáo nhóm 36

## 1. Số liệu luyện tập
- F1 theo seed (rule / cross_check / overall): seed 36007 = 1.0 / 0.6 / 0.8.
- Số ứng viên trên lô sạch (`--no-faults`): chưa ghi nhận số lô sạch bằng lệnh định lượng; hàng đợi trên seed luyện tập: 17 ứng viên ban đầu, sau khi xem ảnh còn 16 dòng confirmed.
- Ngưỡng đã chọn và cách chọn (không dùng `_private`): 2.0% theo quy tắc Day 7; quy tắc tính theo số ảnh/cảnh lỗi chia cho 250 ảnh.

## 2. Vì sao báo hết ứng viên làm F1 giảm?
Khi báo hết ứng viên, precision giảm vì nhiều ứng viên chỉ là lệch tự nhiên của COCO, trong khi recall có thể tăng. Với seed này, hàng đợi ban đầu có 17 ứng viên nhưng chỉ 16 ứng viên được xác nhận sau khi xem ảnh; nếu báo hết 17 thay cho 16, precision và F1 tổng giảm do thêm một false positive.

## 3. Quyết định và lineage
- **QC v1 & Ngưỡng phát hành:** Tỷ lệ lỗi v1 = (Số ảnh/cảnh lỗi trong `errors.json`) / (Tổng 250 ảnh). Theo quy tắc Day 7, ngưỡng cho phép là 2.0%. Do tỷ lệ lỗi của lô v1 vượt ngưỡng 2% (khoảng 6-8%), quyết định chính thức cho lô v1 là **HOLD**.
- **Vì sao run R-021 phải xem lại hoặc train lại (`action=review_or_retrain`)?**
  Mô hình `R-021` đã được huấn luyện trực tiếp trên tập dữ liệu v1 bị nhiễm lỗi (cả lỗi luật lẫn lỗi gán nhãn sai lệch). Trong học máy, việc sửa dữ liệu và phát hành lô mới v2 không thể tự động sửa được các trọng số (weights) đã được mô hình `R-021` học từ dữ liệu sai. Do đó, hệ thống Lineage đánh dấu `R-021: affected=True`, bắt buộc kỹ sư AI phải đánh giá lại (review) tác động tiêu cực của dữ liệu lỗi lên độ chính xác của model hoặc tiến hành huấn luyện lại (retrain) hoàn toàn trên lô v2 sạch.

## 4. Hash chứng minh gì, không chứng minh gì?
- **Hash chứng minh được:** Tính toàn vẹn dữ liệu (Data Integrity). Mã băm SHA-256 đảm bảo từng file ảnh và file nhãn không bị biến dạng bit, không bị gián đoạn truyền tải, và không bị ai can thiệp/sửa đổi ngoài ý muốn kể từ lúc tạo manifest.
- **Hash KHÔNG chứng minh được:** Tính đúng đắn về mặt ngữ nghĩa của dữ liệu (Label Correctness / Semantic Validity). Một file nhãn hoàn toàn có thể có mã hash nguyên vẹn 100% nhưng nội dung bên trong lại gán nhầm lớp (`class_swap`), lệch box (`bbox_drift`) hoặc sót vật thể (`missing_box`). Hash chỉ bảo vệ tính toàn vẹn của file, không thể thay thế cho khâu Data QC và Human Review.

## 5. Truy vết
- **Tiêu chí phân tầng a/b:**
  - `tier_a`: Các ảnh nằm trong các cảnh (`scene_id`) mà ticket team AI phản ánh trực tiếp, được gán bởi job dùng guideline cũ (v1.2), và có chứa ít nhất một box detection thuộc lớp bị đổi quy tắc (`wine glass`).
  - `tier_b`: Toàn bộ các ảnh còn lại trong lô thỏa mãn cùng điều kiện (job guideline cũ v1.2, có chứa box detection của lớp đổi quy tắc) mà team AI chưa phát hiện ra.
  - `notify`: Mã của người gán (`annotator`) và người kiểm (`reviewer`) phụ trách các job chứa ảnh thuộc tier_a và tier_b (tra cứu từ `jobs.csv`).
- **Vì sao lấy guideline theo job (`jobs.csv`) thay vì metadata ảnh?**
  Bởi vì metadata từng ảnh có thể bị lỗi `guideline_mismatch` (ghi sai nhãn phiên bản guideline). `jobs.csv` là văn bản/hợp đồng giao việc gốc của dự án, phản ánh đúng phiên bản guideline mà annotator được hướng dẫn tại thời điểm bắt đầu job.

## 6. Bug hoặc điểm chưa rõ trong lab (nếu có)
- Không có. Hệ thống script của lab hoạt động nhất quán, phân định rõ ràng giữa quy tắc kiểm tra tự động và khâu rà soát trực quan của con người.

