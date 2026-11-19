# Dùng AI

| # | Ai dùng | Công cụ | Prompt (tóm tắt) | AI sai ở đâu | Nhóm kiểm lại thế nào |
|---|---|---|---|---|---|
| 1 | Nguyễn Quang Tùng | Antigravity AI | Viết script tự động hóa đóng gói bàn giao Lô v2 (`build_handover.py`) đạt chuẩn H1-H15 | AI ban đầu đưa cả 80 lớp vào `label_summary.csv` kể cả lớp có count=0, làm `check_handover.py` báo FAIL tại mã H9 do lệch key với nhãn thật. | Nhóm chạy script so khớp dict `summ` và `counts`, phát hiện lỗi và sửa script chỉ lưu các lớp có count > 0; chạy lại `check_handover.py` đạt PASS H9. |
