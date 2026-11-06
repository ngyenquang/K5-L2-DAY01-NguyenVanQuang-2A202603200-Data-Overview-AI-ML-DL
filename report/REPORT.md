# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 06/11/2026

**Runtime Colab:** CPU

**Python / PyTorch / Ultralytics:** 3.13.18 / 2.11.0+cpu / 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)


## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?

   **Trả lời:** Record này nói về **cả bức ảnh**. Trong record chỉ có thông tin ảnh (`sample_id`=traffic, kích thước 640 × 428) cùng lớp, `rank` và `score`; không có `bbox_xyxy`, polygon hay `instance_id`, nên không biết vật nằm ở đâu hay có bao nhiêu vật. Ảnh `traffic` có 5 record, ứng với rank 1–5 (`cab` 0.51, `minibus` 0.16, `police_van` 0.09, `recreational_vehicle` 0.05, `streetcar` 0.05). Đây là top-5 lớp cho toàn ảnh, không phải 5 vật thể khác nhau.

3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?

   **Trả lời:** Danh sách lớp **đi kèm taxonomy/checkpoint**. `yolo11n-cls.pt` được huấn luyện trên ImageNet-1K, nên đầu ra luôn là một trong 1000 lớp cố định của taxonomy này. Notebook lấy tên lớp từ `result.names[class_id]`, tức là tra từ bảng tên lưu sẵn trong checkpoint, không phải model tự đặt tên.

4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?

   **Trả lời:** Vì tên lớp đứng một mình dễ mơ hồ. Cùng một ID hay một tên có thể mang nghĩa khác nhau ở các taxonomy khác nhau. Ví dụ `cab` (ID 468) có trong ImageNet-1K, nhưng COCO-80 mà model detection/segmentation dùng không có lớp `cab`, chỉ có `car`. Mỗi trường có vai trò riêng: `class_id` là mã cho máy đọc, `class_name` để người đọc hiểu, còn `taxonomy_name` cho biết ID đó tra trong bảng nào. Có đủ cả ba thì mới so sánh hay ghép nhãn giữa các dataset/model mà không nhầm.

5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?

   **Trả lời:** Guideline cần nói rõ:
   - Tiêu chí chọn **chủ thể chính**: vật lớn nhất, nằm ở tiền cảnh hoặc ở trung tâm bố cục.
   - Thứ tự ưu tiên khi có hai chủ thể ngang nhau.
   - Cách chọn giữa các lớp gần nghĩa. Ví dụ ảnh `traffic` có `cab`, `minibus`, `police_van` đều là xe, nên phải quy định chọn lớp cụ thể nhất mà người gán chắc chắn.
   - Khi nào ảnh không có chủ thể rõ ràng thì phải escalation thay vì đoán.
   - Kèm ví dụ minh họa đúng/sai.

6. Vì sao không được lấy `score` của model làm ground truth?

   **Trả lời:** Vì `score` chỉ là mức "tự tin" của model, không phải sự thật; model có thể tự tin mà vẫn sai. Ở ảnh `traffic`, top-1 `cab` chỉ đạt 0.51, nghĩa là gần một nửa xác suất còn lại thuộc về các lớp khác. Nếu lấy prediction làm ground truth thì lỗi của model sẽ quay lại thành dữ liệu huấn luyện, và cũng không còn thước đo độc lập nào để đánh giá model. Ground truth phải do người gán theo guideline rồi qua QC.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=person ; `score`=0.912625 ; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92] ; `bbox_width`=113.58 ; `bbox_height`=279.68
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?

   **Trả lời:** Ảnh `kitchen` rộng 640 px, cao 427 px. Tọa độ tính bằng pixel, gốc (0, 0) ở góc trên-trái ảnh, x tăng sang phải, y tăng xuống dưới.
   - Góc trên-trái của box ở (385.33, 69.24): cách mép trái khoảng 385 px và cách mép trên khoảng 69 px.
   - Góc dưới-phải ở (498.92, 348.92).
   - Box rộng 498.92 − 385.33 = 113.58 px và cao 348.92 − 69.24 = 279.68 px.

   Đây là một hình chữ nhật đứng ôm người đầu bếp quay lưng ở phía bên phải ảnh, từ đỉnh đầu xuống tới chân.

3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?

   **Trả lời:** Theo output ô so sánh ngưỡng trong notebook:
   - **Ngưỡng 0.20:** 17 prediction (`person` ×2, `bowl` ×5, `oven` ×2, `cup` ×2, `spoon` ×3, `potted plant`, `dining table`, `bottle`).
   - **Ngưỡng 0.60:** 6 prediction (`person` ×2, `bowl` ×2, `oven` ×2).
   - Để tham khảo, ngưỡng 0.35 dùng khi tạo JSON cho 11 prediction.

   Hạ ngưỡng từ 0.60 xuống 0.20 làm số hộp tăng gần gấp ba. Phần thêm vào là các vật model kém tự tin như `spoon`, `bottle`, `dining table`. Ngược lại, ở 0.60 thì mất cả hai cái `cup` (score 0.45 và 0.38), dù đó là vật thật trên bàn.

4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?

   **Trả lời:** Reviewer phải:
   - **Kiểm tra từng hộp có score thấp** xem có phải false positive không. Ví dụ `person` 0.61 ở mép trái, bbox [0.12, 263.18, 61.55, 310.78], thực chất chỉ là một bàn tay thò vào khung. `bowl` 0.46 trên kệ phía trên cũng chưa chắc là bowl.
   - **Loại hộp trùng** trên cùng một vật.
   - **Sửa nhãn lớp sai**, ví dụ bowl, cup hay khuôn bánh, và **sửa box lệch** hoặc chưa ôm sát.

   Ở ngưỡng 0.20 có 17 hộp, so với 6 hộp ở 0.60, nên khối lượng review tăng rõ rệt và cần quy tắc rõ ràng để quyết định giữ hay loại một hộp.

5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).

   **Trả lời:** Vẽ một hộp chữ nhật ôm sát **phần nhìn thấy được** của mỗi vật thuộc danh sách lớp: bốn cạnh chạm pixel ngoài cùng của vật, không chừa nền và không cắt mất phần nào của vật. Mỗi vật một hộp riêng, kể cả khi các vật cùng lớp nằm sát nhau. Không ước đoán phần bị che hoặc phần nằm ngoài ảnh; vật bị cắt mép thì cạnh hộp đặt đúng tại mép ảnh.

6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?

   **Trả lời:** Vật bị cắt mép là **người ở mép trái ảnh** (`person`, score 0.61, bbox [0.12, 263.18, 61.55, 310.78]). Cạnh trái của box gần như bằng 0, tức là sát mép ảnh; trên ảnh chỉ thấy bàn tay và cánh tay thò vào, phần thân nằm ngoài khung.

   Tôi chọn **viết thành quy tắc trong guideline**, vì vật bị cắt mép xuất hiện ở rất nhiều ảnh và có thể quy định thống nhất: vẫn gán nhãn nếu nhận ra được lớp, box chỉ ôm phần nhìn thấy, cạnh box đặt tại mép ảnh. Chỉ cần escalation một lần để Lab Coach chốt mức tối thiểu (ví dụ chỉ thấy bàn tay thì có tính là `person` không), rồi bổ sung câu trả lời vào guideline.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=kitchen-001 ; `class_name`=person ; `score`=0.899318 ; số điểm polygon=`polygon_point_count`=348 ; 3–5 điểm đầu của `polygon_xy`=[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?

   **Trả lời:** Box của `kitchen-001` là hình chữ nhật [385.45, 66.44, 498.02, 348.58]. Hình chữ nhật này chứa cả phần nền, như tường và móc treo nồi ở cạnh khuỷu tay và giữa hai chân. Polygon gồm 348 điểm bám theo **đường viền thật** của người (đầu, vai, khuỷu tay chống hông, khe giữa hai chân), nên cho biết chính xác pixel nào thuộc vật và pixel nào là nền. Nhờ đó biết được hình dạng, tư thế và diện tích thật của vật, và tách được các vật chồng lấn nhau. Rõ nhất là `dining table` (`kitchen-005`): box phủ gần nửa dưới ảnh, còn polygon 598 điểm mới thể hiện đúng mặt bàn và chân bàn.

3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?

   **Trả lời:** `instance_id` dùng để phân biệt **từng vật riêng** trong một ảnh, kể cả các vật cùng lớp. Ảnh `kitchen` có 11 instance, từ `kitchen-001` đến `kitchen-011`, trong đó có 4 cái `bowl` (`kitchen-002`, `-003`, `-006`, `-010`). Cả 4 đều có `class_id` = 45, nhưng mỗi cái có `instance_id` và polygon riêng.
   - **Không phải class ID:** class ID cho biết vật "là loại gì" (45 = bowl), còn instance ID cho biết đó "là cái nào".
   - **Không phải tracking ID:** notebook tạo ID theo dạng `{sample_id}-{số thứ tự}`, chỉ là thứ tự của vật trong output của một ảnh. Sang ảnh hoặc frame khác, hoặc chạy lại model, thì cùng một ID có thể chỉ một vật khác, nên không dùng để theo dõi một vật qua thời gian.

4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.

   **Trả lời:** Đặt điểm polygon dọc theo **biên nhìn thấy được** của vật, thêm điểm ở mọi góc hoặc chỗ đổi hướng. Mask chỉ phủ pixel của vật: khoét bỏ phần nền lọt qua (vd. khe giữa tay và thân, giữa hai chân), không phủ sang vật khác. Không vẽ đoán phần bị che. Hai vật chạm nhau thì phần biên chung chỉ thuộc về một vật, mask không được chồng lấn.

5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?

   **Trả lời:** Chỗ khó là **các `bowl` chồng lên nhau trên `dining table`** ở góc dưới-trái ảnh: `kitchen-002` (0.74), `kitchen-003` (0.70) và `kitchen-006` (0.53). Các tô chạm và che nhau. `kitchen-006` có bbox bắt đầu từ x = 0.11, tức là bị cắt mép trái. Cả ba nằm trong vùng mask của `dining table` (`kitchen-005`).
   - Việc vẽ biên giữa hai vật chạm nhau thì dùng **guideline**: vật nằm trên hoặc phía trước giữ phần biên chung, mask không chồng lấn. Đây là tình huống phổ biến, quy định thống nhất được.
   - Việc xác định các vật tròn này là `bowl` hay khuôn bánh thì cần **escalation**. Đây là câu hỏi về ranh giới lớp mà guideline chưa định nghĩa; nếu mỗi annotator tự quyết thì nhãn sẽ không nhất quán.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn cho cả ảnh: `class_id` + `class_name` + `taxonomy_name` (vd. 468 / `cab` / ImageNet-1K), không có score | Ảnh `traffic`: top-1 `cab` chỉ 0.51; các lớp sau gần nghĩa (`minibus` 0.16, `police_van` 0.09); ảnh có nhiều xe nên khó chọn một lớp | Xác định chủ thể chính theo guideline, chọn một lớp trong taxonomy, không chép top-1 của model; không chắc thì escalation | Lớp có đúng chủ thể chính và đúng taxonomy không; `class_id` có khớp `class_name` không; các annotator có chọn nhất quán giữa các lớp gần nghĩa không |
| Phát hiện vật thể | Mỗi vật một dòng: lớp + `bbox_xyxy` (pixel, gốc trên-trái) ôm sát phần nhìn thấy, không có score | Ảnh `kitchen`: `person` 0.61 ở mép trái chỉ là bàn tay; `bowl`/`cup` ở góc dưới-trái chồng nhau; số hộp đổi mạnh theo ngưỡng (17 ở 0.20, 11 ở 0.35, 6 ở 0.60) | Vẽ box cho mọi vật thuộc danh sách lớp, kể cả vật nhỏ, bị che hoặc cắt mép; mỗi vật một box | Có sót vật hoặc thừa/trùng box không; nhãn lớp có đúng không; box có sát không; sai thì trả về rework |
| Instance segmentation | Mỗi vật một instance: `instance_id` + lớp + polygon bám biên; vật cùng lớp vẫn tách riêng | Ảnh `kitchen`: 4 `bowl` sát nhau nằm trên `dining table` (`kitchen-005`); `person` 0.43 (`kitchen-009`) chỉ là bàn tay; `oven` 0.49 (`kitchen-007`) có biên tối, khó thấy | Vẽ polygon theo biên nhìn thấy, khoét phần nền, không vẽ phần bị che, mask không chồng lấn | Biên có sát không; có gộp nhầm hai vật hoặc tách nhầm một vật không; lớp có đúng không |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)

   **Trả lời:** Chỉ dùng ba ảnh COCO công khai có sẵn trong notebook (`traffic`, `kitchen`, `dining`, giấy phép CC BY 2.0, ghi trong `IMAGE_ATTRIBUTION.md`); không tải ảnh cá nhân, khuôn mặt, biển số, dữ liệu khách hàng hay dữ liệu nội bộ lên Colab/GitHub; không ghi họ tên, MSSV hay thông tin cá nhân vào báo cáo/output.

2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?

   **Trả lời:** Dừng ngay, không upload hay push dữ liệu đó, và báo **GV/Lab Coach**. Nếu đã lỡ push lên GitHub thì xóa khỏi báo cáo/output rồi báo GV/Lab Coach.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] REPORT.md` (file này đã điền)
- [x] day1_lab_outputs/classification_predictions.json
- [x] day1_lab_outputs/detection_predictions.json
- [x] day1_lab_outputs/segmentation_predictions.json
- [x] day1_lab_outputs/IMAGE_ATTRIBUTION.md
- [x] day1_lab_outputs/visuals/classification_top5.png
- [x] day1_lab_outputs/visuals/detection_predictions.png
- [x] day1_lab_outputs/visuals/segmentation_prediction.png
- [x] Ô validation cuối notebook báo `PASS
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [x] Đã nộp link repository trên LMS
