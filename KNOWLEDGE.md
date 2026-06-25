Dưới đây là bản kế hoạch tổng thể và cực kỳ chi tiết đi từ lý thuyết giao thông, toán học động lực học cho đến kỹ thuật lập trình API Blender.
*https://docs.blender.org/*

Lưu ý về việc **"không quan sát vùng bên trong giao lộ"**. Về mặt kỹ thuật, nó biến ngã tư thành một **Hộp đen (Black Box)**. Phương tiện sẽ biến mất ở camera đầu vào (In-camera), trải qua một khoảng trễ thời gian (Time Delay) tỷ lệ thuận với vận tốc, rồi xuất hiện lại ở camera đầu ra (Out-camera). Điều này giúp giảm bớt việc phải dựng mô hình xoay cua phức tạp, nhưng lại yêu cầu tính toán thời gian xuất hiện cực kỳ chính xác.

$$\text{Tổng chiều rộng một trục đường} = (4 \text{ làn} \times 3.5\text{ m}) + 2.0\text{ m (Dải phân cách)} + (4 \text{ làn} \times 3.5\text{ m}) = 30\text{ mét}$$
Kích Thước "Hộp Đen" Giao Lộ (Intersection Box): Lòng giao lộ sẽ là một hình vuông có kích thước 30m x 30m

---

## GIAI ĐOẠN 1: TOÁN HỌC VÀ LOGIC DI CHUYỂN (TRAFFIC KINEMATICS)

*Mục tiêu: Làm chủ các quy luật vật lý của phương tiện để viết code di chuyển chuẩn xác.*

### 1.1. Biên giới của tầm nhìn (Vạch dừng xe)

* **Camera Vào (In):** Chỉ ghi hình đoạn đường tiếp cận từ xa cho đến **Vạch dừng đèn đỏ (Stop Line)**.
* **Camera Ra (Out):** Chỉ ghi hình đoạn đường tính từ **Vạch đi bộ phía đối diện (Crosswalk)** kéo dài ra xa.
* **Vùng mù (Blind Zone):** Chính là lòng giao lộ nơi các luồng xe giao cắt.

### 1.2. Công thức tính thời gian trễ trong vùng mù

Khi xe đi hết làn vào, nó biến mất. Thời gian xe "tàng hình" trong giao lộ trước khi xuất hiện ở làn ra được tính bằng công thức động học cơ bản:

$$\Delta t = \frac{d_{\text{intersection}}}{v}$$

* $v$: Vận tốc của xe khi đi qua giao lộ (thường là vận tốc đều ổn định vì xe không dừng giữa ngã tư trừ khi kẹt).
* **Ví dụ:** Xe đi với vận tốc $40 \text{ km/h} \approx 11.1 \text{ m/s}$, qua ngã tư rộng $16\text{m}$ sẽ mất khoảng $1.44\text{ s}$. Ở tốc độ 24 fps, xe sẽ phải "tàng hình" đúng 35 frames trước khi xuất hiện ở video đầu ra.

---

## GIAI ĐOẠN 2: KIẾN THỨC NỀN TẢNG BLENDER CHO LẬP TRÌNH VIÊN

*Mục tiêu: Hiểu cách Blender quản lý dữ liệu dưới dạng code để thao tác mà không cần mở UI.*

### 2.1. Cấu trúc dữ liệu của Blender (`bpy.data`)

Blender quản lý mọi thứ theo dạng khối dữ liệu (Data-blocks). Bạn cần nắm chắc 3 khái niệm:

* **Objects (`bpy.data.objects`):** Các thực thể có vị trí, góc xoay trong không gian (Camera, Xe, Mặt đường).
* **Meshes (`bpy.data.meshes`):** Phần lõi hình học (vỏ xe, biển số) nằm trong Object.
* **Materials (`bpy.data.materials`):** Vật liệu quy định màu sắc, độ bóng, kết cấu ảnh (Texture) của biển số.

### 2.2. Hệ trục tọa độ và Đơn vị

* Blender mặc định dùng hệ tọa độ **Right-Handed Z-Up** (Trục X sang phải, Trục Y ra phía sau, Trục Z lên trời).
* Đơn vị mặc định là **Meter (Mét)** và **Radian** (cho góc xoay). Bạn luôn phải dùng hàm `math.radians(độ)` khi viết code xoay camera hoặc xe.

---

## GIAI ĐOẠN 3: LÀM CHỦ BLENDER PYTHON API (`bpy`)

*Mục tiêu: Viết script tự động hóa toàn bộ vòng đời của dữ liệu.*

### 3.1. Kỹ thuật Keyframing qua Code

Để xe di chuyển, bạn không đổi vị trí xe ngẫu nhiên mà phải gán tọa độ vào dòng thời gian (Timeline) và chọn kiểu nội suy (Interpolation):

* **Linear (Tịnh tiến đều):** Dùng cho xe đang chạy tốc độ ổn định.
* **Bezier (Mượt mà):** Dùng cho xe đang giảm tốc vạch dừng hoặc tăng tốc rời nút.


### 3.2. Dynamic Texture Mapping (Đổi biển số tự động)

Để phục vụ LPR, mỗi xe sinh ra phải có một biển số độc nhất.

* **Logic:** Code sẽ tìm vật liệu tên là `LicensePlate_Mat`, truy cập vào nút `Image Texture` của nó và nạp một file ảnh biển số mới vào trước khi render khung hình tiếp theo.

---

## GIAI ĐOẠN 4: ĐẶT VỊ TRÍ CAMERA

*Mục tiêu: Cấu hình camera triệt tiêu vùng mù đồ họa, ép góc nhìn giống CCTV thực tế.*

### 4.1. Quang học Camera (Focal Length)

* Đây là đặc trưng của **Ống kính tiêu cự dài (Telephoto Lens)**.
* **Cấu hình Code:** Đặt thuộc tính `camera.data.lens = 60` hoặc `85` (thay vì 35 mặc định). Tiêu cự lớn giúp ép dẹt không gian, làm biển số xe hiển thị rõ ràng hơn, rất có lợi cho việc test mô hình LPR.

### 4.2. Cắt sơ đồ khung hình (Frustum Clipping)

Để camera chỉ nhìn thấy làn đường vào/ra mà không nhìn thấy lòng giao lộ:

* Đặt camera lùi lại phía sau vạch dừng xe.
* Điều chỉnh góc chúi (Pitch) vừa đủ để cạnh dưới của khung hình video trùng khít với vạch dừng xe (đối với Camera In) hoặc vạch đi bộ (đối với Camera Out).

---

## GIAI ĐOẠN 5: XÂY DỰNG DATA PIPELINE HOÀN CHỈNH

*Mục tiêu: Tạo ra một hệ thống sinh dữ liệu khép kín tự động.*

```
[Bảng tham số xe/biển số] ──> [Script Python] ──> [Blender Headless] ──> [8 Video .mp4] + [File Mapping JSON]

```

Để hoàn thiện dự án, file code cuối cùng sẽ quản lý một cấu trúc logic như sau:

1. **Hàm sinh kịch bản (Scenario Generator):** Tạo ra một danh sách xe ngẫu nhiên (Xe 1: màu đỏ, biển số X, xuất phát lúc frame 1 hướng Nam đi thẳng; Xe 2: màu xanh, biển số Y, xuất phát lúc frame 20 hướng Đông rẽ phải...).
2. **Hàm chạy mô phỏng (Simulation Runner):** Áp dụng các công thức ở Giai đoạn 1 để tính toán tọa độ xuất hiện, biến mất của từng xe trên các làn đường.
3. **Hàm Render & Xuất Metadata:** Render ra 8 luồng video. Đồng thời xuất ra một file `metadata.json` lưu chính xác vị trí $XYZ$ của từng chiếc xe tại mỗi khung hình để làm dữ liệu đối chứng (Ground Truth) tuyệt đối cho hệ thống kiểm thử sau này.

---

Giai đoạn đầu tiên cần thực hiện ngay là **chuẩn bị tài nguyên hình khối 3D (Asset)** bằng cách tìm hiểu và đọc đúng các file 3D định dạng của xe ô tô thực tế  trong folder models