import PyPDF2
import cv2
import numpy as np

# เปิดไฟล์ PDF
file_path = "แบบคำขอโอนและรับโอน.pdf"
pdf_file = open(file_path, "rb")

# สร้างตัวอ่าน PDF
pdf_reader = PyPDF2.PdfFileReader(pdf_file)

# รับหมายเลขหน้าที่มีลายเซ็นต์
page_number = int(input("Enter the page number containing the signature: "))

# สร้างอินสแตนซ์หน้า PDF
page = pdf_reader.getPage(page_number - 1)

# แปลงหน้า PDF เป็นรูปภาพ
pdf_writer = PyPDF2.PdfFileWriter()
pdf_writer.addPage(page)
pdf_bytes = io.BytesIO()
pdf_writer.write(pdf_bytes)
pdf_bytes.seek(0)

# อ่านรูปภาพจากไฟล์ PDF
image_bytes = pdf_bytes.read()
image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

# แปลงภาพเป็นสีเทา
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# ใช้เกณฑ์ (threshold) เพื่อแบ่งส่วนภาพลายเซ็นต์
_, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY_INV)

# ตรวจหาลายเซ็นต์ในภาพ
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if len(contours) > 0:
    # รับบริเวณลายเซ็นต์ที่ใหญ่ที่สุด
    max_contour = max(contours, key=cv2.contourArea)

    # ล้อมกรอบลายเซ็นต์
    x, y, w, h = cv2.boundingRect(max_contour)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # แสดงรูปภาพที่มีลายเซ็นต์ที่ตรวจพบ
    cv2.imshow("Detected Signature", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No signature detected on the given page.")