import aspose.pdf as ap

# สร้างวัตถุ PdfFileSignature
pdfSign = ap.facades.PdfFileSignature()

# ผูกวัตถุ PdfFileSignature กับ PDF
pdfSign.bind_pdf("แบบคำขอโอนและรับโอน.pdf")

# ตรวจสอบลายเซ็น
if (pdfSign.verify_signature("Signature1")):
    print("Verified...")
