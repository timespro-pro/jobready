import boto3
from fpdf import FPDF
from io import BytesIO

# AWS S3 Configuration
AWS_ACCESS_KEY = "your-access-key"
AWS_SECRET_KEY = "your-secret-key"
BUCKET_NAME = "your-bucket-name"
OBJECT_NAME = "output.pdf"  # Name of the file in S3

# Create an S3 client
s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY
)

# Generate a PDF in-memory
pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=12)
pdf.cell(200, 10, txt="Hello, this is a sample PDF!", ln=True, align="C")

# Save PDF to a BytesIO buffer
pdf_buffer = BytesIO()
pdf.output(pdf_buffer)
pdf_buffer.seek(0)  # Move the buffer position to the start

# Upload PDF to S3
s3_client.upload_fileobj(pdf_buffer, BUCKET_NAME, OBJECT_NAME, ExtraArgs={'ContentType': 'application/pdf'})

print(f"PDF successfully uploaded to S3: s3://{BUCKET_NAME}/{OBJECT_NAME}")
