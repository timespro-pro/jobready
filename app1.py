import streamlit as st
import PyPDF2
import requests
import boto3

# Load AWS credentials from Streamlit secrets
AWS_ACCESS_KEY = st.secrets["aws"]["AWS_ACCESS_KEY "]
AWS_SECRET_KEY = st.secrets["aws"]["AWS_SECRET_KEY"]
S3_BUCKET_NAME = "tensorflow-titans-bucket"  # Replace with your S3 bucket name
LAMBDA_URL = "YOUR_LAMBDA_FUNCTION_URL"  # Replace with your AWS Lambda endpoint
S3_FOLDER = "job_descriptions/"  # Folder in S3 bucket

# Initialize S3 Client
s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY
)

# Function to upload file to S3
def upload_to_s3(file, filename):
    s3_key = S3_FOLDER + filename  # Store file in the job_descriptions folder
    s3_client.upload_fileobj(file, S3_BUCKET_NAME, s3_key)
    return f"s3://{S3_BUCKET_NAME}/{s3_key}"

# Streamlit UI
st.title("AI Interview Question Generator")

st.write("Upload a job description and enter additional details to generate interview questions.")

# Job Description Upload Button
job_desc_file = st.file_uploader("Upload Job Description (PDF)", type=["pdf"])

# Additional input field
candidate_details = st.text_area("Enter job id from the HR portal ")

uploaded = False
s3_path = ""

if st.button("Upload to S3"):
    if job_desc_file and candidate_details:
        filename = job_desc_file.name
        with st.spinner("Uploading file..."):
            s3_path = upload_to_s3(job_desc_file, filename)
            uploaded = True
            st.success("File uploaded successfully!")
    else:
        st.error("Please upload a job description and enter a job ID before submitting.")

# Show Generate Questions button only after upload
if uploaded or s3_path:
    if st.button("Generate Interview Questions"):
        with st.spinner("Triggering AWS Lambda function..."):
            payload = {"s3_path": s3_path, "candidate_details": candidate_details}
            response = requests.post(LAMBDA_URL, json=payload)
            
            if response.status_code == 200:
                questions = response.json().get("questions", [])
                st.subheader("Generated Interview Questions:")
                for question in questions:
                    st.write(question)
            else:
                st.error("Error fetching questions from AWS Lambda.")
