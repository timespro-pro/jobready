import streamlit as st
import PyPDF2
import requests
import boto3
import json

# Load AWS credentials from Streamlit secrets
AWS_ACCESS_KEY = st.secrets["aws"]["AWS_ACCESS_KEY"]
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
candidate_details = st.text_area("Enter job ID from the HR portal")

if "uploaded" not in st.session_state:
    st.session_state.uploaded = False
    st.session_state.s3_path = ""

if st.button("Upload to S3"):
    if job_desc_file and candidate_details:
        filename = job_desc_file.name
        with st.spinner("Uploading file..."):
            st.session_state.s3_path = upload_to_s3(job_desc_file, filename)
            st.session_state.uploaded = True
            st.success("File uploaded successfully!")
    else:
        st.error("Please upload a job description and enter a job ID before submitting.")

# Show Generate Questions button only after upload
if st.session_state.uploaded:
    if st.button("Generate Interview Questions"):
        with st.spinner("Triggering AWS Lambda function..."):
            payload = {"s3_path": st.session_state.s3_path, "candidate_details": candidate_details}
            response = requests.post(LAMBDA_URL, json=payload)
            
            if response.status_code == 200:
                questions_data = response.json()
                
                st.subheader(f"Job Title: {questions_data.get('JobTitle', 'N/A')}")
                st.write(f"Experience Required: {questions_data.get('Experience', 'N/A')} months")
                st.write(f"Eligibility Score: {questions_data.get('Score', 'N/A')}/10")
                
                st.subheader("Generated Interview Questions:")
                for i in range(1, 6):
                    question = questions_data.get("Questions", {}).get(f"Q{i}", "")
                    answer = questions_data.get("Questions", {}).get(f"A{i}", "")
                    if question:
                        st.write(f"**Q{i}:** {question}")
                        st.write(f"**A{i}:** {answer}")
                        st.markdown("---")
            else:
                st.error("Error fetching questions from AWS Lambda.")
