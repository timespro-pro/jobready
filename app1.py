import streamlit as st
import boto3
import os
from botocore.exceptions import NoCredentialsError

# Load AWS credentials from Streamlit secrets
st.secrets["aws"]
AWS_ACCESS_KEY_ID = st.secrets["aws"]["AWS_ACCESS_KEY"]
AWS_SECRET_ACCESS_KEY = st.secrets["aws"]["AWS_SECRET_ACCESS"]
AWS_REGION = st.secrets["aws"]["REGION_NAME"]

def upload_to_s3(file, bucket_name, job_id):
    """Uploads a file to S3 with a structured filename including the job ID."""
    s3 = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION
    )
    
    file_extension = file.name.split('.')[-1]
    s3_filename = f"uploads/{job_id}.{file_extension}"
    
    try:
        s3.upload_fileobj(file, bucket_name, s3_filename)
        file_url = f"https://{bucket_name}.s3.amazonaws.com/{s3_filename}"
        return file_url
    except NoCredentialsError:
        st.error("AWS credentials not found. Please configure them correctly.")
        return None

# Streamlit UI
st.title("PDF Upload to S3 with Job ID")

bucket_name = "your-s3-bucket-name"  # Replace with your actual S3 bucket name

# Job ID input
job_id = st.text_input("Enter Job ID:", "")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF File", type=["pdf"])

if uploaded_file and job_id:
    if st.button("Upload to S3"):
        file_url = upload_to_s3(uploaded_file, bucket_name, job_id)
        if file_url:
            st.success(f"File successfully uploaded! [View File]({file_url})")
else:
    st.warning("Please enter a Job ID and upload a PDF file.")
