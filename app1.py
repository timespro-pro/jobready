import streamlit as st
import boto3
import os
from botocore.exceptions import NoCredentialsError

# Load AWS credentials from Streamlit secrets
AWS_ACCESS_KEY_ID = st.secrets["aws"]["AWS_ACCESS_KEY"]
AWS_SECRET_ACCESS_KEY = st.secrets["aws"]["AWS_SECRET_KEY"]
AWS_REGION = st.secrets["aws"]["REGION_NAME"]

def upload_to_s3(file, job_id):
    """Welcome to JobReady AI question generator"""
    bucket_name = "tensorflow-titans-bucket"
    s3 = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION
    )
    
    file_extension = file.name.split('.')[-1]
    s3_filename = f"job_descriptions/{job_id}.{file_extension}"
    
    try:
        s3.upload_fileobj(file, bucket_name, s3_filename)
        file_url = f"https://{bucket_name}.s3.amazonaws.com/{s3_filename}"
        return file_url
    except NoCredentialsError:
        st.error("AWS credentials not found. Please configure them correctly.")
        return None

# Streamlit UI
st.title("Welcome to JobReady: AI question generator")

bucket_name = "tensorflow_titans_job"  # Updated bucket name

# Job ID input
job_id = st.text_input("Enter Job ID from the SenseHQ portal:", "")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF File", type=["pdf"])

if uploaded_file and job_id:
    if st.button("Upload to S3"):
        file_url = upload_to_s3(uploaded_file, job_id)
        if file_url:
            st.success(f"File successfully uploaded! [View File]({file_url})")
else:
    st.warning("Please enter a Job ID and upload a PDF file.")
