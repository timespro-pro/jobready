import streamlit as st
import boto3
import os
import PyPDF2
import openai
from botocore.exceptions import NoCredentialsError

# Load AWS credentials from Streamlit secrets
AWS_ACCESS_KEY_ID = st.secrets["aws"]["AWS_ACCESS_KEY"]
AWS_SECRET_ACCESS_KEY = st.secrets["aws"]["AWS_SECRET_KEY"]
AWS_REGION = st.secrets["aws"]["REGION_NAME"]

openai.api_key = st.secrets["openai"]["api_key"]

def upload_to_s3(file, job_id):
    """Uploads the file to S3 and returns the file URL."""
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

def extract_text_from_pdf(pdf_file):
    """Extracts text from an uploaded PDF file."""
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text.strip()

def generate_questions(job_description, job_id):
    """Generates 5 interview questions using GPT-4 based on the job description and job ID."""
    prompt = f"""
    You are a hiring manager creating interview questions for a job candidate.
    Based on the following job description and job ID, generate 5 relevant and thoughtful interview questions.
    
    ### Job Description:
    {job_description}
    
    ### Job ID:
    {job_id}
    
    The questions should assess technical skills, job-specific knowledge, and behavioral traits.
    
    Respond ONLY with a numbered list of 5 questions.
    """
    
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert recruiter and content generator generating job interview questions."},
            {"role": "user", "content": prompt}
        ]
    )
    
    questions = response.choices[0].message.content.strip()
    return questions.split("\n")

# Streamlit UI
st.title("JobReady HR portal ")

bucket_name = "tensorflow_titans_job"  # Updated bucket name

# Job ID input
job_id = st.text_input("Enter Job ID from the SenseHQ portal:", "")

# File uploader
uploaded_file = st.file_uploader("Upload the job description PDF file", type=["pdf"])

if uploaded_file and job_id:
    if st.button("Upload file to JobReady system"):
        file_url = upload_to_s3(uploaded_file, job_id)
        if file_url:
            st.success(f"File successfully uploaded! [View File]({file_url})")

    if st.button("Generate Interview Questions"):
        with st.spinner("Generating questions..."):
            job_description = extract_text_from_pdf(uploaded_file)
            questions = generate_questions(job_description, job_id)
            st.subheader("Generated Interview Questions:")
            for question in questions:
                st.write(question)

    #if st.button("Save question and generate video link"):
    #    st.success("Video link will be sent shortly")
    #    st.image("https://i.ibb.co/G3T9xPKY/download.jpg")

    if st.button("Save question and generate video link"):
        st.success("Interview Video link will be sent shortly")
    
        # Custom CSS to display the image in a controlled size
        st.markdown(
            """
            <style>
            .interview-image-container {
                display: flex;
                justify-content: center;
                align-items: center;
                height: auto;
                margin-top: 20px;
            }
            .interview-image {
                max-width: 80%;
                height: auto;
                border-radius: 10px;
                box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
            }
            </style>
            
            <div class="interview-image-container">
                <img src="https://i.ibb.co/LdYPdkYB/pexels-mjlo-2872418.jpg" class="interview-image">
            </div>
            """,
            unsafe_allow_html=True
        )


else:
    st.warning("Please enter a Job ID and upload a PDF file.")
