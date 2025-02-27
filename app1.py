import streamlit as st
import PyPDF2
import boto3
import json

# Load AWS credentials from Streamlit secrets
AWS_ACCESS_KEY = st.secrets["aws"]["AWS_ACCESS_KEY"]
AWS_SECRET_KEY = st.secrets["aws"]["AWS_SECRET_KEY"]
S3_BUCKET_NAME = "tensorflow-titans-bucket"  # Replace with your S3 bucket name
S3_FOLDER = "job_descriptions/"  # Folder in S3 bucket

# Initialize S3 Client
s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY
)

# Initialize AWS Bedrock Client
bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1",  # Change to your AWS region
)

# Function to upload file to S3
def upload_to_s3(file, filename):
    s3_key = S3_FOLDER + filename  # Store file in the job_descriptions folder
    s3_client.upload_fileobj(file, S3_BUCKET_NAME, s3_key)
    return f"s3://{S3_BUCKET_NAME}/{s3_key}", s3_key

MAX_JOB_DESC_LENGTH = 4000  # Adjust based on token limits

def truncate_text(text, max_length=MAX_JOB_DESC_LENGTH):
    """Truncate text to avoid exceeding the model input size."""
    return text[:max_length] + "..." if len(text) > max_length else text


# Function to retrieve job description from S3
def get_job_description(s3_key):
    obj = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
    content = obj["Body"].read()

    # Try decoding with UTF-8, fallback to other encodings
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        return content.decode("ISO-8859-1")  # Alternative encoding
job_description = truncate_text(job_description)

import boto3
import json

def generate_interview_questions(job_description):
    """
    Sends a structured prompt to Claude 3 Sonnet on AWS Bedrock to generate interview questions.
    """

    # Reinitialize the AWS Bedrock client inside the function
    bedrock = boto3.client(
        service_name="bedrock-runtime",
        region_name="us-east-1",  # Change to your AWS region
        aws_access_key_id=st.secrets["aws"]["AWS_ACCESS_KEY"],
        aws_secret_access_key=st.secrets["aws"]["AWS_SECRET_KEY"]
    )

    prompt = f"""
    **Task:**
    You are an AI assistant skilled in analyzing job descriptions and resumes.
    - Extract the **Job Title** from the Job Description.
    - Extract the **total months of experience** from the resume.
    - Analyze the **Job Description (JD)** and extract **the top 5 most critical skills**.
    - Compare these skills with the **resume** to check matching/missing skills and give a **score from 1 to 10** for eligibility.
    - Generate **3 technical or conceptual questions** based on those extracted JD skills and candidate's experience and their answers.
      - For beginners: Simple definition-based questions.
      - For senior/mid-level roles: More advanced scenario-based questions.
    - Generate **2 project-based questions** from the resume related to past projects or experience that test the candidate’s real-world application of these skills along with their answers.

    **Job Description:**
    {job_description}

    **Resume:**

    **Output Format:**
    ```
    {{
      "JobTitle": "[Job Title extracted from the job description]",
      "Experience": "[Experience in Months extracted from resume]",
      "Score": "[Matching score between the resume and the job description]",
      "Questions": {{
        "Q1": "[Technical question 1]",
        "A1": "[Answer for Q1]",
        "Q2": "[Technical question 2]",
        "A2": "[Answer for Q2]",
        "Q3": "[Conceptual question 3]",
        "A3": "[Answer for Q3]",
        "Q4": "[Technical question 4]",
        "A4": "[Answer for Q4]",
        "Q5": "[Scenario-based question 5]",
        "A5": "[Answer for Q5]"
      }}
    }}
    ```
    """

    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 700,
        "temperature": 0.0,
        "top_p": 0.9
    }

    response = bedrock.invoke_model(
        modelId="anthropic.claude-3-sonnet-20240229-v1:0",
        body=json.dumps(payload)
    )

    response_body = json.loads(response["body"].read().decode("utf-8"))
    return json.loads(response_body["content"][0]["text"])


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
    st.session_state.s3_key = ""

if st.button("Upload to S3"):
    if job_desc_file and candidate_details:
        filename = job_desc_file.name
        with st.spinner("Uploading file..."):
            s3_path, s3_key = upload_to_s3(job_desc_file, filename)
            st.session_state.s3_path = s3_path
            st.session_state.s3_key = s3_key
            st.session_state.uploaded = True
            st.success("File uploaded successfully!")
    else:
        st.error("Please upload a job description and enter a job ID before submitting.")

if st.session_state.uploaded:
    if st.button("Generate Interview Questions"):
        with st.spinner("Fetching job description and generating questions..."):
            job_description_text = get_job_description(st.session_state.s3_key)
            questions_data = generate_interview_questions(job_description_text)
            
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
