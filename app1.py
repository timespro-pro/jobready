import streamlit as st
import boto3
import json
import PyPDF2
from io import BytesIO
from botocore.exceptions import NoCredentialsError, ClientError

# Load AWS Credentials from Streamlit Secrets
AWS_ACCESS_KEY = st.secrets["aws"]["AWS_ACCESS_KEY"]
AWS_SECRET_KEY = st.secrets["aws"]["AWS_SECRET_KEY"]
REGION_NAME = st.secrets["aws"]["REGION_NAME"]
#S3_BUCKET_NAME = st.secrets["aws"]["S3_BUCKET_NAME"]

# Initialize AWS Clients
bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")
s3 = boto3.client("s3", region_name="us-east-1")

# S3 Bucket Name (Change this to your S3 bucket)
S3_BUCKET_NAME ="tensorflow_titans_job"

# Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text.strip()

# Function to generate interview questions using Claude 3
def generate_interview_questions(job_description, resume):
    prompt = f"""
    You are an AI assistant skilled in analyzing job descriptions and resumes.
    - Extract the **Job Title** from the Job Description.
    - Extract the **total months of experience** from the resume.
    - Identify the **top 5 critical skills** from the JD.
    - Compare these skills with the **resume** and give a **matching score (1-10)**.
    - Generate **5 technical/conceptual questions** and **5 project-based questions** with answers.

    **Job Description:**
    {job_description}

    **Resume:**
    {resume}

    **Output Format:**
    ```json
    {{
      "JobTitle": "[Extracted Job Title]",
      "Experience": "[Extracted Experience in Months]",
      "Score": "[Matching Score]",
      "Questions": {{
        "Q1": "[Question 1]", "A1": "[Answer 1]",
        "Q2": "[Question 2]", "A2": "[Answer 2]",
        "Q3": "[Question 3]", "A3": "[Answer 3]",
        "Q4": "[Question 4]", "A4": "[Answer 4]",
        "Q5": "[Question 5]", "A5": "[Answer 5]",
        "Q6": "[Project Question 6]", "A6": "[Answer 6]",
        "Q7": "[Project Question 7]", "A7": "[Answer 7]",
        "Q8": "[Project Question 8]", "A8": "[Answer 8]",
        "Q9": "[Project Question 9]", "A9": "[Answer 9]",
        "Q10": "[Project Question 10]", "A10": "[Answer 10]"
      }}
    }}
    ```
    """

    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 800,
        "temperature": 0.5,
        "top_p": 0.9
    }

    try:
        response = bedrock.invoke_model(
            modelId="anthropic.claude-3-sonnet-20240229-v1:0",
            body=json.dumps(payload)
        )
        response_body = json.loads(response["body"].read().decode("utf-8"))
        output_text = response_body["content"][0]["text"]
        return json.loads(output_text)

    except Exception as e:
        st.error(f"Error generating interview questions: {str(e)}")
        return None

# Function to upload generated questions to S3
def upload_to_s3(job_id, job_description, questions):
    file_name = f"interview_questions_{job_id}.json"
    data = {
        "JobID": job_id,
        "JobDescription": job_description,
        "Questions": questions
    }
    json_data = json.dumps(data, indent=4)

    try:
        s3.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=file_name,
            Body=json_data,
            ContentType="application/json"
        )
        return f"File uploaded successfully: {S3_BUCKET_NAME}/{file_name}"
    except NoCredentialsError:
        return "AWS Credentials not found."
    except ClientError as e:
        return f"Error uploading to S3: {e}"

# Streamlit UI
st.title("AI-Powered Interview Question Generator")

# Job ID input field
job_id = st.text_input("Enter Job ID:", "")

# File Uploaders
job_desc_file = st.file_uploader("Upload Job Description (PDF)", type=["pdf"])
resume_files = st.file_uploader("Upload Resumes (PDF)", type=["pdf"], accept_multiple_files=True)

# Process Uploaded Files
if job_desc_file and resume_files and job_id:
    job_description = extract_text_from_pdf(job_desc_file)
    resumes_text = "\n\n".join([extract_text_from_pdf(file) for file in resume_files])

    if st.button("Generate Interview Questions"):
        with st.spinner("Generating questions..."):
            generated_data = generate_interview_questions(job_description, resumes_text)

            if generated_data:
                st.subheader("Generated Interview Questions:")
                for i in range(1, 11):
                    st.markdown(f"**Q{i}:** {generated_data['Questions'][f'Q{i}']}")
                    st.write(f"**A{i}:** {generated_data['Questions'][f'A{i}']}")

                # Save to S3
                if st.button("Save to S3"):
                    with st.spinner("Saving..."):
                        s3_result = upload_to_s3(job_id, job_description, generated_data["Questions"])
                        st.success(s3_result)

else:
    st.warning("Please upload files and enter Job ID to proceed.")

