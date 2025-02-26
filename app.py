import openai
import PyPDF2
import pandas as pd

# Set OpenAI API key
openai.api_key = "your_openai_api_key"

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text.strip()

def generate_questions(job_description):
    """Generates 10 interview questions based on the job description using GPT-4."""
    
    prompt = f"""
    You are a hiring manager creating interview questions for a job. 
    Based on the following job description, generate 10 relevant and thoughtful interview questions.
    
    ### Job Description:
    {job_description}
    
    The questions should assess technical skills, job-specific knowledge, and behavioral traits.

    Respond ONLY with a numbered list of 10 questions.
    """

    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert recruiter generating job interview questions."},
            {"role": "user", "content": prompt}
        ]
    )

    questions = response.choices[0].message.content.strip()
    return questions.split("\n")  # Splitting numbered questions into a list

# Load job description from PDF
pdf_path = "job_description.pdf"  # Replace with your actual file
job_description = extract_text_from_pdf(pdf_path)

# Generate 10 questions
questions = generate_questions(job_description)

# Save questions to an Excel file
df = pd.DataFrame({"Interview Questions": questions})
df.to_excel("generated_questions.xlsx", index=False)

print("Questions generated and saved to 'generated_questions.xlsx'.")
