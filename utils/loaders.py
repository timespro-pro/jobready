import requests
from bs4 import BeautifulSoup
import PyPDF2

def load_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

def load_url_content(urls):
    texts = []
    for url in urls:
        if url:
            try:
                response = requests.get(url, timeout=10)
                soup = BeautifulSoup(response.text, "html.parser")
                texts.append(soup.get_text(separator=" ", strip=True))
            except Exception as e:
                texts.append(f"Error loading {url}: {str(e)}")
    return texts
