import requests
from bs4 import BeautifulSoup
import PyPDF2

def load_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def load_url_content(urls: list) -> dict:
    url_contexts = {}
    headers = {"User-Agent": "Mozilla/5.0"}   # helps avoid some 403s
    for url in urls:
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            soup = BeautifulSoup(resp.text, "html.parser")
            # NO TRUNCATION  – grab full visible text
            url_contexts[url] = soup.get_text(separator=" ", strip=True)
        except Exception as e:
            url_contexts[url] = f"Error fetching URL content: {e}"
    return url_contexts

