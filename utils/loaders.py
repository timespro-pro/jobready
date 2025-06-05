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
    from bs4 import BeautifulSoup
    import requests

    url_contexts = {}
    for url in urls:
        try:
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.text, "html.parser")
            text = soup.get_text(separator=" ", strip=True)
            url_contexts[url] = text[:3000]  # truncate to 3000 chars if needed
        except Exception as e:
            url_contexts[url] = f"Error fetching URL content: {e}"
    return url_contexts

