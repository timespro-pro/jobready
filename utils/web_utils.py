import requests
from bs4 import BeautifulSoup

def extract_text_from_url(url: str) -> str:
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text(separator=" ", strip=True)
        return text
    except Exception as e:
        return f"Error fetching URL: {e}"
