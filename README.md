# 📚 PDF + Web Chatbot

This chatbot lets you upload a PDF, input up to 2 URLs, and ask a question. It processes all inputs and generates an answer using an LLM via LangChain.

## 🚀 How to Run

### 1. Clone or unzip the repo
```bash
git clone https://github.com/yourusername/pdf-web-chatbot.git
cd pdf-web-chatbot/chatbot_app
```

### 2. Install requirements
```bash
pip install -r requirements.txt
```

### 3. Set your OpenAI key
Edit `credentials/config.yaml` and replace `"your-api-key"` with your actual OpenAI key.

### 4. Run the app
```bash
streamlit run app.py
```

## 🛠️ Features
- Upload and parse PDF files
- Scrape text from 2 URLs
- Ask analytical questions across all data sources
- Powered by OpenAI and LangChain

---
