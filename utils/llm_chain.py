from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI  # ✅ Correct import for chat models
import os

def get_combined_response(pdf_text, url_texts, question):
    content = ""

    if pdf_text.strip():
        content += "\n\n--- PDF Content ---\n" + pdf_text

    for i, text in enumerate(url_texts):
        if text.strip():
            content += f"\n\n--- URL {i+1} Content ---\n" + text

    prompt = PromptTemplate.from_template(
        "Given the following documents:\n{docs}\n\nAnswer the question: {question}"
    )

    # ✅ Correct LLM usage with env variable fallback
    openai_key = os.getenv("OPENAI_API_KEY", "your-fallback-key-if-applicable")
    llm = ChatOpenAI(temperature=0.3, model_name="gpt-3.5-turbo", openai_api_key=openai_key)

    chain = LLMChain(llm=llm, prompt=prompt)

    response = chain.run({"docs": content, "question": question})
    return response
