import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from utils.loaders import load_pdf, load_url_content
from utils.llm_chain import get_combined_response
from load_vectorstore_from_gcp import load_vectorstore_from_gcp
import tempfile

# ====== SECRETS & GCP CREDENTIALS ======
openai_key = st.secrets["OPENAI_API_KEY"]
gcp_credentials_dict = dict(st.secrets["GCP_SERVICE_ACCOUNT"])

gcp_config = {
    "bucket_name": "test_bucket_brian",
    "prefix": "vectorstores",
    "credentials": gcp_credentials_dict,
}

# ====== PAGE CONFIG ======
st.set_page_config(page_title="AI Sales Assistant", layout="centered")
st.title("📚 AI Sales Assistant")

# ====== MODEL CHOICE ======
model_choice = st.selectbox(
    "Choose a model",
    ["gpt-3.5-turbo", "gpt-4", "gpt-4-1106-preview", "gpt-4o"],
    index=0,
)

# ====== INPUTS ======
pdf_file = st.file_uploader("Upload a PDF file", type="pdf")

program_options = [
    "-- Select a program --",
    "https://timespro.com/executive-education/iim-calcutta-senior-management-programme",
    "https://timespro.com/executive-education/iim-kashipur-senior-management-programme",
    "https://timespro.com/executive-education/iim-raipur-senior-management-programme",
    "https://timespro.com/executive-education/iim-indore-senior-management-programme",
    "https://timespro.com/executive-education/iim-kozhikode-strategic-management-programme-for-cxos",
    "https://timespro.com/executive-education/iim-calcutta-lead-an-advanced-management-programme",
]

selected_program = st.selectbox("Select TimesPro Program URL", program_options, index=0)
url_1 = selected_program if selected_program != "-- Select a program --" else None
url_2 = st.text_input("Input Competitor Program URL")

# ====== SANITIZE URL ======
def sanitize_url(url: str) -> str:
    return url.strip("/").split("/")[-1].replace("-", "_")

# ====== LOAD VECTORSTORE ======
retriever = None
if url_1:
    folder_name = f"timespro_com_executive_education_{sanitize_url(url_1)}"
    with st.spinner("Loading TimesPro program details ..."):
        try:
            vectorstore_path = f"{gcp_config['prefix']}/{folder_name}"
            vectorstore = load_vectorstore_from_gcp(
                bucket_name=gcp_config["bucket_name"],
                path=vectorstore_path,
                creds_dict=gcp_config["credentials"],
            )
            retriever = vectorstore.as_retriever()
            st.success("Vectorstore loaded successfully ✔️")
        except Exception as e:
            st.error(f"Vectorstore loading failed: {e}")

# ====== SESSION STATE INIT ======
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="question",
        output_key="answer",
        return_messages=True,
        k=7,
    )

if "comparison_output" not in st.session_state:
    st.session_state.comparison_output = ""
if "comparison_injected" not in st.session_state:
    st.session_state.comparison_injected = False

# ====== COMPARISON & CLEAR CACHE BUTTONS ======
col_compare, col_clear = st.columns([3, 1])

with col_compare:
    compare_disabled = selected_program == "-- Select a program --"
    compare_clicked = st.button("Compare", disabled=compare_disabled)

with col_clear:
    clear_clicked = st.button("Clear Cache 🧹", help="Reset chat + comparison")

if clear_clicked:
    st.session_state.clear()
    st.rerun()

# ====== COMPARISON LOGIC ======
if compare_clicked:
    if selected_program == "-- Select a program --":
        st.warning("Please select a valid TimesPro program from the dropdown.")
    elif not (pdf_file or url_1 or url_2):
        st.warning("Please upload a PDF or enter at least one URL.")
    else:
        with st.spinner("Generating sales‑enablement brief ..."):
            # 1️⃣  Extract PDF
            pdf_text = ""
            if pdf_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                    tmp_pdf.write(pdf_file.read())
                    pdf_path = tmp_pdf.name
                pdf_text = load_pdf(pdf_path)

            # 2️⃣  Extract URLs ⇒ dict[ url -> content ]
            url_texts = load_url_content([url_1, url_2]) or {}

            # 3️⃣  NEW call signature with TimesPro + Competitor URLs
            response = get_combined_response(
                pdf_text,
                url_texts,
                timespro_url=url_1,
                competitor_url=url_2,
                model_choice=model_choice,
            )
            st.session_state.comparison_output = response
            st.session_state.comparison_injected = False

# ====== DISPLAY COMPARISON ======
if st.session_state.comparison_output:
    st.success("### 📝 Sales‑Enablement Brief")
    st.write(st.session_state.comparison_output)

# ====== CHATBOT SECTION ======
st.subheader("💬 Ask a follow‑up question")
user_prompt = st.text_input("Enter your question")

if user_prompt:
    if not retriever:
        st.warning("Knowledge base is not available. Please select a program to load its data.")
    else:
        with st.spinner("Answering ..."):
            pdf_text = ""
            if pdf_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                    tmp_pdf.write(pdf_file.read())
                    pdf_path = tmp_pdf.name
                pdf_text = load_pdf(pdf_path)

            url_contexts = load_url_content([url_1, url_2]) or {}
            timespro_context = url_contexts.get(url_1, "")
            competitor_context = url_contexts.get(url_2, "")
            comparison_context = st.session_state.comparison_output or ""

            system_prompt = f"""
You are a **strategic program analyst** helping the sales team pitch a TimesPro program to learners.
Use the context below for all answers.  
• If a **full sales‑enablement brief** is requested, follow **exactly** the structure under *Sales‑enablement brief format* (otherwise ignore the structure and simply answer the question).  
• Be concise, confident, and benefit‑driven; no vague adjectives; cite numbers or facts where possible.  
• If unsure, say you don’t know.  
• Never mention delivery platforms such as Emeritus, upGrad or Coursera.

**Sales‑enablement brief format (only when the user asks for a brief)**
1. **Opening Summary Paragraph** (2–3 lines) – highlight 1‑2 strongest differentiators.
2. **What Makes TimesPro’s Program Better** – 3–4 bold bullet points:  
   **Bold header** (benefit) – supporting line comparing to competitor.
3. **Who This Program Is Built For** – table with 2–3 audience points **and** 2–3 curriculum‑strength points.  
   *(Columns: TimesPro Program | Competitor Program)*
4. **2 Taglines for Learner Interaction**  
   • Aspiration‑focused tagline  
   • Curriculum‑advantage tagline
5. **Price Justification & ROI** *(include only if TimesPro is more expensive)*  
   • 2–3 specific reasons justifying higher price  
   • Finish with a confident value statement

--- EXISTING COUNSELOR INSTRUCTIONS ---
Answer user queries using **only** the context provided. Remain neutral and factual.

--- TIMESPRO DATA ---
{timespro_context}

--- COMPETITOR DATA ---
{competitor_context}

--- PDF DATA ---
{pdf_text}

--- SALES‑ENABLEMENT BRIEF ---
{comparison_context}
"""

            custom_llm = ChatOpenAI(
                model_name=model_choice,
                openai_api_key=openai_key,
                temperature=0,
            )

            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=custom_llm,
                retriever=retriever,
                memory=st.session_state.memory,
                return_source_documents=True,
            )

            if not st.session_state.comparison_injected:
                st.session_state.memory.chat_memory.add_user_message("SYSTEM CONTEXT")
                st.session_state.memory.chat_memory.add_ai_message(system_prompt)
                st.session_state.comparison_injected = True

            result = qa_chain.invoke({"question": user_prompt})
            st.write(f"💬 **Answer:** {result['answer']}")
