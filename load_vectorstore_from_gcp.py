import os
import gcsfs
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document

def upload_vectorstore(local_folder_path: str, bucket_name: str, folder_path: str, creds_dict: dict):
    """Upload an existing FAISS vectorstore folder to GCP"""
    fs = gcsfs.GCSFileSystem(token=creds_dict)
    index_faiss = os.path.join(local_folder_path, "index.faiss")
    index_pkl = os.path.join(local_folder_path, "index.pkl")

    if not (os.path.exists(index_faiss) and os.path.exists(index_pkl)):
        raise FileNotFoundError("FAISS index files not found in the given path.")

    fs.put(index_faiss, f"{bucket_name}/{folder_path}/index.faiss", overwrite=True)
    fs.put(index_pkl, f"{bucket_name}/{folder_path}/index.pkl", overwrite=True)
    return True

def create_and_upload_vectorstore_from_pdf(pdf_text: str, bucket_name: str, folder_path: str, creds_dict: dict):
    """Create a FAISS vectorstore from raw PDF text and upload it to GCP"""
    # 1. Split the PDF text into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents([Document(page_content=pdf_text)])

    # 2. Create FAISS vectorstore
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)

    # 3. Save vectorstore to a temporary local folder
    local_folder = "/tmp/vectorstore_upload"
    os.makedirs(local_folder, exist_ok=True)
    vectorstore.save_local(local_folder)

    # 4. Upload to GCP using existing function
    return upload_vectorstore(local_folder, bucket_name, folder_path, creds_dict)
