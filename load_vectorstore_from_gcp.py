import os
import tempfile
from google.cloud import storage
from google.oauth2 import service_account
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings


def download_vectorstore_from_gcp(bucket_name: str, prefix: str, gcp_credentials: dict) -> str:
    """
    Downloads all files under a given GCS prefix (folder) to a local temporary directory.

    Args:
        bucket_name (str): Name of the GCS bucket.
        prefix (str): GCS path prefix to the vectorstore folder.
        gcp_credentials (dict): GCP service account credentials from Streamlit secrets.

    Returns:
        str: Path to the local directory containing the downloaded vectorstore files.
    """
    # Ensure GCP credentials are structured correctly
    credentials = service_account.Credentials.from_service_account_info(gcp_credentials)
    client = storage.Client(credentials=credentials, project=gcp_credentials.get("project_id"))
    bucket = client.bucket(bucket_name)

    # Temporary local directory
    local_dir = tempfile.mkdtemp()

    blobs = client.list_blobs(bucket_name, prefix=prefix)
    for blob in blobs:
        if blob.name.endswith("/"):  # skip folders
            continue

        relative_path = os.path.relpath(blob.name, prefix)
        local_path = os.path.join(local_dir, relative_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        blob.download_to_filename(local_path)

    return local_dir


def load_vectorstore(local_path: str, openai_api_key: str) -> FAISS:
    """
    Loads the FAISS vectorstore from the local directory.

    Args:
        local_path (str): Path to the local directory containing FAISS files.
        openai_api_key (str): Your OpenAI API key.

    Returns:
        FAISS: A LangChain FAISS vectorstore object.
    """
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.load_local(folder_path=local_path, embeddings=embeddings, allow_dangerous_deserialization=True)
    return vectorstore
