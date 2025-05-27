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
    credentials = service_account.Credentials.from_service_account_info(gcp_credentials)
    client = storage.Client(credentials=credentials, project=gcp_credentials.get("project_id"))
    bucket = client.bucket(bucket_name)

    # Create a temporary local directory
    local_dir = tempfile.mkdtemp()

    # List and download all blobs under the given prefix
    blobs = client.list_blobs(bucket_name, prefix=prefix)
    for blob in blobs:
        if blob.name.endswith("/"):  # Skip folders
            continue

        # Determine relative path to save locally
        relative_path = os.path.relpath(blob.name, prefix)
        local_path = os.path.join(local_dir, relative_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        # Download blob
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
    vectorstore = FAISS.load_local(
        folder_path=local_path,
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore


def load_local_vectorstore_from_repo(relative_path: str, openai_api_key: str) -> FAISS:
    """
    Loads a FAISS vectorstore stored in the same GitHub repository as the code.

    Args:
        relative_path (str): Relative path from the current file to the vectorstore folder.
        openai_api_key (str): Your OpenAI API key.

    Returns:
        FAISS: A LangChain FAISS vectorstore object.
    """
    abs_path = os.path.join(os.path.dirname(__file__), relative_path)

    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Vectorstore path does not exist: {abs_path}")

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.load_local(
        folder_path=abs_path,
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore
