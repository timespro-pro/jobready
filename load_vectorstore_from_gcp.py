import os
import tempfile
from google.cloud import storage
from google.oauth2 import service_account
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings


def sanitize_program_name(url: str) -> str:
    """
    Sanitize the program URL to create a folder name.
    """
    return url.strip("/").split("/")[-1].replace("-", "_")


def download_vectorstore_from_gcp(bucket_name: str, prefix: str, gcp_credentials: dict) -> str:
    """
    Downloads all files under a given GCS prefix (folder) to a local temporary directory.
    """
    try:
        credentials = service_account.Credentials.from_service_account_info(gcp_credentials)
        client = storage.Client(credentials=credentials, project=gcp_credentials.get("project_id"))
        bucket = client.bucket(bucket_name)

        local_dir = tempfile.mkdtemp()
        blobs = client.list_blobs(bucket_name, prefix=prefix)

        found_blob = False
        for blob in blobs:
            if blob.name.endswith("/"):  # skip folders
                continue

            found_blob = True
            relative_path = os.path.relpath(blob.name, prefix)
            local_path = os.path.join(local_dir, relative_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            blob.download_to_filename(local_path)

        if not found_blob:
            raise FileNotFoundError("No vectorstore files found at GCP prefix.")

        return local_dir

    except Exception as e:
        print(f"[GCP Load Failed] {e}")
        return None


def load_vectorstore(folder_name: str, openai_api_key: str, gcp_config: dict = None) -> FAISS:
    """
    Loads the FAISS vectorstore from GCP if available, or falls back to local repo folder.
    """
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    # Try GCP first
    local_path = None
    if gcp_config:
        print("[INFO] Attempting to load vectorstore from GCP...")
        local_path = download_vectorstore_from_gcp(
            bucket_name=gcp_config["bucket_name"],
            prefix=gcp_config["prefix"] + "/" + folder_name,
            gcp_credentials=gcp_config["credentials"]
        )

    # Fallback to local GitHub repo path
    if not local_path or not os.path.exists(local_path):
        local_path = os.path.join("vectorstores", folder_name)
        print(f"[INFO] Falling back to local path: {local_path}")
        if not os.path.isdir(local_path):
            raise FileNotFoundError(f"No vectorstore available or folder not found: {local_path}")

    vectorstore = FAISS.load_local(
        folder_path=local_path,
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore
