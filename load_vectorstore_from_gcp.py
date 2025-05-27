import os
import tempfile
from google.cloud import storage
from google.oauth2 import service_account
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

def download_vectorstore_from_gcp(bucket_name: str, prefix: str, gcp_credentials: dict) -> str:
    credentials = service_account.Credentials.from_service_account_info(gcp_credentials)
    client = storage.Client(credentials=credentials, project=gcp_credentials.get("project_id"))
    bucket = client.bucket(bucket_name)

    local_dir = tempfile.mkdtemp()
    blobs = client.list_blobs(bucket_name, prefix=prefix)

    found_blob = False
    for blob in blobs:
        if blob.name.endswith("/"):
            continue
        found_blob = True
        relative_path = os.path.relpath(blob.name, prefix)
        local_path = os.path.join(local_dir, relative_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        blob.download_to_filename(local_path)

    if not found_blob:
        raise FileNotFoundError("No vectorstore files found in GCP path.")
    return local_dir

def load_vectorstore(folder_name: str, openai_api_key: str, gcp_config: dict = None) -> FAISS:
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    local_path = None

    if gcp_config:
        try:
            print(f"[INFO] Loading vectorstore from GCP bucket '{gcp_config['bucket_name']}' path '{gcp_config['prefix']}/{folder_name}'")
            local_path = download_vectorstore_from_gcp(
                bucket_name=gcp_config["bucket_name"],
                prefix=f"{gcp_config['prefix']}/{folder_name}",
                gcp_credentials=gcp_config["credentials"]
            )
        except Exception as e:
            print(f"[GCP Load Failed] {e}")

    if not local_path or not os.path.exists(local_path):
        local_path = os.path.join("vectorstores", folder_name)
        if not os.path.isdir(local_path):
            raise FileNotFoundError(f"No vectorstore available or folder not found: {local_path}")

    return FAISS.load_local(local_path, embeddings=embeddings, allow_dangerous_deserialization=True)
