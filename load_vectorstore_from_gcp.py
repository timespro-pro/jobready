from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from google.oauth2 import service_account
import gcsfs
import pickle

def load_vectorstore_from_gcp(bucket_name: str, path: str, creds_dict: dict):
    """
    Load a FAISS vectorstore from a GCP bucket using a credentials dictionary.

    Args:
        bucket_name (str): GCP bucket name.
        path (str): Path to vectorstore files (without trailing slash).
        creds_dict (dict): GCP service account credentials as dictionary.

    Returns:
        FAISS: Loaded vectorstore object.
    """
    # Use the credentials dictionary directly with gcsfs
    fs = gcsfs.GCSFileSystem(token=creds_dict)

    index_path = f"{bucket_name}/{path}/index.faiss"
    store_path = f"{bucket_name}/{path}/index.pkl"

    with fs.open(index_path, "rb") as f:
        index_data = f.read()

    with fs.open(store_path, "rb") as f:
        store_data = f.read()

    # Save to temp files for FAISS to load
    with open("/tmp/index.faiss", "wb") as f:
        f.write(index_data)
    with open("/tmp/index.pkl", "wb") as f:
        f.write(store_data)

    # Load from local temp files
    with open("/tmp/index.pkl", "rb") as f:
        store = pickle.load(f)
    
    vectorstore = FAISS.load_local(
        folder_path="/tmp",
        index_name="index",
        embeddings=OpenAIEmbeddings(),
        index=store
    )
    return vectorstore
