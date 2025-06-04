from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
import gcsfs
import os

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
    # Use gcsfs with passed credentials
    fs = gcsfs.GCSFileSystem(token=creds_dict)

    index_path = f"{bucket_name}/{path}/index.faiss"
    store_path = f"{bucket_name}/{path}/index.pkl"

    # Save GCS files to temp folder
    local_folder = "/tmp/vectorstore"
    os.makedirs(local_folder, exist_ok=True)

    with fs.open(index_path, "rb") as f_in, open(f"{local_folder}/index.faiss", "wb") as f_out:
        f_out.write(f_in.read())

    with fs.open(store_path, "rb") as f_in, open(f"{local_folder}/index.pkl", "wb") as f_out:
        f_out.write(f_in.read())

    # Load vectorstore from local files using LangChain’s FAISS loader
    vectorstore = FAISS.load_local(
        folder_path=local_folder,
        embeddings=OpenAIEmbeddings(),
        allow_dangerous_deserialization=True  # 👈 KEY FIX HERE
    )
    return vectorstore
