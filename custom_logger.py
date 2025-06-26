import uuid
import datetime
import platform
import streamlit as st
from typing import List, Tuple
import gcsfs

class Logger:
    def __init__(self, gcp_bucket: str, gcp_creds: dict, base_path: str = "logs"):
        self.session_id = str(uuid.uuid4())[:8]
        self.device_type = platform.system()
        self.run_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.today = datetime.datetime.now().strftime("%Y-%m-%d")
        self.bucket = gcp_bucket
        self.creds = gcp_creds
        self.base_path = base_path
        self.log_lines: List[str] = []

    def log_metadata(self, timespro_url: str, competitor_url: str):
        self.log_lines.append(f"Session ID: {self.session_id}")
        self.log_lines.append(f"Device Type: {self.device_type}")
        self.log_lines.append(f"Run Datetime: {self.run_datetime}")
        self.log_lines.append(f"Selected TimesPro Program: {timespro_url}")
        self.log_lines.append(f"Entered Competitor URL: {competitor_url}")
        self.log_lines.append("")

    def log_comparison_output(self, comparison_output: str):
        self.log_lines.append("=== COMPARISON OUTPUT ===")
        self.log_lines.append(comparison_output.strip() or "[Empty]")
        self.log_lines.append("")

    def log_chatbot_qa(self, metadata: dict):
        self.log_lines.append("=== SESSION METADATA ===")
        for key, value in metadata.items():
            if key != "qa_pairs":
                self.log_lines.append(f"{key.replace('_', ' ').title()}: {value}")
        self.log_lines.append("")
    
        self.log_lines.append("=== CHATBOT Q&A ===")
        qa_pairs = metadata.get("qa_pairs", [])
        for i, qa in enumerate(qa_pairs, 1):
            if isinstance(qa, (list, tuple)) and len(qa) == 2:
                question, answer = qa
            else:
                question, answer = "Invalid format", str(qa)
            self.log_lines.append(f"Q{i}: {question}")
            self.log_lines.append(f"A{i}: {answer}")
            self.log_lines.append("")

    def write_to_gcs(self):
        log_text = "\n".join(self.log_lines).strip()
        log_path = f"{self.base_path}/{self.today}/{self.session_id}.txt"

        fs = gcsfs.GCSFileSystem(token=self.creds)
        with fs.open(f"{self.bucket}/{log_path}", "w") as f:
            f.write(log_text)

        return f"gs://{self.bucket}/{log_path}"
