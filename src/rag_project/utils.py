import yaml
import os
import re

def load_config(config_path: str = "config.yaml") -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def preprocess_text(text: str) -> str:
    try:
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        text = re.sub(r'[\r\n]+', ' ', text)
        text = re.sub(r'\\n', '', text)
        text = text.lower()
        return text

    except Exception as e:
        raise e

