import os
import requests
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

RAW_DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
RAW_DATA_DIR = "data/raw"
RAW_DATA_PATH = os.path.join(RAW_DATA_DIR, "telco_churn_raw.csv")

def download_data(url: str = RAW_DATA_URL, save_path: str = RAW_DATA_PATH):
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    if os.path.exists(save_path):
        logging.info(f"Raw dataset already exists at {save_path}. Skipping download.")
        return save_path
    
    logging.info(f"Downloading raw dataset from {url}...")
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    
    with open(save_path, "wb") as f:
        f.write(response.content)
    logging.info(f"Raw dataset successfully saved to {save_path}")
    return save_path

if __name__ == "__main__":
    download_data()