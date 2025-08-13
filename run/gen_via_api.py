import json
import uuid
import time
import requests
import hydra
from omegaconf import DictConfig

URL = "http://127.0.0.1:8188"

def preprocess_json(config: DictConfig):
    with open(config.json_file_path, "r", encoding="utf-8") as f:
        prompt_data = json.load(f)
    if "62" in prompt_data and "inputs" in prompt_data["62"] and "image" in prompt_data["62"]["inputs"]:
        prompt_data["62"]["inputs"]["image"] = config.people
    if "6" in prompt_data and "inputs" in prompt_data["6"] and "text" in prompt_data["6"]["inputs"]:
        prompt_data["6"]["inputs"]["text"] = config.positive
    if "7" in prompt_data and "inputs" in prompt_data["7"] and "text" in prompt_data["7"]["inputs"]:
        prompt_data["7"]["inputs"]["text"] = config.negative
    return prompt_data

def submit_job(url, prompt_data):
    client_id = str(uuid.uuid4())
    prompt_url = f"{url}/prompt"
    data = {
        "client_id": client_id,
        "prompt": prompt_data,
    }
    response = requests.post(prompt_url, json=data)
    assert response.status_code == 200, f"Failed to submit job: {response.text}"
    resp_json = response.json()
    return client_id, resp_json["prompt_id"]

def check_job_status(prompt_id, url):
    while True:
        r = requests.get(f"{url}/history/{prompt_id}")
        data = r.json()
        if data and prompt_id in data:
            print("Job finished!")
            return True
        time.sleep(1)

@hydra.main(config_path="config", config_name="default")
def main(config: DictConfig):
    prompt_data = preprocess_json(config)
    print(f"Preprocessed JSON: {prompt_data}")
    client_id, prompt_id = submit_job(URL, prompt_data)
    print(f"Job submitted with client ID: {client_id}, prompt ID: {prompt_id}")
    if check_job_status(prompt_id, URL):
        print("Processing completed successfully.")

if __name__ == "__main__":
    main()
