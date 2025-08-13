import requests
import time

prompt_id = "40547a03-c404-48c2-8ff3-61ced0f36b08"
while True:
    r = requests.get(f"http://127.0.0.1:8188/history/{prompt_id}")
    data = r.json()
    # print(data)
    time.sleep(1)
    if data and prompt_id in data:
        print("Job finished!")
        break