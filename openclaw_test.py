import requests

TOKEN = "73178da61098c134815289cb7f837a768dfcd2b395a6dbed"

headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json"
}

payload = {
    "model": "openclaw/default",
    "messages": [
        {
            "role": "user",
            "content": "Describe what a smart assistive vision system should do."
        }
    ]
}

response = requests.post(
    "http://127.0.0.1:18789/v1/chat/completions",
    headers=headers,
    json=payload
)

print(response.status_code)
print(response.json())