import requests


OPENCLAW_TOKEN = "73178da61098c134815289cb7f837a768dfcd2b395a6dbed"

OPENCLAW_URL = "http://127.0.0.1:18789/v1/chat/completions"


def get_ai_response(prompt):
    headers = {
        "Authorization": f"Bearer {OPENCLAW_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "openclaw/default",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    try:
        response = requests.post(
            OPENCLAW_URL,
            headers=headers,
            json=payload,
            timeout=60
        )

        data = response.json()

        return data["choices"][0]["message"]["content"]

    except Exception as e:
        return f"AI Error: {str(e)}"