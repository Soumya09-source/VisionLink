from ai_bridge import get_ai_response


prompt = """
You are an AI assistant for a smart assistive vision system.

A user is standing in front of:
- a book
- text saying "Big Text"
- a table nearby

Describe the scene naturally in 1 short sentence.
"""

response = get_ai_response(prompt)

print("\nAI RESPONSE:\n")
print(response)