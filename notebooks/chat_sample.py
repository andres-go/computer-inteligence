from openai import OpenAI
import os
from dotenv import load_dotenv

SYSTEM_MESSAGE = "You are a chatbot. You will have a conversation with a user. Be friendly and concise"

if __name__ == "__main__":
    load_dotenv()
    URL = os.environ.get('URL')
    KEY = os.environ.get('KEY')
    MODEL = os.environ.get('MODEL')

    client = OpenAI(
        base_url=URL,
        api_key=KEY,
    )

    print(f"Chatting with {MODEL} model at {URL}\n")

    conversation = [
        {'role': 'system', 'content': SYSTEM_MESSAGE},
    ]

    while True:
        message = input("> ")
        conversation.append({'role': 'user', 'content': message})
        response = client.chat.completions.create(
            model=MODEL,
            messages=conversation
        )
        assistant_reply = response.choices[0].message.content
        print(assistant_reply)
        conversation.append({'role': 'assistant', 'content': assistant_reply})
        