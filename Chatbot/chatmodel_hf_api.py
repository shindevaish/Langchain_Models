from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

load_dotenv()

client = InferenceClient(
    provider="nebius",  # e.g., works for Meta Llama models
    token=os.getenv("HF_TOKEN"),
)

messages = [
    {"role": "user", "content": "What is the capital of India?"}
]

response = client.chat_completion(
    model="meta-llama/Llama-3.1-8B-Instruct",  # pick a supported Llama model
    messages=messages,
    max_tokens=256,
)

print(response.choices[0].message.content)