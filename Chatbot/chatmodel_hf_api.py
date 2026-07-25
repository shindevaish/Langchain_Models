from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

load_dotenv()

client = InferenceClient(
    provider="auto",
    api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)

resp = client.chat_completion(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "What is LLM?"}],
    max_tokens=128,
)

print(resp.choices[0].message.content)