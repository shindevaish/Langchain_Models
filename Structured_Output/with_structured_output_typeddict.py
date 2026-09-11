from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
from typing import TypedDict

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    temperature=0.0,
    max_new_tokens=256,
)

model = ChatHuggingFace(llm=llm)

class Review(TypedDict):
    summary: str
    sentiment: str

parser = JsonOutputParser(pydantic_object=Review)

prompt = """
You are a product review analyzer. Output ONLY valid JSON with this exact schema:
{
  "summary": "<one-sentence summary>",
  "sentiment": "positive" | "negative" 
}

Review text:
The hardware is great, but the software feels bloated. There are too many pre-installed apps that I can't remove. Also, the UI looks outdated compared to other brands. Hoping for a software update to fix this.
"""

raw_response = model.invoke(prompt)
print("Raw content:")
print(raw_response.content)

review: Review = parser.parse(raw_response.content)
print("\nParsed Review:")
print(review)