from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    temperature=0.0,
    max_new_tokens=256,
)

model = ChatHuggingFace(llm=llm)

class Review(TypedDict):
    key_themes: Annotated[list[str], "Write down all the key themes discussed in the review"]
    summary: Annotated[str, "A brief summary of the review"]
    sentiment: Annotated[str, "The overall sentiment of the review either positive or negative"]
    pros: Annotated[Optional[list[str]], "Write down all the pros mentioned in the review"]
    cons: Annotated[Optional[list[str]], "Write down all the cons mentioned in the review"]

parser = JsonOutputParser(pydantic_object=Review)

prompt = """
You are an expert product review analyzer.

Analyze the review below and return ONLY a valid JSON object.
Do not include Markdown, code fences, comments, explanations, or any text before or after the JSON.

Return exactly this structure:

{
  "key_themes": [
    "theme 1",
    "theme 2"
  ],
  "summary": "A concise one- or two-sentence summary of the review.",
  "sentiment": "positive, negative, or mixed",
  "pros": [
    "positive point 1",
    "positive point 2"
  ],
  "cons": [
    "negative point 1",
    "negative point 2"
  ]
}

Rules:
- "key_themes" must be a list of the major topics discussed in the review.
- "summary" must be a concise summary of the complete review.
- "sentiment" must be exactly one of: "positive", "negative", or "mixed".
- Use "mixed" when the review contains both meaningful positive and negative opinions.
- "pros" must contain every positive point explicitly mentioned in the review.
- "cons" must contain every negative point explicitly mentioned in the review.
- Every item in "key_themes", "pros", and "cons" must be a string.
- If no pros are mentioned, return "pros": [].
- If no cons are mentioned, return "cons": [].
- Do not create facts that are not stated in the review.
- Do not use any JSON keys other than:
  "key_themes", "summary", "sentiment", "pros", and "cons".

Review text:
I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:
- Insanely powerful processor for gaming and productivity
- Stunning 200MP camera with incredible zoom capabilities
- Long battery life with fast charging
- S-Pen support is unique and useful

Review by Nitish Singh
"""

raw_response = model.invoke(prompt)
print(parser.parse(raw_response.content))