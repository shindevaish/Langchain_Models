from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0,
)

# template 1
template1 = PromptTemplate(
    template = 'Write a detailed report on {topic}',
    input_variables = ['topic']
)
# template 2
template2 = PromptTemplate(
    template = 'Write a 5 line summary on the following text. /n {text}',
    input_variables = ['text']
)

prompt1 = template1.invoke({'topic': 'Black Hole'})

result = model.invoke(prompt1)

prompt2 = template2.invoke({'text': result.content})

result1 = model.invoke(prompt2)

print(result1.content)