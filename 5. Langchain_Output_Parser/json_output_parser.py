from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0,
)

parser = JsonOutputParser()

template = PromptTemplate(
    template = 'Give me the name, age and city of fictional person \n {format_instruction}',
    input_variables = [],
    partial_variables = {'format_instruction' : parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({})

print(result)