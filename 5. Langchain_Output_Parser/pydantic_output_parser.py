from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0,
)

class Person(BaseModel):
    name: str = Field(description = 'The name of the person')
    age: int = Field(gt = 18, description = 'Age of the person')
    city: str = Field(description = 'Name of city the person belongs to')

parser = PydanticOutputParser(pydantic_object = Person)

template = PromptTemplate(
    template = 'Generate the name, age and city of a fictional {place} person \n{format_instructions}',
    input_variables = ['place'],
    partial_variables = {'format_instructions' : parser.get_format_instructions()}
)

chain = template | model | parser

final_result = chain.invoke({'place':'sri lankan'})

print(final_result)