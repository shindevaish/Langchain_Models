from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0,
)


schema = [
    ResponseSchema(name = 'fact_1', description = 'Fact 1 about the topic'),
    ResponseSchema(name = 'fact_2', description = 'Fact 2 about the topic'),
    ResponseSchema(name = 'fact_3', description = 'Fact 3 about the topic'),
    ResponseSchema(name = 'fact_4', description = 'Fact 4 about the topic'),
    ResponseSchema(name = 'fact_5', description = 'Fact 5 about the topic'),
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template = 'Give five facts about {topic} \n {format_instruction}',
    input_variables = ['topic'],
    partial_variables = {'format_instruction' : parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({'topic': 'Python programming'})

print(result)
