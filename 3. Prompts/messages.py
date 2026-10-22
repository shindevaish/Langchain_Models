from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = 'meta-llama/Llama-3.1-8B-Instruct',
    task = 'text-generation',
)

model = ChatHuggingFace(llm=llm)

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Tell me a joke about programming."),
]

result = model.invoke(messages)

messages.append(AIMessage(content=result.content))

print(messages)