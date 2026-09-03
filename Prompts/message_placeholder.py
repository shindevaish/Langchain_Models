from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage

chat_template = ChatPromptTemplate([
    ('system', "You are a helpful customer support agent."),
    MessagesPlaceholder(variable_name="chat_history"),
    ('human', '{query}')
])

chat_history = []
with open("Prompts/chat_history.txt", "r") as f:
    chat_history.extend(f.readlines())


prompt = chat_template.invoke({'chat_history': chat_history, 'query':  "Where is my refund?"})

print(prompt)