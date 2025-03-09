from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama.llms import OllamaLLM

llm = OllamaLLM(model="llama3.2")

messages = [
    SystemMessage("You are an expert in social media content strategy"),
    HumanMessage("Give a short tip to create engagin posts on Instagram")
]

result = llm.invoke(messages)

print(result)

