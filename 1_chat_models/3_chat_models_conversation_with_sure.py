from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_ollama.llms import OllamaLLM

llm = OllamaLLM(model="llama3.2")

chat_history = []

chat_history.append(SystemMessage("You are a hindu motivator who preaches about bhagvad gita"))

while True:

    query = input("You: ")

    if query.lower() == "exit":
        break

    chat_history.append(HumanMessage(query))

    result = llm.invoke(chat_history)
    
    print("AI: ", result)
