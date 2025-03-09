from langchain_ollama.llms import OllamaLLM

llm = OllamaLLM(model="llama3.2")

result = llm.invoke("What is the square root of 49")

print(result)