from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

llm = OllamaLLM(model="llama3.2")

messages = [
    ("system", "You are a comedian who tells jokes about {topic}."),
    ("human", "Tell me {joke_count} jokes about {theme}.")
]

prompt_template = ChatPromptTemplate.from_messages(messages)

prompt = prompt_template.invoke({
    "topic" : "Politics",
    "joke_count" : 3,
    "theme" : "Trump"
})

result = llm.invoke(prompt)

print(result)

