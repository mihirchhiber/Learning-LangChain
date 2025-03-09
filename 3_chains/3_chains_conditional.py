from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain.schema.runnable import RunnableBranch

llm = OllamaLLM(model="llama3.2")

classification_template = ChatPromptTemplate.from_messages([
    ("system", "You are a feedback reviewer."),
    ("human", "Classify the sentiment of this feedback as positive, negative, neutral, or escalate: {feedback}.")
])

positive_template = ChatPromptTemplate.from_messages([
    ("system", "You are a feedback reviewer."),
    ("human", "Generate a thank you note for this positive feedback: {feedback}.")
])

negative_template = ChatPromptTemplate.from_messages([
    ("system", "You are a feedback reviewer."),
    ("human", "Generate an apologetic response for this feedback: {feedback}.")
])

neutral_template = ChatPromptTemplate.from_messages([
    ("system", "You are a feedback reviewer."),
    ("human", "Generate a response for this feedback: {feedback}.")
])

escalate_template = ChatPromptTemplate.from_messages([
    ("system", "You are a feedback reviewer."),
    ("human", "Generate a message to escalate this feedback to a human agent for this feedback: {feedback}.")
])

branches = RunnableBranch(
    (
        lambda x: "positive" in x,
        positive_template | llm
    ),
    (
        lambda x: "negative" in x,
        negative_template | llm
    ),
    (
        lambda x: "neutral" in x,
        neutral_template | llm
    ),
    escalate_template | llm
)

chain = classification_template | llm | branches

result = chain.invoke({"feedback": "The product is terrible. It broke after 3 uses and no one is able to repair it. Waste of money"})

print(result)

