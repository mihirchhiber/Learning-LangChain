from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain.schema.runnable import RunnableLambda, RunnableParallel

llm = OllamaLLM(model="llama3.2")

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a book critic."),
    ("human", "Provide a brief summary of the book {book_name}.")
])

plot_template = ChatPromptTemplate.from_messages([
    ("system", "You are a book critic."),
    ("human", "Analyse the plot: {summary}. Can you suggest strengths and weaknesses of the plot?")
])

similar_story_template = ChatPromptTemplate.from_messages([
    ("system", "You are a book critic."),
    ("human", "Analyse the plot: {summary}. Can you suggest 3 books with similar storyline. Do share if this story is unique?")
])

rating_template = ChatPromptTemplate.from_messages([
    ("system", "You are a book critic."),
    ("human", "On a scale from 1 to 10, how would you rate {summary}? Only return a number.")
])

plot_chain = plot_template | llm
similar_story_chain = similar_story_template | llm
rating_chain = rating_template | llm


def combine_verdicts(plot_analysis, similar_story_analysis, rating_analysis):
    return f"Plot Analysis:\n{plot_analysis}\n\nSimilar Stories:\n{similar_story_analysis}\n\nRating:\n{rating_analysis}"

chain = (
    prompt_template 
    | llm  
    | RunnableLambda(lambda x: {"summary": x}) 
    | RunnableParallel(branches={
        "plot": plot_chain,  
        "similar_story": similar_story_chain,  
        "rating": rating_chain  
    })
    | RunnableLambda(lambda x: combine_verdicts(x["branches"]["plot"], x["branches"]["similar_story"], x["branches"]["rating"]))  # Combine the results
)

result = chain.invoke({"book_name": "Harry Potter"})

print(result)

