from langchain_ollama.llms import OllamaLLM
from langchain.agents import create_react_agent, AgentExecutor, tool
import datetime

# llm = OllamaLLM(model="deepseek-r1:1.5b")
llm = OllamaLLM(model="llama3.2")

@tool
def get_system_time(format: str = "%Y-%m-%d %H:%M:%S"):
    """ Returns the current date and time in the specified format """

    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime(format)
    return formatted_time

@tool
def evaluate_expression(expression: str):
    """Evaluates a mathematical expression."""
    try:
        result = eval(expression)
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Error in evaluating expression: {e}"

@tool
def get_system_time(format: str = "%Y-%m-%d %H:%M:%S"):
    """ Returns the current date and time in the specified format """

    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime(format)
    return formatted_time

# query = "What is the time right now?"
query = "Can you calculate my wage per hour if I am earing 15000 per month when I work 20 days a month with 8 hours each day? "

# Create a LANGSMITH_API_KEY in Settings > API Keys
from langsmith import Client
client = Client(api_key="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")
prompt = client.pull_prompt("hwchase17/react", include_model=True)

tools = [get_system_time, evaluate_expression]

agent = create_react_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=30, max_execution_time=150)

output = agent_executor.invoke({
    "input" : query
})

print(output)
