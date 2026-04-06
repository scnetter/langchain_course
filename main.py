from dotenv import load_dotenv
load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI


@tool
def search(query: str) -> str:
    """
    Tool that searches over the internet
    Args: 
        query: the query to search for
    Returns:
        The search result
    """
    print(f"Searching for: {query}")
    return "Tokyo Weather is sunny"

llm = ChatOpenAI(model="gpt-4o-mini")
tools = [search]
agent = create_agent(model=llm, tools=tools)

def main():
    print("Hello from langchain search agent")
    result = agent.invoke({"messages":HumanMessage(content="What is the weather in Tokyo?")})
    print(result["messages"][-1].content)

if __name__ == "__main__":
    main()
