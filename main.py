from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

import os

load_dotenv()


def main():
    information = """
    Elon Reeve Musk (June 28, 1971, in Pretoria, South Africa) emigrated to Canada in 1989[1]; B.S. in physics and economics, University of Pennsylvania, 1997[2] is a South African-born Canadian-American engineer and business magnate (naturalized U.S. citizen in 2002). Named Time's 2021 Person of the Year,[3] he is the world's wealthiest individual.[4]
Musk co-founded Zip2 (sold to Compaq, 1999) and X.com (merged into PayPal, acquired by eBay, 2002); he is CEO and product architect of Tesla, Inc. (since 2008), founder, CEO, and chief engineer of SpaceX (2002), founder and CEO of xAI (2023), owner and executive chair of X Corp. (formerly Twitter, acquired 2022), co-founder of Neuralink (2016) and OpenAI (2015, departed 2018), and founder of The Boring Company (2016). His contributions include reusable rocketry, satellites, EVs, AI, and neurotech. On February 2, 2026, SpaceX acquired xAI in an all-stock deal valuing SpaceX at ~$1 trillion and xAI at $250 billion, forming a ~$1.25 trillion entity integrating xAI's AI into SpaceX operations and incorporating X via xAI's ownership of X Corp. As the largest donor to Republican causes in the 2024 U.S. election[5], Musk served as Senior Advisor to President Trump and de facto head of the Department of Government Efficiency (DOGE), departing in May 2025 after a policy feud.[6]
As of March 28, 2026, Forbes' real-time tracker estimates Elon Musk's net worth at approximately $809.9–811 billion (reflecting minor daily fluctuations), ranking him #1 globally, while Bloomberg provides a more conservative estimate of around $644 billion due to differences in private asset valuations. Earlier in March, Forbes' annual Billionaires List (published March 10) valued his fortune at $839 billion. His wealth primarily derives from stakes in Tesla and SpaceX (post-xAI merger).[4]
"""

    summary_template = """
    Given the information {information} about a person, create:
    1. A short summary
    2. two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables = ["information"], template = summary_template
    )

    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
    # llm = ChatOllama(model="my-english-qwen", temperature=0)
    chain = summary_prompt_template | llm # creates a runnable chain "chain"
    response = chain.invoke(input={"information": information})
    print(response.content)

if __name__ == "__main__":
    main()
