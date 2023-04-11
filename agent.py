import fire
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI
from dotenv import load_dotenv

load_dotenv()

class ZeroShotReactDescriptionAgent:
    def __init__(self, llm):
        self.llm = llm

    def run(self, prompt):
        tools = load_tools(["serpapi", "llm-math"], llm=self.llm) # todo create other tools (ex. pinterest search)
        agent = initialize_agent(tools, self.llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
        agent.run(prompt)

def main():
    llm = OpenAI(temperature=0)
    agent = ZeroShotReactDescriptionAgent(llm)
    print("こんにちは！私は知識を持っています。")
    print("私はあなたの質問に答えることができます。")
    print("例えば、")
    print("日本で利用されているSNSの上位５つのユーザ数はそれぞれ何人ですか？そのユーザー数の合計を求めてください。")
    prompt = input("質問を入力してください: ")
    agent.run(prompt)

fire.Fire(main)
