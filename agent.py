import os
import transformers
import torch
import fire
from langchain.llms.openai import OpenAI

class Agent:
    def __init__(self):
        self.llm = OpenAI(temperature=0, top_p=0.9, frequency_penalty=0.0, presence_penalty=0.0)
        pass

    def start(self):
        user_input = input("何か入力してください: ")
        return self.llm.generate([user_input])


def main():
    if os.environ["OPENAI_API_KEY"] is None:
        raise KeyError("OPENAI_API_KEY is not set")
    
    agent = Agent()
    result = agent.start()
    print(result)


fire.Fire(main)
