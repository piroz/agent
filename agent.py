import transformers
import torch
import fire

class Agent:
    def __init__(self):
        pass

    def start(self):
        user_input = input("何か入力してください: ")
        model = transformers.AutoModelForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
        tokenizer = transformers.AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")


def main():
    agent = Agent()
    result = agent.start()
    print(result)


fire.Fire(main)
