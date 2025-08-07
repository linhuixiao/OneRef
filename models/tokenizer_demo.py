
""""
    This demo is to demonstrate the usage of sentencepiece tokenizer.
"""

def get_sentencepiece_model_for_beit3(sentencepiece_model):
    from transformers import XLMRobertaTokenizer
    return XLMRobertaTokenizer(sentencepiece_model)

class beit_demo():
    def __init__(self):
        self.tokenizer = get_sentencepiece_model_for_beit3(sentencepiece_model='/hdd/lhxiao/beit3/checkpoint/beit3.spm')  # 根据传入的 text tokenizer信息 初始化tokenizer

    def tokenize(self, text):
        tokens = self.tokenizer.tokenize(text)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return token_ids


if __name__ == "__main__":
    demo = beit_demo()
    token_ids = demo.tokenize('a giraffe is looking at a herd of horned animals.')
    print("token_ids: ", token_ids)


