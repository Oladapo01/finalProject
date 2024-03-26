class EnglishDecoder:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        #  Print the vocabulary of the English tokenizer
        print("English Tokenizer Vocabulary:")
        print(self.tokenizer.get_vocab())

    def decode(self, token_ids):
        decoded_text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        return decoded_text