from gen_berts import LatinBERT


class LatinBERTDecoder():
    def __init__(self,  latin_bert_model):
        self.encoder = latin_bert_model
        # Assuming you have a method to convert token IDs back to tokens
        self.tokenizer = latin_bert_model.wp_tokenizer

    def decode(self, encoded_input):
        # Decode the encoded input
        decoded_sentences = []
        for sent in encoded_input:
            decoded_sent = []
            for token, _ in sent:
                decoded_sent.append(token)
            decoded_sentences.append(" ".join(decoded_sent))
        return decoded_sentences


if __name__ == "__main__":
    # Initialize an instance of LatinBERT
    encoder = LatinBERT(tokenizerPath="../models/subword_tokenizer_latin/latin.subword.encoder", bertPath="../models/latin_bert")
    # Create an instance of the decoder
    decoder = LatinBERTDecoder(encoder)

    while True:
        # Prompt the user to enter a sentence in Latin
        text = input("Enter a sentence in Latin: ")


        if text.lower() == "exit":
            break
        # Get the encoded input
        bert_sentiments = encoder.get_berts([text])
        # Decode the encoded input
        decoded_sentences = decoder.decode(bert_sentiments)
        # Print the decoded sentences
        for sent in decoded_sentences:
            print(sent)
