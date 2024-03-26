import argparse, sys
import numpy as np
import torch
from torch import nn
from transformers import BertModel, BertTokenizer, BertConfig #,BertPreTrainedModel
import os
import spacy
import re
import json



script_dir = os.path.dirname(os.path.abspath(__file__))
default_bert_path = os.path.join(script_dir, '..', 'models', 'latin_bert')
conf_path = os.path.join(default_bert_path, 'config.json')
# Check if GPU is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LatinBERT():

    def __init__(self, bertPath=None, config_path=None):
        self.wp_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased', do_basic_tokenize=True)
        if bertPath is None:
            raise ValueError("bertPath must be provided.")
        config_path = os.path.join(bertPath, 'config.json')

        if not os.path.exists(config_path):
            raise ValueError(f"Configuration file not found at: {config_path}")  # Assume default location

        print("Config path:", config_path)  # Debugging line

        if not os.path.exists(config_path):
            raise ValueError(f"Configuration file not found at: {config_path}")
        self.model = BertLatin(config_path)
        self.model.to(device)
        bert_vocab_size = len(self.wp_tokenizer.vocab)
        if bert_vocab_size != 105879:
            raise ValueError("Vocabulary size mismatch: Expected 105879, got {}".format(bert_vocab_size))


    def get_batches(self, sentences, max_batch):
        # Preprocess the sentences with the tokenizer
        tokenized_sentences = [self.custom_latin_tokenizer(sent) for sent in sentences]
        # Using the BertTokenizer to encode the sentences
        all_data = self.wp_tokenizer(tokenized_sentences, padding=True, truncation=True, return_tensors='pt')
        # Bert vocabulary size
        input_ids = all_data['input_ids']

        if input_ids.max() >= len(self.wp_tokenizer):
            print("Warning: Input IDs exceed vocabulary size. This shouldn't happen!")
        """ for tensor_row in input_ids:
            tensor_row[tensor_row >= len(self.wp_tokenizer)] = self.wp_tokenizer.unk_token_id"""
        dataset = torch.utils.data.TensorDataset(all_data['input_ids'], all_data['attention_mask'])
        return torch.utils.data.DataLoader(dataset, batch_size=max_batch, shuffle=False)

    def custom_latin_tokenizer(text, latin_english_mapping):
        """
        Performs basic Latin-specific tokenization, handling common abbreviations,
        contractions, and punctuation, while also considering English.

        Args:
            text: The input text string.

        Returns:
            A list of tokenized words and punctuation.
        """

        # Debugging line to check the input text
        print("Input text:", text)

        # Replace common Latin abbreviations (can be extended as needed)
        text = re.sub(r"\bM\.(\s+)", " Marcus ", text)  # Handle "M." for "Marcus"
        text = re.sub(r"\bC\.(\s+)", " Gaius ", text)  # Handle "C." for "Gaius"

        # Split on whitespace and punctuation, with some Latin-specific considerations
        tokens = re.findall(r"[\w\-']+|[.,!?;]", text)

        # Debugging line to check the tokens after basic tokenization
        print("Tokens after basic tokenization:", tokens)

        # Debugging line to check the tokens after handling unknown English tokens
        print("Tokens after handling unknown English tokens:", tokens)

        return tokens

    def get_berts(self, raw_sents):

        # Debugging line to check the raw input sentences
        print("Raw sentences:", raw_sents)
        tokenized_texts = self.wp_tokenizer(raw_sents, padding=True, truncation=True, return_tensors="pt")
        print("Tokenized texts:", tokenized_texts)
        print("Size of my BertTokenizer:", len(self.wp_tokenizer.vocab))

        vocab_size = len(self.wp_tokenizer.vocab)
        #vocab_size = bert.wp_tokenizer.vocab_size
        print("Tokenizer Vocabulary Size:", vocab_size)
        expected_vocab_size = 105879
        if vocab_size == expected_vocab_size:
            print("Tokenizer vocabulary size matches the expected size.")
        else:
            print("Tokenizer vocabulary size does not match the expected size.")
        input_ids = tokenized_texts['input_ids'].to(device)
        print("Input IDs:", input_ids)
        print("Input IDs shape:", input_ids.shape)
        # Convert token IDs to tokens for printing
        tokens_list = []
        for ids in input_ids:
            tokens = [self.wp_tokenizer.convert_ids_to_tokens(id.item()) for id in ids]
            tokens_list.append(tokens)
        print("Corresponding tokens:", tokens_list)

        print("Input ID max items:", input_ids.max().item())
        max_token_id = input_ids.max().item()
        if max_token_id >= vocab_size:
            print("Warning: Token IDs exceed the tokenizer's vocabulary size.")
            print("Maximum token ID:", max_token_id)
        else:
            print("All token IDs are within the tokenizer's vocabulary size.")
        attention_mask = tokenized_texts['attention_mask'].to(device)
        print("Attention masks:", attention_mask)
        print("Attention masks shape:", attention_mask.shape)
        # Verify the size of the embedding layer in the BERT model
        embedding_size = self.model.config.hidden_size
        print("Embedding size:", embedding_size)
        # Check for Unknown Tokens
        if input_ids.max() >= vocab_size:
            raise ValueError("Token IDs exceed vocabulary size")
        print("Input Id after the input_ids,max()", input_ids)
        outputs = self.model.bert(input_ids, attention_mask=attention_mask)
        return outputs

class BertLatin(nn.Module):
    def __init__(self, config_path):
        super(BertLatin, self).__init__()
        with open(config_path, 'r') as f:
            config = json.load(f)

        self.config = BertConfig.from_dict(config)
        self.bert = BertModel(self.config)  # Initialize BertModel with the BertConfig object
        self.bert.eval()

    def get_config(self):
        return self.config

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        return sequence_output


# python3 scripts/gen_berts.py --bertPath models/latin_bert/ --tokenizerPath models/subword_tokenizer_latin/latin.subword.encoder
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--bertPath', help='path to pre-trained BERT', required=False, default=default_bert_path)
    args = vars(parser.parse_args())
    bertPath = args["bertPath"]
    bert = LatinBERT(bertPath=bertPath)


    sents = ["arma virumque cano", "arma gravi numero violentaque bella parabam"]

    bert_sents = bert.get_berts(sents)

    for sent in bert_sents:
        print(sent.shape)
        print(sent)