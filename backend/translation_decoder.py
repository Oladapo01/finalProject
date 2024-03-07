import torch
from torch import nn
from transformers import AutoModel
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Create positional encodings once in log space
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arrange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        # Register as buffer so that it will be saved in the model state dict
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :]
        return x


class LatinTranslationDecoder(nn.Module):
    def __init__(self, target_vocab_size, embedding_dim=256, n_head=8, num_decoder_layers=6, learning_rate=1e-3):
        super().__init__()
        self.target_vocab_size = target_vocab_size
        # Embedding layer for the English language tokens
        self.embedding = nn.Embedding(target_vocab_size, embedding_dim)
        self.positional_encoding = PositionalEncoding(embedding_dim)
        # Transformer decoder layer
        self.decoder_layers = nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=n_head)
        self.transformer_decoder = nn.TransformerDecoder(
            self.decoder_layers, num_layers=num_decoder_layers
        )

        # Output layer
        self.output_layer = nn.Linear(embedding_dim, target_vocab_size)


    def forward(self, latin_bert_embeddings, target_ids=None, src_key_padding_mask=None, tgt_key_padding_mask=None):
        memory =latin_bert_embeddings
        # If targets_ids are not provided, then geneerate them
        if target_ids is not None:
            target_embeddings = self.embedding(target_ids)
            target_embeddings *= torch.sqrt(torch.tensor(self.embedding.embedding_dim, device=target_embeddings.device))
            target_embeddings = self.positional_encoding(target_embeddings)
            tgt_seq_len = target_embeddings.size(0)
            attention_mask = torch.triu(torch.ones((tgt_seq_len, tgt_seq_len), device=target_embeddings.device) * float('-inf'), diagonal=1)
        else:
            target_embeddings = memory
            attention_mask = None # No need to mask if not decoding

        transformer_output = self.transformer_decoder(
            tgt=target_embeddings,
            memory=memory,
            tgt_mask=attention_mask,
            memory_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )

        # Get the logits for the output layer
        logits = self.output_layer(transformer_output)

        return logits