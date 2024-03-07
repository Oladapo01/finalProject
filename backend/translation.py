import torch
from torch import nn
import pytorch_lightning as pl
from transformers import AutoModel

class LatinTranslationDecoder(pl.LightningModule):
    def __init__(self,encoder, target_vocab_size, learning_rate=1e-3):
        super().__init__()
        self.encoder = encoder
        self.target_vocab_size = target_vocab_size


        # Embedding layer forf the English language tokens
        self.embedding = nn.Embedding(target_vocab_size, embedding_dim=256)

        # Transformer decoder Helsinki-NLP/opus-mt-en-la for English to Latin translation
        self.transformer_decoder =AutoModel.from_pretrained("Helsinki-NLP/opus-mt-en-la")

        # Output layer
        self.output_layer = nn.Linear(self.transformer_decoder.config.hidden_size, target_vocab_size)

        self.save_hyperparameters()

    def forward(self, latin_bert_embeddings, english_token, target_ids=None):
        # Get the embeddings for the English tokens
        english_embeddings = self.embedding(english_token)

        # Concatenate the Latin BERT embeddings and the English embeddings
        combined_embeddings = torch.cat([latin_bert_embeddings, english_embeddings], dim=1)

        # Pass the combined embeddings through the transformer decoder
        transformer_output = self.transformer_decoder(combined_embeddings, target_ids=target_ids)

        # Get the logits for the output layer
        logits = self.output_layer(transformer_output.last_hidden_state)

        return logits