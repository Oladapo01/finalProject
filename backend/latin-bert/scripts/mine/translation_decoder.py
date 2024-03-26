import torch
from torch import nn
from transformers import AutoModel
import torch.optim as optim
from torch.nn import KLDivLoss
import math
# Testing
from torch.utils.data import DataLoader, dataset

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Create positional encodings once in log space
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        # Register as buffer so that it will be saved in the model state dict
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :]
        return x


class CustomScheduler:
    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.current_step = 0

    def step(self):
        self.current_step += 1
        lr = (self.d_model ** -0.5) * min(self.current_step ** (-0.5), self.current_step * self.warmup_steps ** (-1.5))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

class LatinTranslationDecoder(nn.Module):
    def __init__(self, target_vocab_size, embedding_dim=1024, n_head=16, num_decoder_layers=12, dropout=0.3, beam_width=5):
        super(LatinTranslationDecoder, self).__init__()
        self.target_vocab_size = target_vocab_size
        self.embedding_dim = embedding_dim
        self.n_head = n_head
        self.beam_width = beam_width
        self.embedding = nn.Embedding(target_vocab_size, embedding_dim)
        self.positional_encoding = PositionalEncoding(embedding_dim)
        self.embedding_projection = nn.Linear(768, embedding_dim)
        # Increase the number of decoder layers and attention heads
        self.decoder_layers = nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=n_head, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layers, num_layers=num_decoder_layers)
        self.output_layer = nn.Linear(embedding_dim, target_vocab_size)


    def calculate_n_head(self, embedding_dim):
        # Calculate the appropriate number of heads based on the embedding dimension
        n_head = 12 # 1 been a minimum value
        while embedding_dim % n_head != 0:
            n_head -= 1
        return n_head


    def length_penalty(self, length, alpha=0.6):
        """Calculates a length penalty based on the given alpha parameter.
            Args:
                sequence_lengths: A tensor of sequence lengths.
                alpha: The strength of the length penalty.
            Returns:
                The length penalty factor.
            """
        return ((5 + length) ** alpha) / ((5 + 1) ** alpha)

    @staticmethod
    def block_ngrams(beam, n, no_repeat_ngram_size=2):
        """No beam search ngram blocking. Ensures the last `n` predictions are not repeated.
        Args:
            beam: The current beam.
            n: The length of ngrams to block.
            no_repeat_ngram_size: The number of ngrams to block.
        Returns:
            The modified beam.
        """
        ngrams = [tuple(beam[i:i + n]) for i in range(len(beam) - n + 1)]

        # Check for any repeated ngrams among the top few beam candidates
        for ngram in ngrams[:no_repeat_ngram_size]:
            if ngrams.count(ngram) > 1:
                return True
        return False

    def generate(self, latin_bert_embeddings, max_length=50, beam_size=5, block_ngram_size=3, temperature=1.0, lengthy_penalty=0.6, src_key_padding_mask=None):
        device = latin_bert_embeddings.device
        sos_token_id = 0
        eos_token_id = 1

        # Adjust beam search width

        # Initialize beams (hypotheses)
        beams = [[sos_token_id]]  # Each beam is a list of token IDs
        scores = [0]  # Scores for each beam


        for _ in range(max_length - 1):
            all_candidates = []

            # Expand each current beam
            for beam, score in zip(beams, scores):
                current_output = self.forward(latin_bert_embeddings,
                                              target_ids=torch.tensor(beam, dtype=torch.long).unsqueeze(0).to(device),
                                              src_key_padding_mask=src_key_padding_mask)
                next_token_logits = current_output[:, -1, :] / temperature
                next_token_probs = torch.softmax(next_token_logits.float(), dim=-1)
                # next_token_idxs = torch.multinomial(next_token_probs, beam_size)
                # topk_scores, topk_indices = torch.topk(next_token_idxs[0], beam_size)
                # all_candidates.extend((beam + [token_id], score + score_prob)
                #                      for token_id, score_prob in next_token_probs[0].item())

                # Generate candidates for each beam
                for token_id, score_prob in next_token_probs[0].items():
                    candidate = beam + [token_id]
                    all_candidates.append((candidate, score + score_prob.log()))

                # Sort candidates and keep top-k
                ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
                beams, scores = list(zip(*ordered[:beam_size]))
                """# Add new hypotheses
               for s, idx in zip(topk_scores.tolist(), topk_indices.tolist()):
                    all_candidates.append((beam + [idx], score + s))"""

            # Keep top k candidates
            ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
            beams, scores = list(zip(*ordered[:beam_size]))
            # beams = [candidate[0] for candidate in ordered[:beam_size]]
            # scores = [candidate[1] for candidate in ordered[:beam_size]]

            # Check for any beams that have ended with <eos>
            for beam in beams:
                if beam[-1] == eos_token_id:
                    return torch.tensor(beam).unsqueeze(0).to(device)

            for i, (beam, score) in enumerate(zip(beams, scores)):
                score -= self.length_penalty(torch.tensor(len(beam), lengthy_penalty))
                scores[i] = score
                if self.block_ngrams(beam, block_ngram_size):
                    # Penalize the score of the beam
                    scores[i] -= 100




        # If none of the beams finished, return the best one
        return torch.tensor(beams[0]).unsqueeze(0).to(device)


    def forward(self, latin_bert_embeddings, target_ids=None, src_key_padding_mask=None, tgt_key_padding_mask=None):
        # Project the input embeddings to the new dimension
        projected_embeddings = self.embedding_projection(latin_bert_embeddings)
        print("Projected Embeddings Shape:", projected_embeddings.shape)
        print("Input Embeddings Shape up:", latin_bert_embeddings.shape)
        # Verify input and output dimensions of embedding layers
        print("Input Embedding Dimension:", self.embedding.embedding_dim)
        print("Output Embedding Dimension:", self.embedding_dim)

        # Verify multi-head attention mechanism
        print("Number of Attention Heads:", self.n_head)
        print("Hidden Dimension:", self.embedding_dim)

        # Verify output layer dimensions
        print("Output Layer Dimension:", self.target_vocab_size)

        # Compare input and output dimensions
        print("Input Embeddings Shape:", latin_bert_embeddings.shape)
        print("Target IDs Shape:", target_ids.shape)

        if latin_bert_embeddings.dim() == 2:
            latin_bert_embeddings = latin_bert_embeddings.unsqueeze(0)

        # Apply the positional encoding to the target embeddings if provided
        if target_ids is not None:
            target_embeddings = self.embedding(target_ids)
            target_embeddings = target_embeddings * math.sqrt(self.embedding_dim)
            target_embeddings = self.positional_encoding(target_embeddings)
            tgt_mask = self.generate_square_subsequent_mask(target_embeddings.size(0))
        else:
            tgt_mask = None

        # Pass the embeddings through the decoder layers
        memory = projected_embeddings # .clone()
        print("Memory Shape Before Reshaping:", memory.shape)  ## Create a copy to avoid modifying latin_bert_embeddings


        print("Memory shape after reshaping:", memory.shape)
        print("Memory:", memory)
        print("What goes into the transformer decoder:", target_embeddings if target_ids is not None else None)
        print("Target Embeddings Shape (Before Decoder):", target_embeddings.shape)
        print("Target Mask Shape:", tgt_mask.shape)
        print("Memory Key Padding Mask Shape:", src_key_padding_mask)
        print("Memory Shape Before Decoder:", memory.shape)
        # Print shapes of tensors involved in attention mechanism
        print("Projected Embeddings Shape:", projected_embeddings.shape)
        print("Input Embeddings Shape up:", latin_bert_embeddings.shape)
        print("Input Embedding Dimension:", self.embedding.embedding_dim)
        print("Output Embedding Dimension:", self.embedding_dim)
        print("Number of Attention Heads:", self.n_head)
        print("Hidden Dimension:", self.embedding_dim)
        print("Output Layer Dimension:", self.target_vocab_size)
        print("Input Embeddings Shape:", latin_bert_embeddings.shape)
        print("Target IDs Shape:", target_ids.shape)

        transformer_output = self.transformer_decoder(
            tgt=target_embeddings,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        print("Transformer Output Shape:", transformer_output.shape)

        # Get the logits for the output layer
        logits = self.output_layer(transformer_output)
        return logits


    @staticmethod
    def generate_square_subsequent_mask(sz):
        """Generates a mask to prevent attention to future positions."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask



# Define some dummy Latin BERT embeddings for testing
"""latin_bert_embeddings = torch.randn(12, 1, 768)  # Adjust sequence_length according to your data
print("Input Embeddings Shape:", latin_bert_embeddings.shape)
target_ids = torch.randint(0, 10000, (12, 1))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latin_bert_embeddings = latin_bert_embeddings.to(device)
target_ids = target_ids.to(device)
# Instantiate the LatinTranslationDecoder class with appropriate parameters
decoder = LatinTranslationDecoder(target_vocab_size=10000, embedding_dim=1024, n_head=16, num_decoder_layers=12, dropout=0.3)
decoder.to(device)
# Pass the Latin BERT embeddings through the decoder
output_logits = decoder(latin_bert_embeddings, target_ids=target_ids)

# Print the shape of the output logits
print("Output Logits Shape:", output_logits.shape)


# Learning Rate Scheduler
optimizer = optim.Adam(decoder.parameters(), lr=0)  # Initial learning rate
scheduler = CustomScheduler(optimizer, d_model=decoder.embedding_dim)



# Define num_epochs
num_epochs = 10
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
for epoch in range(num_epochs):
    for inputs, targets in dataloader:
        # ... other parts ...
        inputs, targets = inputs.to(device), targets.to(device)
        output_logits = decoder(inputs, target_ids=targets)

        # Label Smoothing with KLDivLoss:
        loss_func = KLDivLoss(reduction='batchmean')  # Adjust reduction if needed
        loss = loss_func(output_logits, targets)

        loss.backward()
        optimizer.step()
        scheduler.step()"""
