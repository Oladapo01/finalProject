import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup, AutoTokenizer, BertModel
from dataset import load_dataset_from_huggingface
from datasets import load_dataset, ClassLabel
from gen_berts import LatinBERT
from translation_decoder import LatinTranslationDecoder
import logging
import nltk
from nltk.translate.bleu_score import sentence_bleu
from decoder import LatinBERTDecoder
from english_decoder import EnglishDecoder



# Comfigure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomTranslationDataset(Dataset):
    """Custom PyTorch dataset for the Latin to English translation dataset."""
    def __init__(self, dataset):
        """
        Initializes the dataset.
        Args:
             dataset (Dataset): The dataset containing the Latin and English sentences.
        """
        self.dataset = dataset

    def __len__(self):
        """
        Get the total number of samples in the dataset.
        Returns:
             int: The number of samples in the dataset.
        """
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        Args:
            idx (int): The index of the sample to retrieve.
        Returns:
             tuple: Containing the Latin text and the English text.
        """
        latin_text = self.dataset['la'][idx]
        english_text = self.dataset['en'][idx]
        return latin_text, english_text


class LatinTranslator:
    def __init__(self, dataset_name="grosenthal/latin_english_translation", split="True",  latin_bert_model="LatinBert", trained_model_path=None, validation_split=0.1):
        self.dataset_name = dataset_name
        self.split = split
        self.latin_bert_model = latin_bert_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.latin_bert_model = LatinBERT(tokenizerPath="../models/subword_tokenizer_latin/latin.subword.encoder",
                                          bertPath="../models/latin_bert")
        self.validation_split = validation_split
        self.latin_tokenizer = self.latin_bert_model.wp_tokenizer  # Latin tokenizer from LatinBERT
        self.english_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.english_decoder = EnglishDecoder(self.english_tokenizer) # Use Latin tokenizer for Latin decoder
        self.decoder_model = None
        self.dataset = None
        self.initialize_models()

        # Load the dataset and assign it to self.dataset
        # self.dataset = load_dataset_from_huggingface(dataset_name=self.dataset_name, split="train")

        if trained_model_path is not None:
            # self.load_model(trained_model_path)
            self.load_dataset()
            # self.initialize_models()
        else:
            # self.load_dataset()
            # self.initialize_models()
            self.dataset = load_dataset_from_huggingface(dataset_name=self.dataset_name, split="train")
            self.initialize_models()

    def load_dataset(self):
        # Load the dataset from HuggingFace
        dataset = load_dataset(self.dataset_name, split="train")
        # Set the format
        dataset.set_format(type='torch', columns=['la', 'en'])
        self.dataset = dataset
        total_size = len(self.dataset)
        train_size = int(total_size * (0.8 - self.validation_split))
        val_size = total_size - train_size


        # Apply tokenization
        self.dataset = self.dataset.map(self.tokenize_function)


        # Convert the dataset to a custom PyTorch dataset
        # self.dataset = CustomTranslationDataset(self.train_data)
        self.dataset, self.val_data = torch.utils.data.random_split(self.dataset, [train_size, val_size])


    def tokenize_latin(self, text):
        return self.latin_bert_model.wp_tokenizer.tokenize(text, return_tensors="pt", padding=True, truncation=True)

    def tokenize_english(self, text):
        self.english_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        return self.english_tokenizer(text, return_tensors="pt", padding=True, truncation=True)



    def tokenize_function(self, examples):
        # Tokenize Latin text
        print(type(self.latin_bert_model))
        print("Latin Tokens:", examples['la'])
        latin_tokens = self.latin_bert_model.wp_tokenizer.encoder.encode(examples['la'])
        print("Latin Tokens:", latin_tokens)
        # Create an attention mask for the Latin text
        latin_attention_mask = [1] * len(latin_tokens)
        # Tokenize English text
        english_tokens = self.english_tokenizer(
            examples['en'], return_tensors="pt", padding=True, truncation=True
        )
        return {
            'latin_input_ids': latin_tokens,
            'latin_attention_mask': latin_attention_mask,
            'english_input_ids': english_tokens['input_ids'],
            'english_attention_mask': english_tokens['attention_mask'],
            'english_labels': english_tokens['input_ids']
        }

    def initialize_models(self):
        # Load the pre-trained Latin BERT model and tokenizer
        self.latin_bert_model = LatinBERT(
            tokenizerPath="../models/subword_tokenizer_latin/latin.subword.encoder",
            bertPath="../models/latin_bert"
        )

        # Load the English tokenizer for the decoder
        self.english_tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")

        # Assuming that the decoder model requires the size of the English vocabulary,
        # and assuming that 'bert-large-uncased' is used for English tokenization,
        # we will initialize the LatinTranslationDecoder with the English vocabulary size.
        self.decoder_model = LatinTranslationDecoder(
            target_vocab_size=len(self.english_tokenizer)
        )

        # Move models to the appropriate device (GPU or CPU)
        self.latin_bert_model.model.to(self.device)
        self.decoder_model.to(self.device)



    def train(self, num_epochs=10, batch_size=16, learning_rate=1e-4, num_warmup_steps=100, validation_split=0.1, patience=3):
        # Create DataLoaders with random split
        train_data, val_data = torch.utils.data.random_split(self.dataset,
                                                             [int(len(self.dataset)*(1-validation_split)),
                                                              int(len(self.dataset)*validation_split)])
        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_data, batch_size=batch_size)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.decoder_model = LatinTranslationDecoder(target_vocab_size=self.dataset['en'].feature.num_classes)

        best_val_loss = float('inf')
        epoch_no_improvement = 0

        #Initialize models
        self.initialize_models()
        self.latin_bert_model.model.bert.to(self.device)
        self.decoder_model.to(self.device)


        # Optimizer and loss function
        optimizer = torch.optim.Adam([
            {"params": self.latin_bert_model.parameters(), 'lr': learning_rate / 2},
            {"params": self.decoder_model.parameters()}
        ])


        #  Scheduler
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=len(train_dataloader) * num_epochs)


        # Loss function
        loss_function = nn.CrossEntropyLoss(ignore_index=0)
        # ----------------------------------- Training loop -----------------------------------
        for epoch in range(num_epochs):
            for i, batch in enumerate(train_dataloader):
                latin_input_ids = batch['latin_input_ids'].to(self.device)
                latin_attention_mask = batch['latin_attention_mask'].to(self.device)
                english_input_ids = batch['english_input_ids'].to(self.device)
                english_attention_mask = batch['english_attention_mask'].to(self.device)
                english_labels = batch['english_labels'].to(self.device)




                # Forward pass
                latin_bert_output = self.latin_bert_model(input_ids=latin_input_ids, attention_mask=latin_attention_mask)
                english_logits = self.decoder_model(latin_bert_output['last_hidden_state'], target_ids=english_input_ids)

                # Calculate the loss and perform backpropagation
                loss = loss_function(english_logits.view(-1, self.decoder_model.target_vocab_size), english_labels.view(-1))

                # Backward pass and Optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()


                # Print the loss and validation loss every 100 batches
                if i % 100 == 0:
                    val_loss = self.evaluate(val_dataloader)
                    logger.info(f'Epoch {epoch+1}/{num_epochs}, Batch {i+1}/{len(train_dataloader)}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}')
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        epochs_no_improvement = 0

                        # -------------------------------------- Save the model --------------------------------------
                        torch.save({
                            'latin_bert_state_dict': self.latin_bert_model.state_dict(),
                            'decoder_state_dict': self.decoder_model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'best_val_loss': best_val_loss
                        }, "latin_bert_model.pth")
                    else:
                        epochs_no_improvement += 1
                        if epochs_no_improvement >= patience:
                            logger.info(f'Early stopping at epoch {epoch + 1} due to no improvement in validation loss')
                            return



    def evaluate(self, dataloader):
        self.latin_bert_model.eval()
        self.decoder_model.eval()
        total_loss = 0.0
        num_batches = 0
        loss_function = nn.CrossEntropyLoss(ignore_index=0)
        all_predictions = []
        all_references = []
        with torch.no_grad():
            for batch in dataloader:
                latin_input_ids = batch['latin_input_ids'].to(self.device)
                latin_attention_mask = batch['latin_attention_mask'].to(self.device)
                english_input_ids = batch['english_input_ids'].to(self.device)
                english_attention_mask = batch['english_attention_mask'].to(self.device)
                english_labels = batch['english_labels'].to(self.device)

                latin_bert_output = self.latin_bert_model(input_ids=latin_input_ids, attention_mask=latin_attention_mask)
                english_logits = self.decoder_model(latin_bert_output['last_hidden_state'], target_ids=english_input_ids)
                # Decode the translations
                translations = [self.latin_bert_model.tokenizer.decode(ids, skip_special_tokens=True) for ids in torch.argmax(english_logits, dim=-1)]
                # Collect the translations and references for the BLEU score
                all_predictions.extend(translations)
                # Assuming english_labels contains the actual text references
                all_references.extend(batch['english_labels'])
                loss = loss_function(english_logits.view(-1, self.decoder_model.target_vocab_size), english_labels.view(-1))
                # Mask out padding token before calculating the loss
                active_loss = english_attention_mask.view(-1) != 0 # 0 is the padding token
                loss = loss.masked_select(active_loss).mean()
                total_loss += loss.item()
                num_batches += 1
        # Calculate the BLEU score
        bleu_score = sentence_bleu(all_references, all_predictions)
        logger.info(f'BLEU score: {bleu_score:.4f}')
        return total_loss / num_batches, bleu_score


    def load_model(self, model_path):
        state_dict = torch.load(model_path, map_location=self.device)
        self.latin_bert_model.load_state_dict(state_dict['latin_bert_state_dict'])
        self.decoder_model.load_state_dict(state_dict['decoder_state_dict'])


    def translate(self, latin_sentence):
        print("Latin BERT Model:", self.latin_bert_model.model.bert)

        # Tokenize the input sentence and prepare input tensors
        latin_inputs_ids = self.latin_bert_model.wp_tokenizer.encoder.encode(latin_sentence)
        print("Latin Input IDs above:", latin_inputs_ids)
        # Print the expected translation

        # Convert input ids to a tensor and add batch dimension
        latin_inputs_ids = torch.tensor([latin_inputs_ids]).to(self.device)
        print("Latin Input IDs below:", latin_inputs_ids)

        # Create attention mask (with the same device as input ids)
        latin_attention_mask = torch.ones_like(latin_inputs_ids).to(self.device)
        print("Latin Attention Mask:", latin_attention_mask)

        # Run through BERT model
        with torch.no_grad():  # Don't calculate gradients
            self.latin_bert_model.model.bert.eval()  # Set the model to evaluation mode
            latin_bert_output = self.latin_bert_model.model.bert(input_ids=latin_inputs_ids,
                                                                 attention_mask=latin_attention_mask)

        print("Latin BERT Output:", latin_bert_output)

        # Get the last hidden states from BERT output
        latin_bert_embeddings = latin_bert_output['last_hidden_state']

        # Reshape 'latin_bert_embeddings' to match source text length
        batch_size, source_seq_len, embedding_dim = latin_bert_embeddings.shape
        latin_bert_embeddings = latin_bert_embeddings.transpose(1, 0)

        # Ensure embeddings are floating point numbers for the decoder
        if latin_bert_embeddings.dtype != torch.float:
            latin_bert_embeddings = latin_bert_embeddings.float()

        print("This is the data type:", latin_bert_embeddings.dtype)
        print("This is the data shape:", latin_bert_embeddings.shape)
        print("This is the data:", latin_bert_embeddings)
        # Prepare the src_key_padding_mask
        src_key_padding_mask = latin_attention_mask == 0

        # Generate translation from the decoder model
        print("Generating translation...")
        print("Latin BERT Embeddings:", latin_bert_embeddings.shape)

        # Generate translation from the decoder model
        output_ids = self.decoder_model.generate(latin_bert_embeddings, src_key_padding_mask=None)

        print("Decoder output shape:", output_ids.shape)


        # Decode the output using your decoder
        english_translation = self.english_decoder.decode(output_ids[0])

        return english_translation


