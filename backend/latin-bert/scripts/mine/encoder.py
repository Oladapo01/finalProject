import spacy
import fasttext.util
import os
import re
import la_core_web_lg
from transformers import BertTokenizer, BertModel, Trainer, TrainingArguments, ProgressCallback
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
import sys



sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)
# Download FastText language identification model
if not os.path.exists("lid.176.bin"):
    fasttext.util.download_model("lid.176.bin", if_exists="ignore")


class PrintCallback(ProgressCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        print("Training is starting")

    def on_train_end(self, args, state, control, **kwargs):
        print("Training is finished")

    def on_epoch_begin(self, args, state, control, **kwargs):
        print(f"Starting epoch {state.epoch}")

    def on_log(self, args, state, control, logs, **kwargs):
        print(logs)



class LatinBERTTrainer:
    def __init__(self, latin_text_dir, lemmatizer_dir, subwords_dir, model_name="bert-base-multilingual-cased", output_dir="/data"):
        self.model_name = model_name
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.training_args = TrainingArguments(
            output_dir="/mine/Latin_text",
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=3,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_dir=f"{output_dir}/logs",
            logging_steps=10,
            save_steps=10,
            do_train=True,
            do_eval=True,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            save_total_limit=2,

        )
        self.trainer = None

        # Setting the directory paths
        self.latin_text_dir = latin_text_dir
        self.lemmatizer_dir = lemmatizer_dir
        self.subwords_dir = subwords_dir





        # Load FastText language identification model
        self.language_model = fasttext.load_model("lid.176.bin")

    # Load spaCy models
    nlp_english = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    nlp_latin = la_core_web_lg.load(disable=["ner", "parser"])
    nlp_latin.max_length = 1500000


    def load_text_files(self, directory):
        print("Loading text files from directory")
        files = []
        for root, _, filenames in os.walk(directory):
            for filename in filenames:
                if filename.endswith(".txt"):
                    with open(os.path.join(root, filename), "r", encoding="utf-8") as file:
                        file_content = file.read()
                        files.append(file_content)
        return files

    def load_files(self):
        print("Loading files")
        latin_text = self.load_text_files(self.latin_text_dir)
        lemmatizers = self.load_text_files(self.lemmatizer_dir)
        subwords = self.load_text_files(self.subwords_dir)
        return latin_text, lemmatizers, subwords



    def preprocess_text(self, text):
        # Since we are not using the parser or NER, we can safely process longer texts
        # Check if the text length exceeds the maximum and process in smaller parts if necessary
        print("Preprocessing text beginning")
        print(f"Processing text: {text}")
        processed_texts = []
        # Check if the input is a list
        if isinstance(text, list):
            # Join the list into a single string
            text = " ".join(text)
        if len(text) > self.nlp_latin.max_length:
            chunks = [text[i:i + self.nlp_latin.max_length] for i in range(0, len(text), self.nlp_latin.max_length)]
            for chunk in chunks:
                # Processing each chunk with nlp_latin to create a Doc object
                doc = self.nlp_latin(chunk)
                processed_texts.extend([token.text for token in doc if token.is_alpha])
        else:
            doc = self.nlp_latin(text)
            processed_texts = [token.text for token in doc if token.is_alpha]
        print(f"Processed text: {processed_texts}")
        return " ".join(processed_texts)

    def tokenize_dataset(self, dataset):
        print("Tokenizing dataset")
        tokenized_texts = []
        for example in dataset:  # Assuming dataset is a list of dictionaries
            la_text = example["la"]  # Access the Latin text using the key "la"
            en_text = example["en"]  # Access the English text using the key "en"
            # Tokenize the Latin and English texts
            tokenized_la = self.tokenizer(la_text, padding=True, truncation=True, return_tensors="pt")
            tokenized_en = self.tokenizer(en_text, padding=True, truncation=True, return_tensors="pt")
            # Combine the tokenized texts and append to the list
            tokenized_texts.append({"la": tokenized_la, "en": tokenized_en})
        print(f"Tokenized Texts: {tokenized_texts}")
        return tokenized_texts


    @staticmethod
    def convert_to_dataset(tokenized_texts):
        # Convert the tokenized texts to a Dataset object
        dataset = Dataset.from_dict(tokenized_texts)
        return dataset



    def train(self, dataset, output_dir):
        # Debugging: Inspect dataset structure
        print("Example from the dataset:")
        print(dataset[0])
        # Split the dataset into training and validation sets
        train_dataset, val_dataset = train_test_split(dataset, test_size=0.1, random_state=42)

        # Initialize the Trainer
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=[PrintCallback()],  # Add progress callback
        )

        # Train the model
        trainer.train()
        # Save the model and the tokenizer
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)


    def evaluate(self, test_data):
        test_data = self.preprocess_dataset(test_data)
        test_data = self.tokenize_dataset(test_data)
        eval_results = self.trainer.evaluate(eval_dataset=test_data)
        return eval_results


    def normalize_text(self, text):
        return text.lower()


    def clean_text(self, text):
        # Remove special characters and extra white spaces
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\s+', ' ', text)
        # Remove leading and trailing white spaces
        text = text.strip()
        # Remove non-breaking spaces
        text = text.replace(u'\xa0', u' ')
        return text

    def split_sentences(self, text):
        doc = self.nlp_latin(text) if self.is_latin_text(text) else self.nlp_english(text)
        return [sent.text for sent in doc.sents]

    def tokenize_words(self, texts):
        print("Tokenizing words")
        tokenized_texts = []
        for text in texts:
            print(f"Length of text: {len(text)}")
            # Check if the text is Latin or English
            use_latin_nlp = self.is_latin_text(text)
            # Get the corresponding NLP object
            nlp = self.nlp_latin if use_latin_nlp else self.nlp_english

            tokens = []
            # Process the text in chunks that are small enough
            for start_idx in range(0, len(text), nlp.max_length):
                # Extract a chunk of text that's within the max_length
                chunk = text[start_idx:start_idx + nlp.max_length]
                # Process the chunk to create a Doc object
                doc = nlp(chunk)
                # Extend the tokens list with alpha tokens from this chunk
                tokens.extend([token.text for token in doc if token.is_alpha])

            print(f"Tokens: {tokens}")
            # Append the tokens for this text to the tokenized_texts list
            tokenized_texts.append(tokens)

        return tokenized_texts

    def is_latin_text(self, text):
        prediction = self.language_model.predict(text)
        return "la" in prediction[0]



    def detect_sentence_language(self, text):
        # Split the text into sentences
        sentences = text.split('.')
        sentence_languages = []
        for sentence in sentences:
            # Perform language detection for each sentence
            # Skip empty sentences
            if sentence.strip():
                prediction = self.language_model.predict(sentence)
                # Extract language code
                language = prediction[0][0].split('_')[0]
                sentence_languages.append(language)
        return sentence_languages



    def preprocess_mixed_text(self, mixed_text):
        # Normalize and clean the mixed text
        normalized_text = self.normalize_text(mixed_text)
        cleaned_text = self.clean_text(normalized_text)

        # Detect the languages of individual sentences
        sentence_languages = self.detect_sentence_language(cleaned_text)

        # Get individual sentences
        sentences = cleaned_text.split('.')
        preprocessed_sentences = []
        # Get individual sentences
        sentences = self.split_sentences(mixed_text)
        for sentence, lang in zip(sentences, sentence_languages):
            # Preprocess each sentence based on its detected language
            if lang == 'la':
                preprocessed_sentence = self.preprocess_latin_text(sentence)
            else:
                preprocessed_sentence = self.preprocess_english_text(sentence)
            preprocessed_sentences.append(preprocessed_sentence)
        return preprocessed_sentences

    def preprocess_latin_text(self, text):
        # Process with Spacy Italian model
        doc = self.nlp_latin(text)
        # Extract alphabetic tokens
        tokens = [token.text for token in doc if token.is_alpha]
        # ... (Research more pre-processing steps for Latin text)
        return " ".join(tokens)

    def preprocess_english_text(self, text):
        # Process with Spacy English model
        doc = self.nlp_english(text)
        # Extract alphabetic tokens
        tokens = [token.text for token in doc if token.is_alpha]
        # ... (Research more pre-processing steps for English text
        return " ".join(tokens)


    def prepare_translation_data(self, dataset):
        #Reorganizing the dataset for translation

        translation_data = {
            'la': [],
            'en': []
        }
        for example in dataset:
            translation_data['la'].append(example['la'])
            translation_data['en'].append(example['en'])
        return translation_data



    @staticmethod
    def load_files_from_directory(directory):
        files = []
        for root, _, filenames in os.walk(directory):
            for filename in filenames:
                if filename.endswith(".txt"):
                    with open(os.path.join(root, filename), "r", encoding="utf-8") as file:
                        file_content = file.read()
                        files.append(file_content)
        return files

    def load_and_tokenize_dataset(self):
        # Load the dataset
        dataset = load_dataset("grosenthal/latin_english_translation", split="train")
        # Tokenize the dataset
        tokenized_dataset = dataset.map(
            lambda examples: self.tokenizer(examples['la'], examples['en'], padding="max_length", truncation=True),
            batched=True
        )
        # Split the tokenized dataset into train and validation sets
        self.train_dataset, self.val_dataset = tokenized_dataset.train_test_split(test_size=0.1).values()
        return tokenized_dataset


    def tokenize_function(self, examples):
        return self.tokenizer(examples['en'], examples['la'], padding="max_length", truncation=True)

    def preprocess_dataset(self, dataset):
        if isinstance(dataset, list):
            # Assuming dataset is a list of dictionaries
            if dataset and isinstance(dataset[0], dict) and 'la' in dataset[0] and 'en' in dataset[0]:  # Add extra check
                return self.tokenizer(dataset, ["la", "en"], padding="max_length", truncation=True)  # Tokenize directly
            else:
                print("Unexpected structure in dataset. Expected a list of dictionaries with 'la' and 'en' keys.")
                return None
        else:
            # Handle the case where dataset is not a list of dictionaries
            print("Dataset is not structured as expected. Expected a list of dictionaries.")
            return None

    def save_model(self, model, output_dir):
        model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    def load_model(self, model_path):
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertModel.from_pretrained(model_path)


# Directories
latin_text_dir = "data\\"
print(latin_text_dir)
lemmatizer_dir = os.path.join(latin_text_dir, "lemmatizer")
subwords_dir = os.path.join(latin_text_dir, "subwords")

# Initialize LatinTextProcessor
text_processor = LatinBERTTrainer(latin_text_dir, lemmatizer_dir, subwords_dir)

# Load files
latin_text, lemmatizers, subwords = text_processor.load_files()

# Preprocess files
preprocessed_latin_text = [text_processor.preprocess_text(text) for text in latin_text]

# Tokenize the preprocessed texts
tokenized_local_texts = text_processor.tokenize_words(preprocessed_latin_text)
local_text = LatinBERTTrainer.convert_to_dataset(tokenized_local_texts)
# Assuming tokenized_local_texts is now a Dataset object or can be made into one:
text_processor.train(local_text, "/mine/Latin_text")
tokenized_dataset = text_processor.load_and_tokenize_dataset()

# Load the model trained on local Latin texts
text_processor.load_model("mine/Latin_text")

text_processor.train(tokenized_dataset, "mine/huggingface")
# Preprocess the mixed text
# preprocessed_text = preprocessor.preprocess_mixed_text(cleaned_text)
dataset = load_dataset("grosenthal/latin_english_translation", split="train")
latin_text = dataset["train"]
train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)

