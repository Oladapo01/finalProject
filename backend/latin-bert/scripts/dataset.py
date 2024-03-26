"""from datasets import load_dataset

def load_dataset_from_huggingface(dataset_name="grosenthal/latin_english_translation", split="train"):
    # Load the dataset
    dataset = load_dataset(dataset_name, split=split)

    # Assuming 'la' is your input column, and 'en' is your target column
    # Adjust as necessary based on your dataset structure
    tokenized_dataset = dataset.map(lambda examples: {'input_ids': examples['la'], 'labels': examples['en']}, batched=True)

    return tokenized_dataset

# Now call the function to load and preprocess the dataset
dataset = load_dataset_from_huggingface()"""




from datasets import load_dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')

def tokenize_function(examples):
    return tokenizer(examples['en'], examples['la'], padding="max_length", truncation=True)

def load_dataset_from_huggingface(dataset_name="grosenthal/latin_english_translation", split="train"):
    # Load the dataset
    dataset = load_dataset(dataset_name, split=split)
    # Print the number of training examples
    dataset.set_format(type='pandas', columns=['en', 'la']).list()
    print("Number of training examples:", len(dataset))

    # Print a few examples of English translations
    print("English translations in training data:")
    for i in range(5):  # Print the first 5 examples
        print("Dataset:", dataset[i]['en'])
    # Tokenize the 'en' column
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Print first 5 examples to verify
    print("Tokenized Dataset:", tokenized_dataset['en'][:5])

    # Assuming 'la' is your input column, and you want to predict 'en' column, adjust as necessary
    tokenized_dataset = tokenized_dataset.map(lambda examples: {'input_ids': examples['la'], 'labels': examples['input_ids']}, batched=True)

    return tokenized_dataset



"""
class LatinTokenizer():
    def __init__(self, tokenizer_path):
"""