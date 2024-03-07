from datasets import load_dataset
import pandas as pd


def load_dataset_from_huggingface(dataset_name="grosenthal/latin_english_translation", split="train", batch_size=16,
                                  csv_path="latin_english.csv", save_csv=False):
    """
    Loads and processes the dataset from HuggingFace for Latin to English translations
    Args:
        dataset_name (str, optional): The name of the dataset on HuggingFace.
        split (str, optional): The split of the dataset to load (train, validation, test). Currently, it is defaulted to train.
        batch_size (int, optional): The batch size to use for the training.
        csv_path (str, optional): The path to save the dataset as a CSV file.
        save_csv (bool, optional): Whether to save the dataset as a CSV file.
    Returns:
         torch.TensorDataset: The preprocessed dataset in Pytorch format.
    """
    # Load the dataset
    dataset = load_dataset(dataset_name, split=split)

    # Print the dataset info to understand its structure
    print(dataset)

    # Set the format
    dataset.set_format(type='torch', columns=['la', 'en'])

    # Map the dataset to convert to required format
    dataset = dataset.map(lambda examples: {'input_ids': examples['la'], 'labels': examples['en']})

    # If save_csv is True, save the dataset to a CSV file
    if save_csv:
        print("Saving dataset to CSV file:", csv_path)
        # Extract the Latin and English sentences
        latin_sentences = dataset['input_ids']
        english_sentences = dataset['labels']
        # Create a DataFrame
        df = pd.DataFrame({'latin': latin_sentences, 'english': english_sentences})
        # Write DataFrame to CSV
        df.to_csv(csv_path, index=False)
        print("Dataset saved successfully.")

    return dataset


# Call the function to see the dataset structure
load_dataset_from_huggingface(save_csv=True)
