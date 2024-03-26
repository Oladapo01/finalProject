from datasets import load_dataset

dataset = load_dataset("Landvision-South-East-Ltd/UK-Species-English-latin-names")

print(dataset['train'][0])