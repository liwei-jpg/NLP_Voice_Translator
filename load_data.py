from datasets import load_dataset

print("🔄 Connecting to Hugging Face and downloading the English-Spanish dataset...")

# Load just the English-Spanish (en-es) subset of the opus-100 database
dataset = load_dataset("Helsinki-NLP/opus-100", "en-es")

# Let's peek at the first 3 translation pairs in the training data
print("\n✅ Success! Here are 3 sample translations from your database:\n")
for i in range(3):
    english_text = dataset["train"][i]["translation"]["en"]
    spanish_text = dataset["train"][i]["translation"]["es"]
    print(f"Sample {i+1}:")
    print(f"   English: {english_text}")
    print(f"   Spanish: {spanish_text}\n")