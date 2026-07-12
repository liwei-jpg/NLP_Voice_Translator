# Import the 'load_dataset' function from the Hugging Face 'datasets' library.
# We use this because it acts like a smart downloader. Instead of manually downloading ZIP files,
# this function automatically connects to the internet, downloads the data, and structures it into a usable format.
from datasets import load_dataset

# Print a message to the terminal so the user knows the download has started.
print("🔄 Connecting to Hugging Face and downloading the English-Spanish dataset...")

# Here we use the load_dataset function. 
# Argument 1 ("Helsinki-NLP/opus-100"): The name of the massive multilingual database hosted on Hugging Face.
# Argument 2 ("en-es"): We only want the specific subset of data that contains English-to-Spanish translations.
# We save all this downloaded data into a variable called 'dataset'.
dataset = load_dataset("Helsinki-NLP/opus-100", "en-es")

# Print a success message to the terminal.
print("\n✅ Success! Here are 3 sample translations from your database:\n")

# Create a loop that will run exactly 3 times (for i = 0, 1, and 2).
# We use this loop to peek inside the massive dataset and prove that it downloaded correctly.
for i in range(3):
    # Navigate through the dataset structure: go to the "train" split, grab the specific row [i],
    # look inside the "translation" dictionary, and pull out the English ("en") string.
    english_text = dataset["train"][i]["translation"]["en"]
    
    # Do the exact same thing, but pull out the Spanish ("es") string.
    spanish_text = dataset["train"][i]["translation"]["es"]
    
    # Print the current sample number (we add + 1 so it says Sample 1 instead of Sample 0).
    print(f"Sample {i+1}:")
    
    # Print the English text we extracted to the terminal.
    print(f"   English: {english_text}")
    
    # Print the Spanish text we extracted to the terminal, adding a newline (\n) for spacing.
    print(f"   Spanish: {spanish_text}\n")