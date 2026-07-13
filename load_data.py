# Import the 'load_dataset' function from the Hugging Face 'datasets' library.
# Why we use this: Instead of manually downloading ZIP files from the internet, unzipping them, 
# and writing complex code to read raw text files, this tool automatically connects to the cloud, 
# downloads the data, caches it on your hard drive, and organizes it perfectly for machine learning.
from datasets import load_dataset

# Print a status message to the terminal screen.
# Why we use this: Downloading massive datasets can take time. This message lets the user 
# (or the lecturer) know that the program hasn't frozen and is actively fetching data over the network.
print("🔄 Connecting to Hugging Face and downloading the English-Spanish dataset...")

# Load just the English-Spanish (en-es) subset of the opus-100 database
# This is the core data-fetching line. We call the function with two specific arguments:
# Argument 1 ("Helsinki-NLP/opus-100"): The exact name of the massive multilingual database hosted on Hugging Face.
# Argument 2 ("en-es"): This tells the system to ignore all other languages and ONLY download the English-to-Spanish translations.
# We save all this structured data into a variable named 'dataset' so we can access it later.
dataset = load_dataset("Helsinki-NLP/opus-100", "en-es")

# Let's peek at the first 3 translation pairs in the training data
# Print a success message to the terminal. 
# The '\n' creates a blank line before and after the text so the terminal output looks clean and easy to read.
print("\n✅ Success! Here are 3 sample translations from your database:\n")

# Start a loop that will repeat exactly 3 times (the variable 'i' will be 0, then 1, then 2). 
# Why we use this: Before we spend hours training an AI model, it is a standard software engineering practice 
# to "peek" at the downloaded data to verify that it is formatted correctly and isn't corrupted.
for i in range(3):
    
    # Navigate through the downloaded dataset's folder structure to get the English text.
    # 1. dataset["train"] goes into the training data section.
    # 2. [i] picks the specific row (row 0, 1, or 2).
    # 3. ["translation"] opens the dictionary column holding the sentence pairs.
    # 4. ["en"] extracts the actual English string.
    # We save this string into the 'english_text' variable.
    english_text = dataset["train"][i]["translation"]["en"]
    
    # Navigate through the exact same structure, but this time extract the string tagged 
    # as "es" (Spanish). We save this matching translation into the 'spanish_text' variable.
    spanish_text = dataset["train"][i]["translation"]["es"]
    
    # Print the current sample number to the screen.
    # We use 'i+1' because Python computers start counting at 0, but humans start counting at 1. 
    # This makes the output say "Sample 1" instead of "Sample 0".
    # The 'f' before the string (an f-string) allows us to inject variables directly inside the {} brackets.
    print(f"Sample {i+1}:")
    
    # Print the English text we extracted earlier. 
    # The empty spaces inside the quotation marks are just to indent the text and make it look visually neat.
    print(f"   English: {english_text}")
    
    # Print the corresponding Spanish text. 
    # The '\n' at the very end adds an extra blank line so the next sample in the loop has proper spacing.
    print(f"   Spanish: {spanish_text}\n")