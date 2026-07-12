# We import the core classes from the 'transformers' library.
# MBartForConditionalGeneration is the actual neural network 'brain' for translation.
# MBart50TokenizerFast is the tool that turns human language into the numbers (tokens) the model understands.
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

# We print a warning because the mBART model is very large (~2.4GB). 
# This shows you understand the resource requirements of modern NLP models.
print("⏳ Downloading the mBART model and tokenizer... (This is a 2.4GB file, so it might take a few minutes!)")

# We define the specific model name hosted on the Hugging Face hub.
# 'many-to-many-mmt' means this model can translate dozens of languages to each other.
model_name = "facebook/mbart-large-50-many-to-many-mmt"

# We initialize the Tokenizer. This loads the 'vocabulary' file the AI uses to look up word-to-number mappings.
tokenizer = MBart50TokenizerFast.from_pretrained(model_name)

# We initialize the Model (the neural network weights). 
# This loads the actual intelligence that calculates probabilities for word sequences.
model = MBartForConditionalGeneration.from_pretrained(model_name)

print("\n✅ Model loaded successfully!")

# 1. Set the source language identifier. 
# mBART uses specific codes (e.g., 'en_XX') so the model knows which language it is reading.
tokenizer.src_lang = "en_XX"

# A test sentence to verify that the base model works before we add our custom LoRA weights.
english_sentence = "Hello, my name is Lee and I am building an artificial intelligence project."
print(f"\n🗣️ English Input: {english_sentence}")
print("⚙️ Translating...")

# 2. Convert the text into numbers (PyTorch Tensors). 
# return_tensors="pt" is essential because the model only performs math using PyTorch formats.
inputs = tokenizer(english_sentence, return_tensors="pt")

# 3. Generate the Spanish translation. 
# We use 'forced_bos_token_id' to force the model to start the sentence by generating the Spanish language code.
# Without this, the model wouldn't know which language to output in!
generated_tokens = model.generate(
    **inputs,
    forced_bos_token_id=tokenizer.lang_code_to_id["es_XX"]
)

# 4. Decode the output tokens (numbers) back into text. 
# 'skip_special_tokens=True' ensures we don't see the technical markers (like </s>) that the AI inserts.
# We pick [0] because the tokenizer returns a batch of results, and we only sent one sentence.
spanish_output = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

print(f"🇪🇸 Spanish Output: {spanish_output}")
