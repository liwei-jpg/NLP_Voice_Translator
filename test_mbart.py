from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

print("⏳ Downloading the mBART model and tokenizer... (This is a 2.4GB file, so it might take a few minutes!)")

# Load the specific model and its associated tokenizer
model_name = "facebook/mbart-large-50-many-to-many-mmt"
tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name)

print("\n✅ Model loaded successfully!")

# 1. Set the source language to English
tokenizer.src_lang = "en_XX"

english_sentence = "Hello, my name is Lee and I am building an artificial intelligence project."
print(f"\n🗣️ English Input: {english_sentence}")
print("⚙️ Translating...")

# 2. Convert the text into numbers (tokens) the model can process
inputs = tokenizer(english_sentence, return_tensors="pt")

# 3. Generate the translation, explicitly forcing the output language to Spanish (es_XX)
generated_tokens = model.generate(
    **inputs,
    forced_bos_token_id=tokenizer.lang_code_to_id["es_XX"]
)

# 4. Decode the output numbers back into human-readable text
spanish_output = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

print(f"🇪🇸 Spanish Output: {spanish_output}")