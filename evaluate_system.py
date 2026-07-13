# Import PyTorch to handle the heavy mathematical tensor calculations for the AI model.
import torch
# Import standard library classes for handling mBART's language translation and tokenization.
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
# Import PEFT (Parameter-Efficient Fine-Tuning) to inject your custom LoRA adapter "skills" into the base model.
from peft import PeftModel
# Import Hugging Face's 'evaluate' library. This provides standardized, academic-grade mathematical formulas to test AI accuracy.
import evaluate
# Import 'jiwer', a specialized library used specifically to calculate Word Error Rate (WER) for speech recognition.
import jiwer

# Print a status message to the terminal to let the user know the metrics are being downloaded/loaded.
print("⚙️ Loading Evaluation Metrics...")
# Load the BLEU metric. BLEU measures exact word-for-word precision (how many exact words the AI got right).
bleu = evaluate.load("bleu")
# Load the ROUGE metric. ROUGE-L measures the Longest Common Subsequence (how well the sentence structure matches).
rouge = evaluate.load("rouge")
# Load the METEOR metric. METEOR is advanced; it gives the AI credit if it uses correct synonyms or stems, evaluating semantic meaning.
meteor = evaluate.load("meteor")

# Print a status update for the translation model loading phase.
print("🧠 Loading mBART Model & LoRA Adapter...")
# Define the exact name of the Facebook foundation model hosted on Hugging Face.
MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"
# Load the tokenizer, which translates human words into mathematical numbers the AI can read.
tokenizer = MBart50TokenizerFast.from_pretrained(MODEL_NAME)
# Load the massive base translation engine into the computer's memory.
base_model = MBartForConditionalGeneration.from_pretrained(MODEL_NAME)
# Inject your specific, fine-tuned conversational LoRA adapter into the base model.
model = PeftModel.from_pretrained(base_model, "./saved_lora_adapter")

# ---------------------------------------------------------
# 1. EVALUATING TRANSLATION (NMT)
# ---------------------------------------------------------
# Print a header to separate the translation results in the terminal.
print("\n--- TRANSLATION EVALUATION (NMT) ---")
# Replace these with 10-20 sentences from your actual validation dataset
# This list holds our input test data. These are the sentences we will ask the AI to translate.
english_sources = [
    "Hello, how are you today?",
    "I need to go to the supermarket.",
    "What is your name?"
]

# The "Ground Truth" (100% perfect human translations)
# This list holds the "Ground Truth". In machine learning, we compare the AI's predictions 
# against these perfect human answers to calculate a mathematical accuracy score.
spanish_references = [
    "Hola, ¿cómo estás hoy?",
    "Necesito ir al supermercado.",
    "¿Cómo te llamas?"
]

# Create an empty list to store the translations that the AI is about to generate.
model_predictions = []

# Explicitly tell the mBART tokenizer that it is reading English text.
tokenizer.src_lang = "en_XX"
# Find the exact mathematical ID number for the Spanish language tag.
tgt_code = tokenizer.lang_code_to_id["es_XX"]

# Print a status update to the terminal.
print("Translating test sentences...")
# Start a loop to go through every single English test sentence one by one.
for text in english_sources:
    # Convert the human text into numerical tensor arrays for the AI to process.
    inputs = tokenizer(text, return_tensors="pt")
    # Use torch.no_grad() to turn off gradient calculation. Since we are just testing (inference) 
    # and not training, this prevents massive memory bloat and speeds up the process.
    with torch.no_grad():
        # Command the AI model to generate the translation, forcing it to output Spanish.
        generated_tokens = model.generate(**inputs, forced_bos_token_id=tgt_code)
    # Convert the AI's output numbers back into human words, hiding any invisible technical tokens.
    translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    # Add the newly generated Spanish translation to our predictions list.
    model_predictions.append(translation)
    # Print the original text and the AI's translation side-by-side so we can visually inspect it.
    print(f"Original: {text} | Translated: {translation}")

# Calculate Scores
# Print a header for the numerical metrics.
print("\n📊 Translation Scores:")
# Use the evaluate library to compare our AI's predictions against the perfect human references using the BLEU formula.
bleu_results = bleu.compute(predictions=model_predictions, references=spanish_references)
# Print the final BLEU score, formatting it to 4 decimal places.
print(f"BLEU Score: {bleu_results['bleu']:.4f} (Higher is better)")

# Compare the AI's predictions against the human references using the ROUGE formula.
rouge_results = rouge.compute(predictions=model_predictions, references=spanish_references)
# Print the ROUGE-L (Longest Common Subsequence) score, formatting it to 4 decimal places.
print(f"ROUGE-L Score: {rouge_results['rougeL']:.4f} (Higher is better)")

# Compare the AI's predictions against the human references using the METEOR formula.
meteor_results = meteor.compute(predictions=model_predictions, references=spanish_references)
# Print the METEOR score, formatting it to 4 decimal places.
print(f"METEOR Score: {meteor_results['meteor']:.4f} (Higher is better)")


# ---------------------------------------------------------
# 2. EVALUATING SPEECH RECOGNITION (ASR)
# ---------------------------------------------------------
# Print a header to separate the speech recognition results in the terminal.
print("\n--- SPEECH RECOGNITION EVALUATION (ASR) ---")
# WER is used to see how many words the AI misheard.
# To do this in your report, you speak a sentence into your app, 
# look at "What the AI Heard", and compare it to what you ACTUALLY said.

# Define the exact words that you physically spoke into the microphone.
actual_spoken_words = "Hola cómo estás hoy"
# Define the exact text that the Whisper model transcribed (the "Ears" output).
what_whisper_heard = "Hola como estas hoy" # (Missing accents, as we saw earlier!)

# Use the jiwer library to calculate the Word Error Rate (WER).
# WER calculates accuracy based on how many substitutions, deletions, and insertions the AI made.
wer_score = jiwer.wer(actual_spoken_words, what_whisper_heard)
# Print the final WER score, formatting it to 4 decimal places. A score of 0.0 means perfect transcription.
print(f"Word Error Rate (WER): {wer_score:.4f} (Lower is better)")