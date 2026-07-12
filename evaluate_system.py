import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from peft import PeftModel
import evaluate
import jiwer

print("⚙️ Loading Evaluation Metrics...")
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
meteor = evaluate.load("meteor")

print("🧠 Loading mBART Model & LoRA Adapter...")
MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"
tokenizer = MBart50TokenizerFast.from_pretrained(MODEL_NAME)
base_model = MBartForConditionalGeneration.from_pretrained(MODEL_NAME)
model = PeftModel.from_pretrained(base_model, "./saved_lora_adapter")

# ---------------------------------------------------------
# 1. EVALUATING TRANSLATION (NMT)
# ---------------------------------------------------------
print("\n--- TRANSLATION EVALUATION (NMT) ---")
# Replace these with 10-20 sentences from your actual validation dataset
english_sources = [
    "Hello, how are you today?",
    "I need to go to the supermarket.",
    "What is your name?"
]

# The "Ground Truth" (100% perfect human translations)
spanish_references = [
    "Hola, ¿cómo estás hoy?",
    "Necesito ir al supermercado.",
    "¿Cómo te llamas?"
]

model_predictions = []

tokenizer.src_lang = "en_XX"
tgt_code = tokenizer.lang_code_to_id["es_XX"]

print("Translating test sentences...")
for text in english_sources:
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        generated_tokens = model.generate(**inputs, forced_bos_token_id=tgt_code)
    translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    model_predictions.append(translation)
    print(f"Original: {text} | Translated: {translation}")

# Calculate Scores
print("\n📊 Translation Scores:")
bleu_results = bleu.compute(predictions=model_predictions, references=spanish_references)
print(f"BLEU Score: {bleu_results['bleu']:.4f} (Higher is better)")

rouge_results = rouge.compute(predictions=model_predictions, references=spanish_references)
print(f"ROUGE-L Score: {rouge_results['rougeL']:.4f} (Higher is better)")

meteor_results = meteor.compute(predictions=model_predictions, references=spanish_references)
print(f"METEOR Score: {meteor_results['meteor']:.4f} (Higher is better)")


# ---------------------------------------------------------
# 2. EVALUATING SPEECH RECOGNITION (ASR)
# ---------------------------------------------------------
print("\n--- SPEECH RECOGNITION EVALUATION (ASR) ---")
# WER is used to see how many words the AI misheard.
# To do this in your report, you speak a sentence into your app, 
# look at "What the AI Heard", and compare it to what you ACTUALLY said.

actual_spoken_words = "Hola cómo estás hoy"
what_whisper_heard = "Hola como estas hoy" # (Missing accents, as we saw earlier!)

wer_score = jiwer.wer(actual_spoken_words, what_whisper_heard)
print(f"Word Error Rate (WER): {wer_score:.4f} (Lower is better)")