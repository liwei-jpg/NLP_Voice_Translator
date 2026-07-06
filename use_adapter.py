import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from peft import PeftModel

print("⚙️ Loading the base mBART model... (This is fast since it is already cached)")
MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"
tokenizer = MBart50TokenizerFast.from_pretrained(MODEL_NAME)
base_model = MBartForConditionalGeneration.from_pretrained(MODEL_NAME)

print("🧬 Injecting your custom fine-tuned LoRA adapter...")
# This merges your newly trained conversational skills into the base model
model = PeftModel.from_pretrained(base_model, "./saved_lora_adapter")

print("✅ Fine-tuned model is ready for use!")

def translate_english_to_spanish(text):
    # Set language tokens
    tokenizer.src_lang = "en_XX"
    
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt")
    
    # Generate translation forcing Spanish output code
    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.lang_code_to_id["es_XX"]
        )
    
    # Decode back to readable text
    translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return translation

# --- TEST THE NEW MODEL ---
if __name__ == "__main__":
    print("\n--- testing custom fine-tuned translator ---")
    
    test_sentence = "Hey friend, are we still meeting up for dinner tonight or what?"
    print(f"🗣️ English Input: {test_sentence}")
    
    print("⚡ Translating with your custom conversational skills...")
    result = translate_english_to_spanish(test_sentence)
    
    print(f"🇪🇸 Conversational Spanish Output: {result}")