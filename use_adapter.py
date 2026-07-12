# Import PyTorch to handle the mathematical operations on your GPU/CPU
import torch
# Import the standard tools to handle the mBART model and its vocabulary
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
# Import PEFT (Parameter-Efficient Fine-Tuning) to manage your custom adapter weights
from peft import PeftModel

# This is an initialization step. Since we already trained the model, 
# this downloads the "Brain" structure quickly from the Hugging Face cache.
print("⚙️ Loading the base mBART model... (This is fast since it is already cached)")
MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"
# The tokenizer acts as the translator between human text and machine numbers
tokenizer = MBart50TokenizerFast.from_pretrained(MODEL_NAME)
# The base_model is the massive "vanilla" translation engine
base_model = MBartForConditionalGeneration.from_pretrained(MODEL_NAME)

print("🧬 Injecting your custom fine-tuned LoRA adapter...")
# This is the "Aha!" moment of the project. 
# We are 'injecting' your tiny 5MB LoRA adapter file into the 2.4GB base model. 
# This adds your specific 'conversational' intelligence to the base model's 'general' intelligence.
model = PeftModel.from_pretrained(base_model, "./saved_lora_adapter")

print("✅ Fine-tuned model is ready for use!")

def translate_english_to_spanish(text):
    # mBART is multilingual; we must explicitly tell it the source language is English
    tokenizer.src_lang = "en_XX"
    
    # Transform human text into numerical tokens (tensors) that PyTorch understands
    inputs = tokenizer(text, return_tensors="pt")
    
    # We use torch.no_grad() because this is 'inference' (testing/use), not training.
    # This prevents the model from calculating gradients, which saves massive amounts of memory.
    with torch.no_grad():
        # Generate the Spanish translation. 
        # The 'forced_bos_token_id' ensures the model starts by outputting a Spanish language tag.
        generated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.lang_code_to_id["es_XX"]
        )
    
    # Take the numerical result and decode it back into plain Spanish text
    # skip_special_tokens=True removes the invisible technical tags used by the AI
    translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return translation

# --- TEST THE NEW MODEL ---
# This block only runs if you run this specific file directly
if __name__ == "__main__":
    print("\n--- testing custom fine-tuned translator ---")
    
    # Test a casual sentence to show off the LoRA's ability to translate slang naturally
    test_sentence = "Hey friend, are we still meeting up for dinner tonight or what?"
    print(f"🗣️ English Input: {test_sentence}")
    
    print("⚡ Translating with your custom conversational skills...")
    # Call our translation function
    result = translate_english_to_spanish(test_sentence)
    
    print(f"🇪🇸 Conversational Spanish Output: {result}")