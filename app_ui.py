import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from peft import PeftModel
import gradio as gr
import whisper
import edge_tts
import asyncio
import tempfile

print("⚙️ Loading AI Models... (This will take a few moments)")

# 1. Load the "Ears" (Whisper)
print("🎧 Loading Whisper Speech Recognition...")
asr_model = whisper.load_model("small")

# 2. Load the "Brain" (mBART + Your LoRA Adapter)
print("🧠 Loading mBART Translation Model...")
MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"
tokenizer = MBart50TokenizerFast.from_pretrained(MODEL_NAME)
base_model = MBartForConditionalGeneration.from_pretrained(MODEL_NAME)
model = PeftModel.from_pretrained(base_model, "./saved_lora_adapter")

print("✅ Full Voice-to-Voice System Online!")

# --- HELPER FUNCTION: TEXT TO SPEECH ---
def generate_voice_audio(text, target_language):
    # Dynamically select the voice based on the target translation language
    if target_language == "es":
        voice_model = "es-ES-AlvaroNeural"  # Spanish Voice
    else:
        voice_model = "en-US-ChristopherNeural" # English Voice
        
    communicate = edge_tts.Communicate(text, voice_model)
    
    # Create temp file that Gradio will manage and automatically clean up
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    temp_filename = temp_file.name
    temp_file.close() 
    
    # Run the asynchronous save operation safely
    asyncio.run(communicate.save(temp_filename))
    return temp_filename

# --- MAIN PIPELINE FUNCTION (AUTO-DETECT) ---
def process_voice_to_voice(audio_filepath):
    if not audio_filepath:
        return "No audio detected.", "Please record something.", None
        
    print("\n🎤 1. Analyzing audio...")
    
    # Step 1: Transcribe with punctuation hints to fix Spanish accents
    transcription = asr_model.transcribe(
        audio_filepath,
        condition_on_previous_text=False # This strictly prevents repetitive hallucination loops
    )
    recognized_text = transcription["text"].strip()
    detected_lang = transcription["language"] # Returns 'en' or 'es'
    
    print("⚙️ 2. Routing & Translating...")
    
    # Step 2: Dynamic Logic based on detected language
    if detected_lang == "en":
        print(f"   Detected English. Translating to Spanish...")
        src_code = "en_XX"
        tgt_code = "es_XX"
        target_audio_lang = "es"
        lang_label = "[Detected: English]"
    elif detected_lang == "es":
        print(f"   Detected Spanish. Translating to English...")
        src_code = "es_XX"
        tgt_code = "en_XX"
        target_audio_lang = "en"
        lang_label = "[Detected: Spanish]"
    else:
        # Fallback if it hears background noise and thinks it's French/German etc.
        return f"[Detected: {detected_lang}] Could not clearly hear English or Spanish.", "", None

    # Step 3: Prepare text for translation
    tokenizer.src_lang = src_code
    inputs = tokenizer(recognized_text, return_tensors="pt")
    
    with torch.no_grad():
        # Step 4: Translate with adapter logic
        # Disable the adapter for ES->EN to prevent the Spanish-only bias
        if detected_lang == "es":
            with model.disable_adapter():
                generated_tokens = model.generate(
                    **inputs,
                    forced_bos_token_id=tokenizer.lang_code_to_id[tgt_code]
                )
        else:
            # Leave the LoRA adapter ON for EN->ES
            generated_tokens = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.lang_code_to_id[tgt_code]
            )
    
    translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    
    print("🔊 3. Generating audio response...")
    # Step 5: Speak the translation
    audio_output_path = generate_voice_audio(translated_text, target_audio_lang)
    
    print("✅ Pipeline complete!")
    
    # Combine the label and the text for a clean UI output
    final_original_text = f"{lang_label} {recognized_text}"
    
    return final_original_text, translated_text, audio_output_path

# --- GRADIO UI LAYOUT ---
with gr.Blocks(title="Smart AI Voice Translator") as demo:
    gr.Markdown("# 🎙️ Smart Auto-Detect AI Translator")
    gr.Markdown("Click record and speak in **either** English or Spanish. The AI will automatically detect your language and translate it.")
    
    with gr.Row():
        audio_in = gr.Audio(
            sources=["microphone"], 
            type="filepath", 
            label="1. Tap to Speak (Wait 1 second before talking)"
        )
        
    with gr.Row():
        submit_btn = gr.Button("⚡ Auto-Translate", variant="primary")
        
    with gr.Row():
        text_out_original = gr.Textbox(label="2. What the AI Heard", interactive=False)
        text_out_translated = gr.Textbox(label="3. Translation", interactive=False)
        
    with gr.Row():
        audio_out = gr.Audio(label="4. Spoken Translation", interactive=False)
        
    # Bind the pipeline to the button
    submit_btn.click(
        fn=process_voice_to_voice,
        inputs=audio_in,
        outputs=[text_out_original, text_out_translated, audio_out]
    )

if __name__ == "__main__":
    print("🚀 Launching the local interface web server...")
    demo.launch(share=False)