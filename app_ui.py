# Import PyTorch, the core library that handles the heavy mathematical tensor calculations for the AI.
import torch
# Import the standard tools to handle the mBART translation model and its text-to-number converter.
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
# Import PEFT (Parameter-Efficient Fine-Tuning) so we can load your custom LoRA weights.
from peft import PeftModel
# Import Gradio, a library that automatically generates a beautiful web-based user interface for Python code.
import gradio as gr
# Import OpenAI's Whisper, which acts as the "Ears" of the system to turn speech into text.
import whisper
# Import Microsoft's Edge Text-to-Speech library, which acts as the "Mouth" to turn text into spoken audio.
import edge_tts
# Import asyncio to handle background downloading tasks (like fetching the TTS audio) without freezing the app.
import asyncio
# Import tempfile to create temporary files in the computer's RAM/storage that safely delete themselves later.
import tempfile

# Print a startup message to the console so the user knows the background loading process has begun.
print("⚙️ Loading AI Models... (This will take a few moments)")

# 1. Load the "Ears" (Whisper)
# Print a status update for the speech recognition loading phase.
print("🎧 Loading Whisper Speech Recognition...")
# Load the 'small' version of Whisper. It is highly optimized for speed and real-time conversation accuracy.
asr_model = whisper.load_model("small")

# 2. Load the "Brain" (mBART + Your LoRA Adapter)
# Print a status update for the translation model loading phase.
print("🧠 Loading mBART Translation Model...")
# Define the exact name of the Facebook foundation model hosted on Hugging Face.
MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"
# Load the tokenizer, which translates human words into mathematical numbers the AI can read.
tokenizer = MBart50TokenizerFast.from_pretrained(MODEL_NAME)
# Load the massive base translation engine into the computer's memory.
base_model = MBartForConditionalGeneration.from_pretrained(MODEL_NAME)
# Inject your specific, fine-tuned conversational LoRA adapter into the base model.
model = PeftModel.from_pretrained(base_model, "./saved_lora_adapter")

# Print a final startup message to confirm everything is ready.
print("✅ Full Voice-to-Voice System Online!")

# --- HELPER FUNCTION: TEXT TO SPEECH ---
# Define a function to convert the final translated text back into a spoken audio file.
def generate_voice_audio(text, target_language):
    # Dynamically select the voice based on the target translation language
    # Check if the AI needs to speak in Spanish.
    if target_language == "es":
        # Assign a high-quality, realistic male Spanish neural voice.
        voice_model = "es-ES-AlvaroNeural"  # Spanish Voice
    # Otherwise, it must speak in English.
    else:
        # Assign a high-quality, realistic male English neural voice.
        voice_model = "en-US-ChristopherNeural" # English Voice
        
    # Configure the Edge-TTS engine with the text and the selected voice model.
    communicate = edge_tts.Communicate(text, voice_model)
    
    # Create temp file that Gradio will manage and automatically clean up
    # 'delete=False' ensures the file isn't deleted before the audio player gets a chance to read it.
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    # Save the exact file path into a variable so we can find it later.
    temp_filename = temp_file.name
    # Close the file system lock so the TTS engine has permission to write audio data into it.
    temp_file.close() 
    
    # Run the asynchronous save operation safely
    # This contacts Microsoft's servers, generates the speech, and saves it to the temporary file path.
    asyncio.run(communicate.save(temp_filename))
    # Return the file path so the Gradio UI knows what audio file to play to the user.
    return temp_filename

# --- MAIN PIPELINE FUNCTION (AUTO-DETECT) ---
# This is the master function that connects the Ears, Brain, and Mouth together.
def process_voice_to_voice(audio_filepath):
    # If the user clicks the translate button without actually recording anything, stop the function to prevent a crash.
    if not audio_filepath:
        return "No audio detected.", "Please record something.", None
        
    # Print a terminal status update to show the pipeline has started.
    print("\n🎤 1. Analyzing audio...")
    
    # Step 1: Transcribe with punctuation hints to fix Spanish accents
    # Pass the user's recorded audio file to the Whisper model to convert it to text.
    transcription = asr_model.transcribe(
        # The path to the recorded audio file.
        audio_filepath,
        # IMPORTANT GUARDRAIL: condition_on_previous_text=False prevents the AI from trying to "remember" 
        # past sentences. This stops a common bug where Whisper hallucinates and repeats itself in an infinite loop.
        condition_on_previous_text=False # This strictly prevents repetitive hallucination loops
    )
    # Extract the transcribed text string and remove any accidental blank spaces from the edges.
    recognized_text = transcription["text"].strip()
    # Extract the language code that Whisper automatically detected (it will return 'en' or 'es').
    detected_lang = transcription["language"] # Returns 'en' or 'es'
    
    # Print a status update to show transcription is done and translation is starting.
    print("⚙️ 2. Routing & Translating...")
    
    # Step 2: Dynamic Logic based on detected language
    # We dynamically route the pipeline variables based on what language Whisper heard.
    if detected_lang == "en":
        # Log that we are doing an English-to-Spanish translation.
        print(f"   Detected English. Translating to Spanish...")
        # Set the source code for the tokenizer to English.
        src_code = "en_XX"
        # Set the target code for the translation generation to Spanish.
        tgt_code = "es_XX"
        # Set the target audio code for the Edge-TTS engine to Spanish.
        target_audio_lang = "es"
        # Create a UI label so the user knows what the AI detected.
        lang_label = "[Detected: English]"
    # If Whisper detected Spanish instead...
    elif detected_lang == "es":
        # Log that we are doing a Spanish-to-English translation.
        print(f"   Detected Spanish. Translating to English...")
        # Set the source code for the tokenizer to Spanish.
        src_code = "es_XX"
        # Set the target code for the translation generation to English.
        tgt_code = "en_XX"
        # Set the target audio code for the Edge-TTS engine to English.
        target_audio_lang = "en"
        # Create a UI label so the user knows what the AI detected.
        lang_label = "[Detected: Spanish]"
    # If Whisper heard a completely different language (like French) or just background static...
    else:
        # Fallback if it hears background noise and thinks it's French/German etc.
        # Safely exit the function and display an error message to the user UI.
        return f"[Detected: {detected_lang}] Could not clearly hear English or Spanish.", "", None

    # Step 3: Prepare text for translation
    # Explicitly tell the mBART tokenizer which language it is reading.
    tokenizer.src_lang = src_code
    # Convert the human text into mathematical tensor arrays that PyTorch can compute.
    inputs = tokenizer(recognized_text, return_tensors="pt")
    
    # Use torch.no_grad() to save massive amounts of computer memory since we are only testing, not training.
    with torch.no_grad():
        # Step 4: Translate with adapter logic
        # Disable the adapter for ES->EN to prevent the Spanish-only bias
        # This is a Domain Bias Fix. Our LoRA adapter was specifically trained to go from English TO Spanish.
        if detected_lang == "es":
            # If the user spoke Spanish, using the adapter backwards causes bad bias. 
            # We use 'disable_adapter()' to temporarily turn off the LoRA and just use the pure base model.
            with model.disable_adapter():
                # Command the model to predict the English translation.
                generated_tokens = model.generate(
                    # Unpack the numerical inputs we made earlier.
                    **inputs,
                    # Force the very first generated word to be the English language tag.
                    forced_bos_token_id=tokenizer.lang_code_to_id[tgt_code]
                )
        # If the user spoke English, we want our custom LoRA adapter turned ON!
        else:
            # Leave the LoRA adapter ON for EN->ES
            # Command the model to predict the Spanish translation using our conversational fine-tuned skills.
            generated_tokens = model.generate(
                # Unpack the numerical inputs.
                **inputs,
                # Force the very first generated word to be the Spanish language tag.
                forced_bos_token_id=tokenizer.lang_code_to_id[tgt_code]
            )
    
    # Take the output numbers generated by the AI and decode them back into human words, hiding the invisible technical tags.
    translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    
    # Print a status update to show the translation is finished and TTS is starting.
    print("🔊 3. Generating audio response...")
    # Step 5: Speak the translation
    # Pass the final translated text and target language code to our helper function to get an MP3 file path.
    audio_output_path = generate_voice_audio(translated_text, target_audio_lang)
    
    # Print a final terminal success message.
    print("✅ Pipeline complete!")
    
    # Combine the label and the text for a clean UI output
    # Example: "[Detected: English] Hello how are you"
    final_original_text = f"{lang_label} {recognized_text}"
    
    # Send the original text, the translated text, and the path to the MP3 file back to the Gradio web interface.
    return final_original_text, translated_text, audio_output_path

# --- GRADIO UI LAYOUT ---
# Create the Gradio web application blocks, setting the browser tab title.
with gr.Blocks(title="Smart AI Voice Translator") as demo:
    # Add a large header to the webpage.
    gr.Markdown("# 🎙️ Smart Auto-Detect AI Translator")
    # Add instructions for the user.
    gr.Markdown("Click record and speak in **either** English or Spanish. The AI will automatically detect your language and translate it.")
    
    # Create a horizontal row in the UI.
    with gr.Row():
        # Create an Audio input component that captures from the user's microphone and saves it as a file.
        audio_in = gr.Audio(
            sources=["microphone"], 
            type="filepath", 
            label="1. Tap to Speak (Wait 1 second before talking)"
        )
        
    # Create another horizontal row.
    with gr.Row():
        # Create a large primary button that the user will click to start the pipeline.
        submit_btn = gr.Button("⚡ Auto-Translate", variant="primary")
        
    # Create another horizontal row for the text results.
    with gr.Row():
        # Create a text box to show exactly what Whisper heard (Read-only).
        text_out_original = gr.Textbox(label="2. What the AI Heard", interactive=False)
        # Create a text box to show the final mBART translation (Read-only).
        text_out_translated = gr.Textbox(label="3. Translation", interactive=False)
        
    # Create the final horizontal row for the audio playback.
    with gr.Row():
        # Create an Audio output component to play the Edge-TTS MP3 file back to the user.
        audio_out = gr.Audio(label="4. Spoken Translation", interactive=False)
        
    # Bind the pipeline to the button
    # When 'submit_btn' is clicked, run 'process_voice_to_voice'.
    # Take the audio from 'audio_in' and send the 3 return values to the 3 output components.
    submit_btn.click(
        fn=process_voice_to_voice,
        inputs=audio_in,
        outputs=[text_out_original, text_out_translated, audio_out]
    )

# Standard Python check to ensure this code only runs if the file is executed directly.
if __name__ == "__main__":
    # Print a status message indicating the web server is starting.
    print("🚀 Launching the local interface web server...")
    # Launch the Gradio web server so it can be accessed in a web browser (share=False means local network only).
    demo.launch(share=False)
