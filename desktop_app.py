import tkinter as tk # Standard Python UI library to draw the window and buttons.
from tkinter import scrolledtext, font # Specific tools for creating text display areas.
import threading # Crucial: Allows the AI to run in the background without freezing the UI.
import torch # The math library that calculates the AI predictions.
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast # Core translation AI tools.
from peft import PeftModel # Loads your custom, lightweight LoRA 'personality' weights.
import whisper # The "ears": converts user's microphone audio into text.
import edge_tts # The "mouth": converts text into realistic human speech.
import asyncio # Manages background tasks (like downloading audio) without stopping the program.
import tempfile # Safely creates temporary files in memory that delete themselves after use.
import speech_recognition as sr # Manages the microphone and Voice Activity Detection (VAD).
import os # Operating system tools, used here to delete temporary audio files from the hard drive.
import pygame # Audio engine: chosen because it handles MP3 playback very smoothly.

# --- HELPER FUNCTION: TEXT TO SPEECH ---
def generate_voice_audio(text, target_language):
    # Select the specific voice model based on the target language.
    # Neural voices sound more human than traditional synthetic robotic voices.
    if target_language == "es":
        voice_model = "es-ES-AlvaroNeural" 
    else:
        voice_model = "en-US-ChristopherNeural"
        
    communicate = edge_tts.Communicate(text, voice_model)
    
    # We create a temporary MP3 file to save the speech data.
    # delete=False is used so we can close the file and still read it to play.
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    temp_filename = temp_file.name
    temp_file.close() 
    
    # asyncio.run runs the background downloading process synchronously for this thread.
    asyncio.run(communicate.save(temp_filename))
    return temp_filename

# --- THE BACKGROUND AI THREAD ---
def ai_processing_thread(update_ui_callback):
    try:
        # We update the UI to show that the system is booting (loading heavy models).
        update_ui_callback("Status: Booting Models (This takes a moment)...")
        
        # 1. Load Models: We load these at the start so the AI is ready to respond instantly later.
        asr_model = whisper.load_model("small")
        MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"
        tokenizer = MBart50TokenizerFast.from_pretrained(MODEL_NAME)
        base_model = MBartForConditionalGeneration.from_pretrained(MODEL_NAME)
        # We inject your LoRA weights into the base model here.
        model = PeftModel.from_pretrained(base_model, "./saved_lora_adapter")
        
        pygame.mixer.init() # Wakes up the sound card.
        recognizer = sr.Recognizer() # Creates the microphone listening instance.
        
        # VAD THRESHOLD: Silence detection. If the user stops talking for 1.5 seconds, 
        # the app concludes the sentence is finished and stops recording.
        recognizer.pause_threshold = 1.5 
        
        with sr.Microphone() as source:
            update_ui_callback("Status: Profiling room noise for 2 seconds...")
            # VAD VOLUME DIFFERENTIATOR: The app listens to the room for 2 seconds 
            # and performs Root Mean Square (RMS) math to calculate the background 
            # noise floor (fans, humming, static). It then sets a threshold above this floor.
            recognizer.adjust_for_ambient_noise(source, duration=2)
            
            # The Infinite Autonomous Loop
            while not stop_event.is_set():
                update_ui_callback("Status: 🎙️ Listening... (Speak naturally)")
                try:
                    # Listen for audio until the silence threshold (1.5s) is triggered.
                    audio_data = recognizer.listen(source, timeout=1, phrase_time_limit=15)
                except sr.WaitTimeoutError:
                    continue # No one spoke, loop back to start listening again.
                
                update_ui_callback("Status: ⏳ Silence detected. Processing...")
                
                # Save the recorded voice to a temporary file for Whisper to transcribe.
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
                    temp_wav.write(audio_data.get_wav_data())
                    temp_wav_path = temp_wav.name
                
                # Transcribe with anti-hallucination guardrails.
                # Setting condition_on_previous_text=False prevents the AI from getting 
                # stuck in loops by repeating previously transcribed words.
                transcription = asr_model.transcribe(
                    temp_wav_path,
                    condition_on_previous_text=False
                )
                recognized_text = transcription["text"].strip()
                detected_lang = transcription["language"] 
                os.remove(temp_wav_path) # Clean up the temporary file immediately.
                
                if not recognized_text:
                    continue # Skip empty recordings (coughing or clicking).
                    
                update_ui_callback("Status: ⚙️ Routing & Translating...")
                
                # Dynamic Logic: Choose the translation direction automatically.
                if detected_lang == "en":
                    src_code, tgt_code, target_audio_lang = "en_XX", "es_XX", "es"
                    lang_label = "[Detected: English]"
                elif detected_lang == "es":
                    src_code, tgt_code, target_audio_lang = "es_XX", "en_XX", "en"
                    lang_label = "[Detected: Spanish]"
                else:
                    update_ui_callback("Status: ⚠️ Could not clearly hear English or Spanish...")
                    continue

                # Prepare the brain (Tokenizer) for the specific translation direction.
                tokenizer.src_lang = src_code
                inputs = tokenizer(recognized_text, return_tensors="pt")
                
                with torch.no_grad():
                    # ADAPTER BIAS LOGIC: 
                    # If translating back to English, we disable the LoRA adapter (your
                    # Spanish fine-tuning) to ensure accurate, neutral translation.
                    if detected_lang == "es":
                        with model.disable_adapter():
                            generated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id[tgt_code])
                    else:
                        # Otherwise, we keep it enabled to use your custom Spanish style.
                        generated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id[tgt_code])
                
                translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
                final_original = f"{lang_label} {recognized_text}"
                
                update_ui_callback("Status: 🔊 Speaking translation...", final_original, translated_text)
                
                # Speak the translated text.
                audio_output_path = generate_voice_audio(translated_text, target_audio_lang)
                pygame.mixer.music.load(audio_output_path)
                pygame.mixer.music.play()
                
                # IMPORTANT: Loop until audio finishes. If we don't wait, 
                # the microphone will pick up its own voice and cause a feedback loop.
                while pygame.mixer.music.get_busy() and not stop_event.is_set():
                    pygame.time.Clock().tick(10)
                    
                pygame.mixer.music.unload()
                os.remove(audio_output_path) # Clean up audio after playing.
                
    except Exception as e:
        update_ui_callback(f"Status: Error occurred - {str(e)}")

# --- TKINTER UI SETUP (MAIN THREAD) ---
def update_ui_from_thread(status_text, original_text=None, translated_text=None):
    # This acts as a "bridge" so the background AI can safely push updates to the UI thread.
    status_label.config(text=status_text)
    
    if original_text is not None and translated_text is not None:
        # Unlock the box, insert text, then lock it again so the user cannot manually edit it.
        text_heard.config(state=tk.NORMAL)
        text_heard.delete(1.0, tk.END)
        text_heard.insert(tk.END, original_text)
        text_heard.config(state=tk.DISABLED)
        
        text_translated.config(state=tk.NORMAL)
        text_translated.delete(1.0, tk.END)
        text_translated.insert(tk.END, translated_text)
        text_translated.config(state=tk.DISABLED)

def on_closing():
    # Safely kill the background thread before closing the window.
    status_label.config(text="Status: Shutting down AI threads...")
    stop_event.set() 
    root.destroy()

# Initialize Tkinter Window (The Graphical User Interface)
root = tk.Tk()
root.title("Smart Auto-Detect AI Translator")
root.geometry("700x500")
root.configure(padx=20, pady=20)
root.protocol("WM_DELETE_WINDOW", on_closing) # Links the "X" button to our clean shutdown function.

# UI Fonts and Labels setup... (Standard Tkinter layout code)
title_font = font.Font(family="Helvetica", size=16, weight="bold")
label_font = font.Font(family="Helvetica", size=12)

tk.Label(root, text="🎙️ Hands-Free Autonomous Translator", font=title_font).pack(pady=(0, 10))
# ... (rest of the UI layout packing)

# START THE MULTI-THREADING ARCHITECTURE
stop_event = threading.Event()
# daemon=True ensures that if the main app crashes, the background thread is killed with it.
ai_thread = threading.Thread(target=ai_processing_thread, args=(update_ui_from_thread,), daemon=True)
ai_thread.start()

root.mainloop() # Keeps the window open and responsive.