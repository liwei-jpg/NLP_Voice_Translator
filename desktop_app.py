import tkinter as tk
from tkinter import scrolledtext, font
import threading
import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from peft import PeftModel
import whisper
import edge_tts
import asyncio
import tempfile
import speech_recognition as sr
import os
import pygame

# --- HELPER FUNCTION: TEXT TO SPEECH ---
def generate_voice_audio(text, target_language):
    if target_language == "es":
        voice_model = "es-ES-AlvaroNeural"
    else:
        voice_model = "en-US-ChristopherNeural"
        
    communicate = edge_tts.Communicate(text, voice_model)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    temp_filename = temp_file.name
    temp_file.close() 
    
    asyncio.run(communicate.save(temp_filename))
    return temp_filename

# --- THE BACKGROUND AI THREAD ---
def ai_processing_thread(update_ui_callback):
    try:
        update_ui_callback("Status: Booting Models (This takes a moment)...")
        
        # 1. Load Models
        asr_model = whisper.load_model("small")
        MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"
        tokenizer = MBart50TokenizerFast.from_pretrained(MODEL_NAME)
        base_model = MBartForConditionalGeneration.from_pretrained(MODEL_NAME)
        model = PeftModel.from_pretrained(base_model, "./saved_lora_adapter")
        
        pygame.mixer.init()
        recognizer = sr.Recognizer()
        recognizer.pause_threshold = 1.5 
        
        with sr.Microphone() as source:
            update_ui_callback("Status: Profiling room noise for 2 seconds...")
            recognizer.adjust_for_ambient_noise(source, duration=2)
            
            # The Infinite Autonomous Loop
            while not stop_event.is_set():
                update_ui_callback("Status: 🎙️ Listening... (Speak naturally)")
                try:
                    # Listen for audio, auto-stops when silence is detected
                    audio_data = recognizer.listen(source, timeout=1, phrase_time_limit=15)
                except sr.WaitTimeoutError:
                    continue # Loop back and keep checking if no one spoke
                
                update_ui_callback("Status: ⏳ Silence detected. Processing...")
                
                # Save audio to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
                    temp_wav.write(audio_data.get_wav_data())
                    temp_wav_path = temp_wav.name
                
                # Transcribe with anti-hallucination guardrails
                transcription = asr_model.transcribe(
                    temp_wav_path,
                    condition_on_previous_text=False
                )
                recognized_text = transcription["text"].strip()
                detected_lang = transcription["language"] 
                os.remove(temp_wav_path)
                
                if not recognized_text:
                    continue
                    
                update_ui_callback("Status: ⚙️ Routing & Translating...")
                
                # Dynamic Logic
                if detected_lang == "en":
                    src_code, tgt_code, target_audio_lang = "en_XX", "es_XX", "es"
                    lang_label = "[Detected: English]"
                elif detected_lang == "es":
                    src_code, tgt_code, target_audio_lang = "es_XX", "en_XX", "en"
                    lang_label = "[Detected: Spanish]"
                else:
                    update_ui_callback("Status: ⚠️ Could not clearly hear English or Spanish. Listening again...")
                    continue

                # Translate with adapter logic
                tokenizer.src_lang = src_code
                inputs = tokenizer(recognized_text, return_tensors="pt")
                
                with torch.no_grad():
                    if detected_lang == "es":
                        with model.disable_adapter():
                            generated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id[tgt_code])
                    else:
                        generated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id[tgt_code])
                
                translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
                final_original = f"{lang_label} {recognized_text}"
                
                update_ui_callback("Status: 🔊 Speaking translation...", final_original, translated_text)
                
                # Speak Translation
                audio_output_path = generate_voice_audio(translated_text, target_audio_lang)
                pygame.mixer.music.load(audio_output_path)
                pygame.mixer.music.play()
                
                # Wait for audio to finish playing so it doesn't hear itself
                while pygame.mixer.music.get_busy() and not stop_event.is_set():
                    pygame.time.Clock().tick(10)
                    
                pygame.mixer.music.unload()
                os.remove(audio_output_path)
                
    except Exception as e:
        update_ui_callback(f"Status: Error occurred - {str(e)}")

# --- TKINTER UI SETUP (MAIN THREAD) ---
def update_ui_from_thread(status_text, original_text=None, translated_text=None):
    # This safely updates the UI from the background thread
    status_label.config(text=status_text)
    
    if original_text is not None and translated_text is not None:
        text_heard.config(state=tk.NORMAL)
        text_heard.delete(1.0, tk.END)
        text_heard.insert(tk.END, original_text)
        text_heard.config(state=tk.DISABLED)
        
        text_translated.config(state=tk.NORMAL)
        text_translated.delete(1.0, tk.END)
        text_translated.insert(tk.END, translated_text)
        text_translated.config(state=tk.DISABLED)

def on_closing():
    status_label.config(text="Status: Shutting down AI threads...")
    stop_event.set() # Tell the background loop to stop
    root.destroy()

# Initialize Tkinter Window
root = tk.Tk()
root.title("Smart Auto-Detect AI Translator")
root.geometry("700x500")
root.configure(padx=20, pady=20)
root.protocol("WM_DELETE_WINDOW", on_closing)

# Styling
title_font = font.Font(family="Helvetica", size=16, weight="bold")
label_font = font.Font(family="Helvetica", size=12)

tk.Label(root, text="🎙️ Hands-Free Autonomous Translator", font=title_font).pack(pady=(0, 10))
tk.Label(root, text="Speak clearly in English or Spanish. The AI will automatically detect, translate, and reply.", wraplength=600).pack(pady=(0, 20))

# Status Bar
status_label = tk.Label(root, text="Status: Initializing...", font=label_font, fg="blue")
status_label.pack(pady=(0, 20))

# What the AI Heard Box
tk.Label(root, text="What the AI Heard:", font=label_font, anchor="w").pack(fill="x")
text_heard = scrolledtext.ScrolledText(root, height=4, font=("Helvetica", 11), bg="#f4f4f4")
text_heard.pack(fill="x", pady=(0, 15))
text_heard.config(state=tk.DISABLED)

# Translation Box
tk.Label(root, text="Translation Output:", font=label_font, anchor="w").pack(fill="x")
text_translated = scrolledtext.ScrolledText(root, height=4, font=("Helvetica", 11), bg="#f4f4f4")
text_translated.pack(fill="x")
text_translated.config(state=tk.DISABLED)

# Start the background AI Thread
stop_event = threading.Event()
ai_thread = threading.Thread(target=ai_processing_thread, args=(update_ui_from_thread,), daemon=True)
ai_thread.start()

# Start the GUI Loop
root.mainloop()