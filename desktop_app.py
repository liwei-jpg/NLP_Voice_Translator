# Import the standard Python library used to create graphical user interfaces (windows, text boxes).
import tkinter as tk
# Import specific UI tools: 'scrolledtext' makes a text box with a scrollbar, 'font' lets us change text size.
from tkinter import scrolledtext, font
# Import threading. This is critical: it lets the computer run the heavy AI in the background without freezing the visible UI.
import threading
# Import PyTorch, the core machine learning library that does all the heavy math on the GPU/CPU.
import torch
# Import the blueprints needed to run the mBART translation model and its text tokenizer.
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
# Import 'Parameter-Efficient Fine-Tuning'. This is needed to load our small custom LoRA training weights.
from peft import PeftModel
# Import OpenAI's Whisper library, which we use as the 'Ears' to turn speech into text.
import whisper
# Import Microsoft's Edge Text-to-Speech library, which we use as the 'Mouth' to turn text into spoken audio.
import edge_tts
# Import a library that handles "asynchronous" code. Needed because edge_tts downloads voice files in the background.
import asyncio
# Import a tool to create temporary files on the hard drive that automatically delete themselves to save storage space.
import tempfile
# Import the library that handles microphone access and Voice Activity Detection (VAD) using RMS math.
import speech_recognition as sr
# Import operating system commands, used here to manually delete temporary audio files when we are done with them.
import os
# Import a library originally made for games, but we use its audio 'mixer' because it plays mp3 files very smoothly.
import pygame

# --- HELPER FUNCTION: TEXT TO SPEECH ---
# This function takes translated text and turns it into a spoken MP3 file.
def generate_voice_audio(text, target_language):
    # Check if the target language is Spanish ("es").
    if target_language == "es":
        # If yes, use a highly realistic Spanish male neural voice.
        voice_model = "es-ES-AlvaroNeural"
    else:
        # If no, use a highly realistic English male neural voice.
        voice_model = "en-US-ChristopherNeural"
        
    # Set up the Edge-TTS communicator with the translated text and the chosen voice model.
    communicate = edge_tts.Communicate(text, voice_model)
    # Create an invisible, temporary mp3 file on the computer's hard drive to hold the audio.
    # delete=False means it won't delete itself immediately, so we have time to read and play it.
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    # Save the exact path/name of this temporary file into a variable.
    temp_filename = temp_file.name
    # Close the file so Edge-TTS has permission to write audio data into it.
    temp_file.close() 
    
    # asyncio.run() tells Python to wait until the audio file finishes downloading completely before moving to the next line.
    asyncio.run(communicate.save(temp_filename))
    # Return the file path so the main loop knows where to find the audio to play it.
    return temp_filename

# --- THE BACKGROUND AI THREAD ---
# This function is the "Brain". It runs on a separate CPU core (Background Thread) so the visual UI doesn't freeze.
def ai_processing_thread(update_ui_callback):
    try:
        # Send a message to the visual UI saying we are loading the models.
        update_ui_callback("Status: Booting Models (This takes a moment)...")
        
        # 1. Load Models
        # Load the small version of Whisper (Speech-to-Text).
        asr_model = whisper.load_model("small")
        # Define the exact name of the Facebook translation model we are using.
        MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"
        # Load the tokenizer (which chops sentences into numbers the AI understands).
        tokenizer = MBart50TokenizerFast.from_pretrained(MODEL_NAME)
        # Load the base "vanilla" translation model.
        base_model = MBartForConditionalGeneration.from_pretrained(MODEL_NAME)
        # Combine the base model with our custom LoRA training weights.
        model = PeftModel.from_pretrained(base_model, "./saved_lora_adapter")
        
        # Initialize the audio player so it is ready to output sound.
        pygame.mixer.init()
        # Initialize the microphone listener.
        recognizer = sr.Recognizer()
        # SETTING THE SILENCE THRESHOLD:
        # Once the user starts talking, the AI sets a stopwatch. 
        # pause_threshold = 1.5 means the AI will wait for exactly 1.5 seconds of total silence before it officially cuts the recording.
        recognizer.pause_threshold = 1.5 
        
        # Open the physical microphone hardware.
        with sr.Microphone() as source:
            # Send a status update to the screen.
            update_ui_callback("Status: Profiling room noise for 2 seconds...")
            # SETTING THE VOLUME DIFFERENTIATOR (RMS MATH):
            # This line listens to the microphone for exactly 2 seconds.
            # It mathematically calculates the RMS (Root Mean Square) of the background room noise (like AC or fans).
            # RMS turns negative sound waves into positive numbers and finds the average volume.
            # It then automatically creates an 'energy_threshold' variable slightly higher than that background noise.
            # Any sound below this threshold is ignored. Any sound above it triggers the recording!
            recognizer.adjust_for_ambient_noise(source, duration=2)
            
            # The Infinite Autonomous Loop. This keeps running forever until the user closes the app (stop_event).
            while not stop_event.is_set():
                # Tell the user they can start talking.
                update_ui_callback("Status: 🎙️ Listening... (Speak naturally)")
                try:
                    # Listen for audio, auto-stops when silence is detected
                    # timeout=1: Wait 1 second for a loud noise. If nothing, throw a WaitTimeoutError and loop back.
                    # phrase_time_limit=15: Force the recording to stop after 15 seconds so the system doesn't crash from memory overload.
                    audio_data = recognizer.listen(source, timeout=1, phrase_time_limit=15)
                except sr.WaitTimeoutError:
                    # If 1 second passed and no one spoke above the energy threshold, just loop back and check again.
                    continue # Loop back and keep checking if no one spoke
                
                # If we get here, the user spoke and then paused for 1.5 seconds.
                update_ui_callback("Status: ⏳ Silence detected. Processing...")
                
                # Save audio to temp file
                # Create a temporary .wav file and save the raw microphone recording into it.
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
                    temp_wav.write(audio_data.get_wav_data())
                    temp_wav_path = temp_wav.name
                
                # Transcribe with anti-hallucination guardrails
                # 'condition_on_previous_text=False' is our strict anti-hallucination fix.
                # It stops Whisper from trying to "remember" past sentences, preventing the AI from getting stuck in an infinite repeating loop.
                transcription = asr_model.transcribe(
                    temp_wav_path,
                    condition_on_previous_text=False
                )
                # Extract the actual text string and strip any accidental blank spaces.
                recognized_text = transcription["text"].strip()
                # Extract the language code that Whisper automatically detected (e.g., "en" or "es").
                detected_lang = transcription["language"] 
                # Delete the temporary .wav file from the hard drive since we have the text now.
                os.remove(temp_wav_path)
                
                # If the AI heard a loud noise (like a cough) but couldn't find any words, loop back and start over.
                if not recognized_text:
                    continue
                    
                # Update the screen to show we are moving to the translation phase.
                update_ui_callback("Status: ⚙️ Routing & Translating...")
                
                # Dynamic Logic
                # We check the language Whisper detected, and configure the variables for the translation model.
                if detected_lang == "en":
                    # If it heard English, setup the codes to translate into Spanish.
                    src_code, tgt_code, target_audio_lang = "en_XX", "es_XX", "es"
                    lang_label = "[Detected: English]"
                elif detected_lang == "es":
                    # If it heard Spanish, setup the codes to translate into English.
                    src_code, tgt_code, target_audio_lang = "es_XX", "en_XX", "en"
                    lang_label = "[Detected: Spanish]"
                else:
                    # If someone spoke French or Mandarin, throw an error to the UI and start listening again.
                    update_ui_callback("Status: ⚠️ Could not clearly hear English or Spanish. Listening again...")
                    continue

                # Translate with adapter logic
                # Tell the mBART tokenizer what language the source text is in.
                tokenizer.src_lang = src_code
                # Convert the raw text into tensor numbers that the AI model can calculate.
                inputs = tokenizer(recognized_text, return_tensors="pt")
                
                # torch.no_grad() tells PyTorch not to calculate gradients, saving massive memory since we are testing, not training.
                with torch.no_grad():
                    # THE LORA BIAS FIX:
                    # Our LoRA adapter was specifically trained to translate English TO Spanish.
                    if detected_lang == "es":
                        # If the user spoke Spanish, using the adapter backward would cause bias or errors.
                        # So, we use 'disable_adapter()' to temporarily turn the LoRA off and just use the base model.
                        with model.disable_adapter():
                            generated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id[tgt_code])
                    else:
                        # If the user spoke English, we leave the LoRA adapter ON, because this is exactly what we trained it to do!
                        generated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id[tgt_code])
                
                # Convert the generated mathematical tokens back into human-readable text.
                translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
                # Combine the language label with what was heard to look nice in the UI.
                final_original = f"{lang_label} {recognized_text}"
                
                # Send the original text and the translated text to the visual interface.
                update_ui_callback("Status: 🔊 Speaking translation...", final_original, translated_text)
                
                # Speak Translation
                # Pass the translated text to our Edge-TTS helper function to get an mp3 file.
                audio_output_path = generate_voice_audio(translated_text, target_audio_lang)
                # Load the mp3 file into the audio player.
                pygame.mixer.music.load(audio_output_path)
                # Play the audio out loud through the laptop speakers.
                pygame.mixer.music.play()
                
                # Wait for audio to finish playing so it doesn't hear itself
                # IMPORTANT: If we don't pause the code here, the microphone will immediately start listening again,
                # hear the laptop speakers, and try to translate its own translation in an infinite loop.
                while pygame.mixer.music.get_busy() and not stop_event.is_set():
                    # Wait 1/10th of a second, then check if the audio is still playing.
                    pygame.time.Clock().tick(10)
                    
                # Once the audio finishes, unload the file from memory so it isn't locked.
                pygame.mixer.music.unload()
                # Delete the mp3 from the hard drive to save space.
                os.remove(audio_output_path)
                
    except Exception as e:
        # If any major error happens, catch it and print it to the screen so the app doesn't just silently crash.
        update_ui_callback(f"Status: Error occurred - {str(e)}")

# --- TKINTER UI SETUP (MAIN THREAD) ---
def update_ui_from_thread(status_text, original_text=None, translated_text=None):
    # This safely updates the UI from the background thread
    # It acts as a bridge so the AI thread can push text to the screen without breaking Tkinter.
    status_label.config(text=status_text)
    
    if original_text is not None and translated_text is not None:
        # Update the "What the AI Heard" box. Unlock it, delete old text, insert new text, then lock it again.
        text_heard.config(state=tk.NORMAL)
        text_heard.delete(1.0, tk.END)
        text_heard.insert(tk.END, original_text)
        text_heard.config(state=tk.DISABLED)
        
        # Update the "Translation Output" box using the exact same logic.
        text_translated.config(state=tk.NORMAL)
        text_translated.delete(1.0, tk.END)
        text_translated.insert(tk.END, translated_text)
        text_translated.config(state=tk.DISABLED)

def on_closing():
    # This function triggers when the user clicks the "X" to close the window.
    status_label.config(text="Status: Shutting down AI threads...")
    stop_event.set() # Tell the background loop to stop
    # Safely destroy the application window.
    root.destroy()

# Initialize Tkinter Window
# Create the main visual desktop window.
root = tk.Tk()
# Set the window title bar text.
root.title("Smart Auto-Detect AI Translator")
# Set the window size to 700 pixels wide by 500 pixels tall.
root.geometry("700x500")
# Add 20 pixels of padding inside the window so text doesn't touch the edges.
root.configure(padx=20, pady=20)
# Tell the window to run our on_closing() function if the user hits the "X".
root.protocol("WM_DELETE_WINDOW", on_closing)

# Styling
# Define the visual fonts we will use for the text titles and labels.
title_font = font.Font(family="Helvetica", size=16, weight="bold")
label_font = font.Font(family="Helvetica", size=12)

# Draw the Title and Subtitle on the screen.
tk.Label(root, text="🎙️ Hands-Free Autonomous Translator", font=title_font).pack(pady=(0, 10))
tk.Label(root, text="Speak clearly in English or Spanish. The AI will automatically detect, translate, and reply.", wraplength=600).pack(pady=(0, 20))

# Status Bar
# Draw the blue Status Bar text on the screen.
status_label = tk.Label(root, text="Status: Initializing...", font=label_font, fg="blue")
status_label.pack(pady=(0, 20))

# What the AI Heard Box
# Draw the "What the AI Heard" header and its grey text box.
tk.Label(root, text="What the AI Heard:", font=label_font, anchor="w").pack(fill="x")
text_heard = scrolledtext.ScrolledText(root, height=4, font=("Helvetica", 11), bg="#f4f4f4")
text_heard.pack(fill="x", pady=(0, 15))
text_heard.config(state=tk.DISABLED)

# Translation Box
# Draw the "Translation Output" header and its grey text box.
tk.Label(root, text="Translation Output:", font=label_font, anchor="w").pack(fill="x")
text_translated = scrolledtext.ScrolledText(root, height=4, font=("Helvetica", 11), bg="#f4f4f4")
text_translated.pack(fill="x")
text_translated.config(state=tk.DISABLED)

# Start the background AI Thread
# Create the event switch that will eventually tell the background thread to shut down.
stop_event = threading.Event()
# Create the actual background thread, telling it to run our 'ai_processing_thread' function.
# daemon=True means if the main window crashes, this background thread will be forcefully killed too.
ai_thread = threading.Thread(target=ai_processing_thread, args=(update_ui_from_thread,), daemon=True)
# Start the background thread.
ai_thread.start()

# Start the GUI Loop
# Keeps the window open on the screen and responsive to mouse clicks.
root.mainloop()