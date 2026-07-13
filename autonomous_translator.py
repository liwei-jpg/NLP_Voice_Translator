# Import torch to handle the heavy math (tensor calculations) for the AI model
import torch
# Import standard library classes for handling mBART's language translation and tokenization
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
# Import PEFT (Parameter-Efficient Fine-Tuning) to inject your custom LoRA adapter "skills" into the base model
from peft import PeftModel
# Import Whisper for speech recognition (This acts as the "Ears" of our AI)
import whisper
# Import Microsoft's Edge-TTS for high-quality audio generation (This acts as the "Mouth")
import edge_tts
# Import asyncio to manage asynchronous tasks like downloading audio files in the background without freezing the app
import asyncio
# Import tempfile to create temporary files in RAM/storage that can automatically delete themselves
import tempfile
# Import speech_recognition (sr) to interface with your microphone hardware and handle Voice Activity Detection (VAD)
import speech_recognition as sr
# Import os (operating system tools) to manually delete temporary files from the hard drive
import os
# Import pygame, an engine that handles MP3 audio playback very smoothly
import pygame  # Used to play the audio locally

# Print a status message to the terminal to let the user know the system is starting
print("⚙️ Booting Autonomous AI Translator...")

# 1. Load Models
# Print a status update for the speech recognition loading phase
print("🎧 Loading Whisper (Ears)...")
# Load the 'base' version of Whisper. It is a highly optimized model that balances fast speed with good transcription accuracy.
asr_model = whisper.load_model("base")

# Print a status update for the translation model loading phase
print("🧠 Loading mBART (Brain)...")
# Define the exact name of the Facebook foundation model hosted on Hugging Face
MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"
# Load the tokenizer, which translates human words into mathematical numbers the AI can read
tokenizer = MBart50TokenizerFast.from_pretrained(MODEL_NAME)
# Load the massive base translation engine into the computer's memory
base_model = MBartForConditionalGeneration.from_pretrained(MODEL_NAME)
# Inject your specific, fine-tuned conversational LoRA adapter into the base model.
# LoRA (Low-Rank Adaptation) is used here because it allows us to add specific conversational intelligence
# without having to retrain the entire 2.4GB model, saving massive amounts of memory and compute time.
model = PeftModel.from_pretrained(base_model, "./saved_lora_adapter")

# Initialize pygame's audio mixer so the computer's sound card is ready to play the generated MP3s
# Initialize audio player
pygame.mixer.init()

# --- HELPER: TEXT TO SPEECH & PLAYBACK ---
# Define a helper function to generate speech audio and play it out loud
def speak_translation(text, target_language):
    # Dynamically assign a high-quality neural voice based on the target language (Spanish or English)
    voice_model = "es-ES-AlvaroNeural" if target_language == "es" else "en-US-ChristopherNeural"
    
    # Configure the Edge-TTS engine with the translated text and the selected voice model
    communicate = edge_tts.Communicate(text, voice_model)
    # Create a temporary MP3 file to store the speech data. 'delete=False' ensures it stays until we manually delete it.
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    # Save the exact file path into a variable so we can find it later
    temp_filename = temp_file.name
    # Close the file system lock so the TTS engine has permission to write audio data into it
    temp_file.close()
    
    # Run the asynchronous save operation synchronously to download the audio from Microsoft's servers
    asyncio.run(communicate.save(temp_filename))
    
    # Load the newly created MP3 file into the pygame audio player
    # Play the audio file locally
    pygame.mixer.music.load(temp_filename)
    # Start playing the audio out loud through the speakers
    pygame.mixer.music.play()
    # Keep the script paused here in a loop as long as the audio is still actively playing.
    # This prevents the microphone from immediately turning back on and accidentally recording its own output!
    while pygame.mixer.music.get_busy():
        # Wait a fraction of a second, then check again if the audio is still playing
        pygame.time.Clock().tick(10)
        
    # Once the audio is completely finished, delete the MP3 file to keep the hard drive clean
    os.remove(temp_filename)

# --- MAIN AUTONOMOUS LOOP ---
# Define the master function that controls the continuous, hands-free listening cycle
def start_autonomous_translator():
    # Create the microphone listening instance
    recognizer = sr.Recognizer()
    
    # Configure the VAD (Voice Activity Detection) silence threshold.
    # This tells the system to wait for exactly 1.5 seconds of total silence before deciding the user has finished speaking.
    # Settings for detecting silence and background noise
    recognizer.pause_threshold = 1.5  # Seconds of silence before stopping
    # Set a default baseline volume sensitivity. 
    # Any sound below this RMS (Root Mean Square) volume level is considered background noise and is ignored.
    recognizer.energy_threshold = 300 # Sensitivity (adjusts automatically)
    
    # Open the physical microphone hardware to start capturing audio
    with sr.Microphone() as source:
        # Print a terminal message letting the user know the system is active
        print("\n✅ System Ready. Press Ctrl+C to exit.")
        # Print a terminal message indicating the background noise calibration phase
        print("🤫 Profiling background noise for 2 seconds...")
        # This is the crucial RMS (Root Mean Square) noise profiling step!
        # The system listens to the room for 2 seconds and mathematically calculates the average ambient volume.
        # It uses RMS to turn oscillating sound waves into a positive energy value, defining the "noise floor".
        # It automatically updates the 'energy_threshold' to be slightly above this floor, ensuring true hands-free operation.
        recognizer.adjust_for_ambient_noise(source, duration=2)
        
        # Start an infinite loop to keep the program running forever (until the user manually stops it)
        while True:
            # Prompt the user to start talking
            print("\n🎙️ Listening... (Speak now!)")
            try:
                # This is a "blocking call". The program pauses on this line and waits.
                # It only starts recording when audio spikes above the RMS energy threshold.
                # It automatically stops recording once audio drops below the threshold for 1.5 seconds (pause_threshold).
                # 1. Auto-Record until silence is detected
                audio_data = recognizer.listen(source, timeout=None)
                # Print a status update indicating the recording finished successfully
                print("⏳ Silence detected. Processing...")
                
                # Create a temporary .wav file to store the raw microphone recording
                # Save audio to temp file for Whisper
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
                    # Write the raw audio bytes into the temporary .wav file
                    temp_wav.write(audio_data.get_wav_data())
                    # Save the file path to a variable so we can pass it to Whisper
                    temp_wav_path = temp_wav.name
                
                # Pass the temporary .wav file path to the Whisper model for transcription
                # 2. Transcribe and Auto-Detect Language
                transcription = asr_model.transcribe(temp_wav_path)
                # Extract the transcribed text string and remove any trailing blank spaces
                detected_text = transcription["text"].strip()
                # Extract the language code that Whisper automatically detected (returns 'en' or 'es')
                detected_lang = transcription["language"] # 'en' or 'es'
                
                # Delete the temporary .wav file immediately to free up hard drive space
                os.remove(temp_wav_path)
                
                # If Whisper heard a loud noise but couldn't find actual words, restart the loop
                if not detected_text:
                    continue
                    
                # Dynamically route the translation settings based on the detected language
                # 3. Dynamic Routing
                if detected_lang == "en":
                    # Log that English was detected
                    print(f"🗣️ Detected English: {detected_text}")
                    # Set the source language code, target language code, and target audio language variables for EN->ES
                    src_code, tgt_code, target_lang_simple = "en_XX", "es_XX", "es"
                elif detected_lang == "es":
                    # Log that Spanish was detected
                    print(f"🗣️ Detected Spanish: {detected_text}")
                    # Set the source language code, target language code, and target audio language variables for ES->EN
                    src_code, tgt_code, target_lang_simple = "es_XX", "en_XX", "en"
                else:
                    # If Whisper detected an unsupported language (like French), print an error and restart the loop
                    print(f"⚠️ Unrecognized language detected: {detected_lang}")
                    continue
                
                # Print a status update indicating the translation phase is starting
                # 4. Translate
                print("⚙️ Translating...")
                # Explicitly tell the mBART tokenizer which language it is currently reading
                tokenizer.src_lang = src_code
                # Convert the transcribed human text into numerical tensor arrays for the AI to process
                inputs = tokenizer(detected_text, return_tensors="pt")
                
                # Use torch.no_grad() to skip calculating gradients, which saves massive amounts of memory and speeds up generation
                with torch.no_grad():
                    # Command the AI model to generate the translation using the custom LoRA conversational skills
                    generated_tokens = model.generate(
                        # Unpack the numerical inputs
                        **inputs,
                        # Force the very first generated word to be the correct target language tag
                        forced_bos_token_id=tokenizer.lang_code_to_id[tgt_code]
                    )
                
                # Convert the AI's output numbers back into human words, hiding any invisible technical tokens
                translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
                # Print the final translated text to the terminal
                print(f"➡️ Translation: {translated_text}")
                
                # Pass the translated text and the target language to our helper function to play the audio
                # 5. Speak
                speak_translation(translated_text, target_lang_simple)
                
            # Catch the specific error triggered when the user presses Ctrl+C in the terminal
            except KeyboardInterrupt:
                # Print a graceful shutdown message and break out of the infinite loop
                print("\n🛑 Shutting down system.")
                break
            # Catch any other unexpected errors so the program doesn't violently crash
            except Exception as e:
                # Print the exact error message and keep the loop running
                print(f"An error occurred: {e}")

# Standard Python check to ensure this code block only runs if the script is executed directly
if __name__ == "__main__":
    # Attempt to import the pygame library
    # Install pygame if not present for local audio playback
    try:
        import pygame
    # If the user doesn't have pygame installed, catch the import error
    except ImportError:
        # Print a message saying we are automatically installing the missing dependency
        print("Installing pygame for audio playback...")
        # Use the operating system command line to install pygame via pip
        os.system("pip install pygame")
        # Import it again now that it is successfully installed
        import pygame
        
    # Call the master function to start the infinite autonomous translation cycle
    start_autonomous_translator()