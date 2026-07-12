import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from peft import PeftModel
import whisper
import edge_tts
import asyncio
import tempfile
import speech_recognition as sr
import os
import pygame  # Used to play the audio locally

print("⚙️ Booting Autonomous AI Translator...")

# 1. Load Models
print("🎧 Loading Whisper (Ears)...")
asr_model = whisper.load_model("base")

print("🧠 Loading mBART (Brain)...")
MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"
tokenizer = MBart50TokenizerFast.from_pretrained(MODEL_NAME)
base_model = MBartForConditionalGeneration.from_pretrained(MODEL_NAME)
model = PeftModel.from_pretrained(base_model, "./saved_lora_adapter")

# Initialize audio player
pygame.mixer.init()

# --- HELPER: TEXT TO SPEECH & PLAYBACK ---
def speak_translation(text, target_language):
    voice_model = "es-ES-AlvaroNeural" if target_language == "es" else "en-US-ChristopherNeural"
    
    communicate = edge_tts.Communicate(text, voice_model)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    temp_filename = temp_file.name
    temp_file.close()
    
    asyncio.run(communicate.save(temp_filename))
    
    # Play the audio file locally
    pygame.mixer.music.load(temp_filename)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
        
    os.remove(temp_filename)

# --- MAIN AUTONOMOUS LOOP ---
def start_autonomous_translator():
    recognizer = sr.Recognizer()
    
    # Settings for detecting silence and background noise
    recognizer.pause_threshold = 1.5  # Seconds of silence before stopping
    recognizer.energy_threshold = 300 # Sensitivity (adjusts automatically)
    
    with sr.Microphone() as source:
        print("\n✅ System Ready. Press Ctrl+C to exit.")
        print("🤫 Profiling background noise for 2 seconds...")
        recognizer.adjust_for_ambient_noise(source, duration=2)
        
        while True:
            print("\n🎙️ Listening... (Speak now!)")
            try:
                # 1. Auto-Record until silence is detected
                audio_data = recognizer.listen(source, timeout=None)
                print("⏳ Silence detected. Processing...")
                
                # Save audio to temp file for Whisper
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
                    temp_wav.write(audio_data.get_wav_data())
                    temp_wav_path = temp_wav.name
                
                # 2. Transcribe and Auto-Detect Language
                transcription = asr_model.transcribe(temp_wav_path)
                detected_text = transcription["text"].strip()
                detected_lang = transcription["language"] # 'en' or 'es'
                
                os.remove(temp_wav_path)
                
                if not detected_text:
                    continue
                    
                # 3. Dynamic Routing
                if detected_lang == "en":
                    print(f"🗣️ Detected English: {detected_text}")
                    src_code, tgt_code, target_lang_simple = "en_XX", "es_XX", "es"
                elif detected_lang == "es":
                    print(f"🗣️ Detected Spanish: {detected_text}")
                    src_code, tgt_code, target_lang_simple = "es_XX", "en_XX", "en"
                else:
                    print(f"⚠️ Unrecognized language detected: {detected_lang}")
                    continue
                
                # 4. Translate
                print("⚙️ Translating...")
                tokenizer.src_lang = src_code
                inputs = tokenizer(detected_text, return_tensors="pt")
                
                with torch.no_grad():
                    generated_tokens = model.generate(
                        **inputs,
                        forced_bos_token_id=tokenizer.lang_code_to_id[tgt_code]
                    )
                
                translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
                print(f"➡️ Translation: {translated_text}")
                
                # 5. Speak
                speak_translation(translated_text, target_lang_simple)
                
            except KeyboardInterrupt:
                print("\n🛑 Shutting down system.")
                break
            except Exception as e:
                print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Install pygame if not present for local audio playback
    try:
        import pygame
    except ImportError:
        print("Installing pygame for audio playback...")
        os.system("pip install pygame")
        import pygame
        
    start_autonomous_translator()