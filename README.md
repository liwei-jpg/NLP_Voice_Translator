# 🎙️ Bidirectional Voice-to-Voice AI Translator

An end-to-end, multilingual Natural Language Processing (NLP) pipeline that provides real-time, bidirectional voice translation between English and Spanish. 

This system features an AI brain fine-tuned for casual, conversational speech and includes both a manual Push-to-Talk Web UI and an Autonomous Hands-Free Desktop application.

## 🚀 Features
* **Neural Machine Translation:** Powered by `facebook/mbart-large-50-many-to-many-mmt`.
* **Conversational Fine-Tuning:** Uses Parameter-Efficient Fine-Tuning (PEFT/LoRA) trained on 5,000 conversational movie subtitles to prioritize natural speech over rigid textbook translations.
* **Smart Language Routing:** Automatically detects whether the user is speaking English or Spanish and dynamically routes the translation to the opposite language.
* **Dual Interfaces:**
  * **Web App (Gradio):** A sleek, browser-based UI for manual push-to-talk translation.
  * **Native Desktop App (Tkinter):** A multi-threaded, autonomous application that utilizes Voice Activity Detection (VAD) to listen, detect silence, translate, and speak entirely hands-free.

## 🛠️ Technology Stack
* **Language/Framework:** Python 3.10+, PyTorch (CUDA optimized)
* **ASR (Speech-to-Text):** OpenAI Whisper (`small` model)
* **NMT (Translation):** mBART-50 + Custom LoRA Adapter
* **TTS (Text-to-Speech):** Edge-TTS (Neural Voices)
* **UI/Hardware Audio:** Gradio, Tkinter, SpeechRecognition, PyAudio, Pygame

## ⚙️ Installation & Setup

**1. Clone the repository and navigate to the directory:**
```bash
git clone [https://github.com/YOUR_USERNAME/NLP_Voice_Translator.git](https://github.com/YOUR_USERNAME/NLP_Voice_Translator.git)
cd NLP_Voice_Translator

**2. Create and activate a virtual environment:**
python -m venv venv
# On Windows:
venv\Scripts\activate

**3. Install dependencies:**
pip install -r requirements.txt


**4. Important Windows Dependencies:**
PyTorch (CUDA): To run the translation quickly, ensure you have the NVIDIA-optimized version of PyTorch installed:
pip install torch --index-url https://download.pytorch.org/whl/cu121

FFmpeg: Whisper requires FFmpeg to process audio. On Windows, install it via:
winget install ffmpeg


## How to Run
Option 1: Web Interface (Gradio)
Provides a clear, interactive web layout.
    python app_ui.py

Option 2: Autonomous Desktop Interface (Tkinter)
Provides a hands-free, auto-detecting translation experience.
    python desktop_app.py
Wait for the "Listening..." prompt, then simply speak out loud. The system will auto-detect silence and reply.
