**SunoSaathi – Speech to Indian Sign Language (ISL) Translation System**
SunoSaathi is an end-to-end applied AI prototype that converts spoken English speech into a sequential Indian Sign Language (ISL) visual output, aimed at improving accessibility for deaf and hard-of-hearing users.
The project focuses on audio processing, NLP-based semantic filtering, and system integration, rather than training new ML models.

🚀 **Why This Project Matters**
- Demonstrates end-to-end system ownership
- Handles real-time audio, NLP, and backend inference
- Shows engineering tradeoffs and limitations clearly
- Built around a real-world accessibility problem
  
🧠 **Pipeline**

[![Pipeline](pipeline.svg)](pipeline.svg)


🏗 **Architecture Overview**
**Frontend**
- Captures live microphone input via browser APIs
- Visualizes audio amplitude using FFT-based analysis
- Displays interim speech captions
- Renders ISL signs as timed visual sequences
**Backend
**
- FastAPI-based inference server
- Whisper ASR for speech transcription
- Text normalization and filler word removal
- spaCy-based lemmatization and keyword extraction
- Rule-based English → ISL gloss mapping

🛠 **Tech Stack**
**Frontend**
- HTML, CSS, JavaScript
- Web Audio API
- MediaRecorder API
**Backend**
- Python
- FastAPI
- OpenAI Whisper (base model)
- spaCy (en_core_web_sm)
- Regex-based text preprocessing

**Key Engineering Features**
- 🎙 Real-time audio capture and analysis
- 📊 Audio amplitude visualization using FFT
- 📝 Live interim speech captions
- 🧹 Linguistic normalization and filler-word removal
- 🧠 Lemma-based semantic keyword extraction
- 🤟 English-to-ISL gloss sequencing
- 🎞 Controlled temporal playback of ISL signs

🧩 **Design Decisions & Tradeoffs**
- **Rule-based ISL mapping**: Chosen for interpretability and controllability in a prototype setting.
- **Whisper** (base model): Balanced transcription accuracy and inference speed.
- **Non-streaming inference**: Implemented using HTTP POST to maintain system simplicity and reliability.
- **Keyword-based translation**: Reflects early-stage sign translation systems focusing on semantic compression.

⚠️ **Known Limitations**
- ISL grammar is not explicitly modelled
- A very limited sign vocabulary defined manually, with synthetically generated png images using Gemini Banana Pro
- No continuous streaming transcription
- Static sign images instead of pose-based animations
- Performance degrades in noisy environments

🚀 **How to Run Locally**
**Backend**
```bash
pip install fastapi uvicorn openai-whisper spacy
python -m spacy download en_core_web_sm
uvicorn main:app --reload
```
**Frontend**
- Open index.html in a browser
- Allow microphone access
- Click Start, speak, then Stop

🔮 **Future Improvements**
- Grammar-aware ISL translation
- Larger and dynamic ISL vocabulary
- Streaming transcription using WebSockets
- Pose-based sign animation
- Noise-robust audio preprocessing

👩‍💻 **Author**
Ishika Walia
