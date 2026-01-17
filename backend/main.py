from fastapi import FastAPI, UploadFile, File
import whisper
import spacy
import re
import tempfile
import os
last_text = ""

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # for demo only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


whisper_model = whisper.load_model("base")
nlp = spacy.load("en_core_web_sm")

SIGN_DICT = {
    "me": "me",
    "you":"you",
    "your":"you",
    "doctor": "doctor",
    "hospital": "hospital",
    "help": "help",
    "water": "water",
    "meet": "meet",
    "need": "need",
    "thank": "thankyou",
    "right": "right",
    "left":"left",
    "stop":"stop",
    "go":"go",
    "bus":"bus",
    "train":"train",
    "food":"food",
    "home":"home",
    "lawyer":"lawyer",
    "toilet":"toilet",
    "washroom":"toilet",
    "house":"home"
}

IMPORTANT_WORDS = set(SIGN_DICT.keys())

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()

@app.post("/translate")
async def translate(file: UploadFile = File(...)):   
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        result = whisper_model.transcribe(tmp_path)
        raw_text = result["text"]
        clean = clean_text(raw_text)


        keywords = []
        if "i" in clean or "me" in clean:
            keywords.append("me")

        doc = nlp(clean)
        for token in doc:
            lemma = token.lemma_
            if lemma in IMPORTANT_WORDS and lemma not in keywords:
                keywords.append(lemma)


        return {
            "raw_text": raw_text,
            "clean_text": clean,
            "keywords": keywords,
            "sequence": [SIGN_DICT[k] for k in keywords]
        }

    finally:
        os.remove(tmp_path)
