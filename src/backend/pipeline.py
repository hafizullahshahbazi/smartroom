##
# @mainpage SmartRoom — Interactive Real-Time Room Translation System
#
# @section intro Introduction
# SmartRoom is a web-based platform that enables people speaking different
# native languages to communicate in real time inside shared virtual rooms.
# Participants speak or type in their own language, and every other
# participant sees and hears the message in their own language.
#
# @section features Features
# - **5 languages supported:** English, Persian, Russian, Chinese, Spanish
# - **Live microphone mode** with continuous speech recognition
# - **Push-to-talk** for voice messages as chat bubbles
# - **Real-time text chat** with automatic multilingual translation
# - **Voice-to-voice translation** using neural TTS
# - **Dark / Light theme** toggle
# - **Transcript export** in TXT and JSON formats
#
# @section architecture Architecture
# The system consists of three layers:
# - **Frontend:** HTML, CSS, JavaScript (WebSocket + MediaRecorder)
# - **Backend:** FastAPI (REST + WebSocket server)
# - **ML Pipeline:** Whisper (STT) → NLLB-200 (Translation) → Edge-TTS (Voice)
#
# @section techstack Technology Stack
# - Python 3.11, FastAPI, WebSocket, Uvicorn
# - OpenAI Whisper, Meta NLLB-200 via CTranslate2
# - Microsoft Edge-TTS (5 neural voices)
# - Docker, Docker Compose
#
# @section team Development Team
# Built by an 8-person team as part of Laboratory Work 1 and 2 in
# Software Engineering (ТРПО).
#
# @section links Related Pages
# - @ref Entity (base class)
# - @ref Audience (derived class)
# - @ref Room (session class)
# - @ref RoomManager (manager class)
#
# @author Hafizullah Shahbazi
# @date 2026-10-01
# @version 1.0.0
"""
@file pipeline.py
@brief SmartRoom ML pipeline — Whisper, NLLB-200, Edge-TTS integration
@details This module handles the complete translation pipeline:
         1. Speech-to-Text via Whisper
         2. Translation via NLLB-200 (CTranslate2 backend)
         3. Text-to-Speech via Microsoft Edge-TTS
@author Hafizullah Shahbazi
@date 2026-09-24
@version 1.0.0
@copyright MIT License
@see main.py
@see rooms.py
"""
import re
import tempfile
import os
import asyncio
import threading
import requests
import edge_tts
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
import ctranslate2


##
# @brief Global lock for NLLB model
# @details CTranslate2 translator is not thread-safe. All calls to
#          translate_batch() must be wrapped in this lock to prevent
#          concurrent access from multiple asyncio tasks.
_nllb_lock = threading.Lock()


# ============ LOAD MODELS ============
print("[Pipeline] Loading NLLB model...")

##
# @brief Local path to the downloaded NLLB-200 model
# @details Downloaded from HuggingFace using snapshot_download.
#          Model: olob0/nllb-200-distilled-600M-ct2-int8_float16
NLLB_PATH = snapshot_download("olob0/nllb-200-distilled-600M-ct2-int8_float16")

##
# @brief NLLB tokenizer
# @details Loaded from the NLLB_PATH directory. Used for encoding
#          input text and decoding translated tokens.
tokenizer = AutoTokenizer.from_pretrained(NLLB_PATH)

##
# @brief CTranslate2 NLLB translator
# @details Runs on CPU with int8 quantization for low memory usage.
#          Supports 200+ languages via the NLLB-200 model.
translator = ctranslate2.Translator(
    NLLB_PATH,
    device="cpu",
    compute_type="int8"
)
print("[Pipeline] NLLB loaded.")


# ============ LANGUAGE MAPPING ============

##
# @brief Language configuration map
# @details Maps short language codes (en, fa, ru, zh, es) to their
#          respective Whisper/NLLB codes and display names.
#          - "whisper": code used by the Whisper ASR service
#          - "nllb": code used by NLLB-200 translation model
#          - "name": human-readable name for UI display
LANG_MAP = {
    "en": {"whisper": "en", "nllb": "eng_Latn", "name": "English"},
    "fa": {"whisper": "fa", "nllb": "pes_Arab", "name": "Persian (Farsi)"},
    "ru": {"whisper": "ru", "nllb": "rus_Cyrl", "name": "Russian"},
    "zh": {"whisper": "zh", "nllb": "zho_Hans", "name": "Chinese"},
    "es": {"whisper": "es", "nllb": "spa_Latn", "name": "Spanish"},
}

##
# @brief Target languages for every translation
# @details Maps short language codes to their NLLB target codes.
#          Every translated message is rendered into all these
#          languages, so any participant can read it.
TARGET_LANGS = {
    "en": "eng_Latn",
    "fa": "pes_Arab",
    "ru": "rus_Cyrl",
    "zh": "zho_Hans",
    "es": "spa_Latn",
}

##
# @brief URL of the Whisper ASR service
# @details Default: http://whisper:9000/asr. Can be overridden by
#          the WHISPER_URL environment variable.
WHISPER_URL = os.getenv("WHISPER_URL", "http://whisper:9000/asr")


# ============ TEXT CLEANUP HELPERS ============

##
# @brief Remove excessive word/phrase repetitions from Whisper output
# @details Whisper sometimes produces "loops" like
#          "basket basket basket basket" when audio is unclear or silent.
#          This function detects runs of 3+ identical words and collapses
#          them to a single instance. It also detects repeated bigrams
#          (2-word sequences).
# @param text Raw text from Whisper
# @return Cleaned text with repetitions removed
# @note If text is shorter than 4 words, it is returned unchanged
def _dedupe_repetitions(text: str) -> str:
    if not text or len(text) < 5:
        return text

    words = text.split()
    if len(words) < 4:
        return text

    # Check if the same word repeats 3+ times consecutively
    cleaned = []
    i = 0
    while i < len(words):
        # Find run length of identical consecutive words
        run_start = i
        while i < len(words) and words[i].lower() == words[run_start].lower():
            i += 1
        run_len = i - run_start

        if run_len >= 3:
            # Keep only 1 instance of a word that repeats 3+ times
            cleaned.append(words[run_start])
        else:
            cleaned.extend(words[run_start:i])

    result = " ".join(cleaned)

    # Also collapse repeated 2-word phrases (bigram dedupe)
    result_words = result.split()
    if len(result_words) >= 6:
        final = []
        i = 0
        while i < len(result_words) - 1:
            bigram = (result_words[i].lower(), result_words[i + 1].lower())
            # Count how many times this bigram repeats
            run = 0
            j = i
            while j < len(result_words) - 1:
                if (result_words[j].lower(), result_words[j + 1].lower()) == bigram:
                    run += 1
                    j += 2
                else:
                    break
            if run >= 3:
                final.extend(result_words[i:i + 2])
                i = j
            else:
                final.append(result_words[i])
                i += 1
        if i == len(result_words) - 1:
            final.append(result_words[i])
        result = " ".join(final)

    return result


# ============ TRANSLATE ============

##
# @brief Translate a text using NLLB-200 (CTranslate2 backend)
# @details Splits the input into sentences based on punctuation,
#          then translates each sentence individually. Uses a global
#          thread lock (_nllb_lock) because CTranslate2 is not
#          thread-safe. Deduplicates identical sentences and
#          post-processes the result with _dedupe_repetitions().
# @param text Text to translate (may contain multiple sentences)
# @param src_lang NLLB source language code (e.g., "eng_Latn")
# @param tgt_lang NLLB target language code (e.g., "pes_Arab")
# @return Translated text as a single joined string
# @note Uses _nllb_lock to ensure thread safety
# @see _dedupe_repetitions
def translate(text: str, src_lang: str, tgt_lang: str) -> str:
    if not text.strip():
        return ""

    sentences = re.split(r'(?<=[.!?。！？])\s+', text.strip())
    results = []
    seen = set()

    with _nllb_lock:
        tokenizer.src_lang = src_lang
        for sentence in sentences:
            if not sentence.strip():
                continue
            # Skip exact duplicate sentences
            if sentence.strip().lower() in seen:
                continue
            seen.add(sentence.strip().lower())

            tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence).input_ids)
            result = translator.translate_batch(
                [tokens],
                target_prefix=[[tgt_lang]],
                max_batch_size=1
            )
            output_tokens = result[0].hypotheses[0][1:]
            translated = tokenizer.decode(tokenizer.convert_tokens_to_ids(output_tokens))
            translated = _dedupe_repetitions(translated)
            results.append(translated)

    return " ".join(results)


##
# @brief Translate text into all target languages
# @details Iterates over TARGET_LANGS and translates the input text
#          into each target language using NLLB. If the source language
#          equals a target language, the original text is kept unchanged.
#          Errors for individual languages are caught and logged.
# @param text Input text in the source language
# @param src_nllb_code NLLB code of the source language
# @return Dictionary mapping short language codes to translated text
# @return Example: {"en": "Hello", "fa": "سلام", "ru": "Привет"}
# @see TARGET_LANGS
# @see translate
def translate_all(text: str, src_nllb_code: str) -> dict:
    translations = {}
    for lang_code, nllb_code in TARGET_LANGS.items():
        if nllb_code == src_nllb_code:
            translations[lang_code] = text
        else:
            try:
                translations[lang_code] = translate(text, src_nllb_code, nllb_code)
            except Exception as e:
                print(f"[Translate] ERROR for {lang_code}: {e}")
                translations[lang_code] = ""
    return translations


# ============ WHISPER ============

##
# @brief Send audio to the Whisper ASR service and get the transcribed text
# @details Writes the audio to a temporary file, sends it via HTTP POST
#          to the Whisper service, then deletes the temporary file.
#          Applies anti-hallucination parameters (temperature=0.0,
#          condition_on_previous_text=false) and post-processes the
#          result with _dedupe_repetitions().
# @param audio_bytes Raw audio file content (WebM format)
# @param source_lang Whisper language code (default: "ru")
# @return Transcribed text (deduplicated)
# @throw requests.HTTPError If the Whisper service returns a non-200 status
# @see _dedupe_repetitions
# @see WHISPER_URL
def transcribe_audio(audio_bytes: bytes, source_lang: str = "ru") -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        params = {
            "task": "transcribe",
            "language": source_lang,
            "output": "json",
            # Anti-hallucination params
            "temperature": "0.0",           # no random sampling
            "condition_on_previous_text": "false",  # don't loop on prior text
            "no_speech_threshold": "0.6",    # skip silent chunks
            "compression_ratio_threshold": "2.2",  # detect loops
        }
        with open(tmp_path, "rb") as f:
            files = {"audio_file": f}
            r = requests.post(WHISPER_URL, params=params, files=files, timeout=120)
        r.raise_for_status()
        result = r.json()
        text = result.get("text", "").strip()
        # Post-process to remove repetition
        text = _dedupe_repetitions(text)
        return text
    finally:
        os.unlink(tmp_path)


# ============ FULL PIPELINE ============

##
# @brief Complete pipeline: audio → Whisper → NLLB → translations
# @details Main entry point of the ML pipeline. Runs two stages
#          sequentially:
#          1. transcribe_audio() — convert audio to text (Whisper)
#          2. translate_all() — translate text to all targets (NLLB)
#          Both stages are synchronous and are usually called from
#          a thread pool by the FastAPI backend.
# @param audio_bytes Raw audio file content (WebM)
# @param source_lang Source language code (default: "ru")
# @return Dictionary with keys:
#         - "original": transcribed text
#         - "source_lang": the input language
#         - "translations": dict of {lang_code: translated_text}
# @see transcribe_audio
# @see translate_all
def process_audio(audio_bytes: bytes, source_lang: str = "ru") -> dict:
    original_text = transcribe_audio(audio_bytes, source_lang)

    if not original_text:
        return {"original": "", "translations": {}}

    src_nllb = LANG_MAP.get(source_lang, LANG_MAP["en"])["nllb"]
    translations = translate_all(original_text, src_nllb)

    return {
        "original": original_text,
        "source_lang": source_lang,
        "translations": translations,
    }


# ============ TEXT-TO-SPEECH (Edge-TTS) ============

##
# @brief Edge-TTS neural voice ID per language
# @details Maps short language codes (en, fa, ru, zh, es) to
#          Microsoft Edge neural voice identifiers.
VOICE_MAP = {
    "en": "en-US-AriaNeural",
    "ru": "ru-RU-SvetlanaNeural",
    "fa": "fa-IR-DilaraNeural",
    "zh": "zh-CN-XiaoxiaoNeural",
    "es": "es-ES-ElviraNeural",
}


##
# @brief Convert text to speech (async, MP3 bytes) using Edge-TTS
# @details Selects a neural voice from VOICE_MAP, opens a stream to
#          the Microsoft Edge TTS service, and concatenates all
#          received audio chunks into a single MP3 byte array.
# @param text Text to synthesize into speech
# @param lang Target language code (en, fa, ru, zh, es)
# @return MP3 audio bytes, or empty bytes on failure
# @see VOICE_MAP
async def synthesize_speech_async(text: str, lang: str) -> bytes:
    if not text or not text.strip():
        return b""

    voice = VOICE_MAP.get(lang, "en-US-AriaNeural")

    try:
        communicate = edge_tts.Communicate(text, voice)
        audio_data = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data += chunk["data"]
        return audio_data
    except Exception as e:
        print(f"[TTS] Error generating speech for lang={lang}: {e}")
        return b""


##
# @brief Synchronous wrapper around synthesize_speech_async()
# @details Creates a new asyncio event loop, runs the async synthesis
#          to completion, then closes the loop. Used when called from
#          a thread pool via run_in_executor().
# @param text Text to synthesize
# @param lang Target language code
# @return MP3 audio bytes, or empty bytes on failure
# @see synthesize_speech_async
def synthesize_speech(text: str, lang: str) -> bytes:
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(synthesize_speech_async(text, lang))
        loop.close()
        return result
    except Exception as e:
        print(f"[TTS] Sync wrapper error: {e}")
        return b""