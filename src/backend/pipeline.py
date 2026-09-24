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

# Global lock for NLLB (not thread-safe)
_nllb_lock = threading.Lock()

# ============ LOAD MODELS ============
print("[Pipeline] Loading NLLB model...")
NLLB_PATH = snapshot_download("olob0/nllb-200-distilled-600M-ct2-int8_float16")
tokenizer = AutoTokenizer.from_pretrained(NLLB_PATH)
translator = ctranslate2.Translator(
    NLLB_PATH,
    device="cpu",
    compute_type="int8"
)
print("[Pipeline] NLLB loaded.")


# ============ LANGUAGE MAPPING (5 languages) ============
LANG_MAP = {
    "en": {"whisper": "en", "nllb": "eng_Latn", "name": "English"},
    "fa": {"whisper": "fa", "nllb": "pes_Arab", "name": "Persian (Farsi)"},
    "ru": {"whisper": "ru", "nllb": "rus_Cyrl", "name": "Russian"},
    "zh": {"whisper": "zh", "nllb": "zho_Hans", "name": "Chinese"},
    "es": {"whisper": "es", "nllb": "spa_Latn", "name": "Spanish"},
}

TARGET_LANGS = {
    "en": "eng_Latn",
    "fa": "pes_Arab",
    "ru": "rus_Cyrl",
    "zh": "zho_Hans",
    "es": "spa_Latn",
}

WHISPER_URL = os.getenv("WHISPER_URL", "http://whisper:9000/asr")


# ============ TEXT CLEANUP HELPERS ============
def _dedupe_repetitions(text: str) -> str:
    """Remove excessive word/phrase repetitions from Whisper output.
    Example: 'basket basket basket' -> 'basket'
             'word word word word' -> 'word'
    """
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
            bigram = (result_words[i].lower(), result_words[i+1].lower())
            # Count how many times this bigram repeats
            run = 0
            j = i
            while j < len(result_words) - 1:
                if (result_words[j].lower(), result_words[j+1].lower()) == bigram:
                    run += 1
                    j += 2
                else:
                    break
            if run >= 3:
                final.extend(result_words[i:i+2])
                i = j
            else:
                final.append(result_words[i])
                i += 1
        if i == len(result_words) - 1:
            final.append(result_words[i])
        result = " ".join(final)

    return result


# ============ TRANSLATE ============
def translate(text: str, src_lang: str, tgt_lang: str) -> str:
    """Translate text using NLLB, split per sentence for accuracy."""
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


def translate_all(text: str, src_nllb_code: str) -> dict:
    """Translate text to all target languages."""
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
def transcribe_audio(audio_bytes: bytes, source_lang: str = "ru") -> str:
    """Send audio to Whisper, get original text."""
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
def process_audio(audio_bytes: bytes, source_lang: str = "ru") -> dict:
    """Full pipeline: audio -> Whisper -> NLLB -> translations."""
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
VOICE_MAP = {
    "en": "en-US-AriaNeural",
    "ru": "ru-RU-SvetlanaNeural",
    "fa": "fa-IR-DilaraNeural",
    "zh": "zh-CN-XiaoxiaoNeural",
    "es": "es-ES-ElviraNeural",
}


async def synthesize_speech_async(text: str, lang: str) -> bytes:
    """Convert text to speech audio (MP3 bytes) using Edge-TTS."""
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


def synthesize_speech(text: str, lang: str) -> bytes:
    """Synchronous wrapper for synthesize_speech_async."""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(synthesize_speech_async(text, lang))
        loop.close()
        return result
    except Exception as e:
        print(f"[TTS] Sync wrapper error: {e}")
        return b""