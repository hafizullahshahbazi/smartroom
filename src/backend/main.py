from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import json
import asyncio
from datetime import datetime

from pipeline import process_audio, LANG_MAP, synthesize_speech, translate_all
from rooms import manager, Room
import base64

app = FastAPI(title="Smart Room Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = "/app/frontend"


def serve_html(filename: str) -> str:
    path = os.path.join(FRONTEND_DIR, filename)
    if not os.path.exists(path):
        return f"<h1>File {filename} not found</h1>"
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/", response_class=HTMLResponse)
async def index():
    return serve_html("index.html")


@app.get("/host", response_class=HTMLResponse)
async def host_page():
    return serve_html("host.html")


@app.get("/join", response_class=HTMLResponse)
async def join_page():
    return serve_html("join.html")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/api/rooms")
async def create_room():
    room = manager.create_room()
    return {"code": room.code, "created_at": room.created_at.isoformat()}


@app.get("/api/rooms/{code}")
async def get_room_info(code: str):
    room = manager.get_room(code)
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")
    return room.get_info()


@app.get("/api/rooms/{code}/history")
async def get_room_history(code: str, lang: str = "en"):
    room = manager.get_room(code)
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")
    return {"code": code, "lang": lang, "history": room.get_history_for_audience(lang)}


@app.get("/api/rooms/{code}/export")
async def export_room_history(code: str, format: str = "txt"):
    room = manager.get_room(code)
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")

    if format == "json":
        content = json.dumps({
            "room_code": room.code,
            "created_at": room.created_at.isoformat(),
            "host_lang": room.host_lang,
            "total_entries": len(room.history),
            "entries": room.history,
            "chat": room.chat_history,
        }, ensure_ascii=False, indent=2)
        return PlainTextResponse(content, media_type="application/json",
                                 headers={"Content-Disposition": f"attachment; filename=room_{code}.json"})

    lines = [f"Smart Room - {room.code}",
             f"Created: {room.created_at.isoformat()}",
             f"Host language: {room.host_lang}",
             f"Total: {len(room.history)} sentences",
             "=" * 60, ""]
    for i, entry in enumerate(room.history, 1):
        ts = entry["timestamp"]
        speaker = entry.get("speaker", "host")
        lines.append(f"[{i}] {ts} ({speaker})")
        lines.append(f"  Original ({entry['source_lang']}): {entry['original_text']}")
        for lang_code, text in entry["translations"].items():
            lang_name = {"en": "English", "fa": "Persian", "ru": "Russian",
                         "zh": "Chinese", "es": "Spanish"}.get(lang_code, lang_code)
            lines.append(f"  {lang_name}: {text}")
        lines.append("")
    content = "\n".join(lines)
    return PlainTextResponse(content, media_type="text/plain; charset=utf-8",
                             headers={"Content-Disposition": f"attachment; filename=room_{code}.txt"})


# ============ WEBSOCKET: HOST ============
@app.websocket("/ws/host/{code}")
async def ws_host(websocket: WebSocket, code: str):
    room = manager.get_room(code)
    if not room:
        await websocket.close(code=4004, reason="Room not found")
        return

    host_lang = websocket.query_params.get("lang", "en")
    if host_lang not in LANG_MAP:
        print(f"[Host] WARNING: '{host_lang}' not in LANG_MAP, defaulting to 'en'")
        host_lang = "en"
    room.host_lang = host_lang

    await websocket.accept()
    room.host_ws = websocket
    print(f"[Host] Connected to room {code} (host_lang={host_lang})")

    try:
        await websocket.send_json({
            "type": "connected",
            "room_code": code,
            "host_lang": host_lang,
            "audiences": len(room.audiences),
            "chat_history": room.get_chat_for_audience(host_lang),
        })

        while True:
            try:
                data = await websocket.receive()
            except RuntimeError:
                break

            if data.get("type") == "websocket.disconnect":
                break

            # Binary = audio
            if "bytes" in data and data["bytes"]:
                audio_bytes = data["bytes"]

                # Check if this is a HOLD audio (chat only) or LIVE mic (transcript)
                if getattr(room, "_pending_hold", False):
                    room._pending_hold = False
                    print(f"[Host] HOLD audio: {len(audio_bytes)} bytes (chat only)")
                    asyncio.create_task(process_hold_audio(room, audio_bytes))
                else:
                    print(f"[Host] LIVE mic audio: {len(audio_bytes)} bytes (transcript)")
                    asyncio.create_task(process_and_broadcast(room, audio_bytes))

            elif "text" in data and data["text"]:
                try:
                    msg = json.loads(data["text"])
                    if msg.get("type") == "ping":
                        await websocket.send_json({"type": "pong"})
                    elif msg.get("type") == "chat":
                        asyncio.create_task(process_chat_message(
                            room, msg.get("text", ""), "Host", room.host_lang, is_host=True
                        ))
                    elif msg.get("type") == "hold_audio_start":
                        room._pending_hold = True
                        print("[Host] Next audio will be treated as HOLD (chat only)")
                except Exception as e:
                    print(f"[Host] Text error: {e}")

    except WebSocketDisconnect:
        print(f"[Host] Disconnected from room {code}")
    finally:
        room.host_ws = None


async def process_and_broadcast(room: Room, audio_bytes: bytes):
    """Process host LIVE mic audio → TRANSCRIPT ONLY (no chat bubble)."""
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, process_audio, audio_bytes, room.host_lang)

        if not result["original"]:
            return

        print(f"[Live] (host_lang={room.host_lang}) {result['original'][:80]}...")
        print(f"[Live] Translations: {list(result['translations'].keys())}")

        entry = room.add_entry(
            original=result["original"],
            source_lang=result["source_lang"],
            translations=result["translations"],
            speaker="host",
        )

        audiences_snapshot = list(room.audiences.items())

        for ws, info in audiences_snapshot:
            try:
                if ws.client_state.name != "CONNECTED":
                    continue

                translated_text = entry["translations"].get(info["lang"], "")
                print(f"[Live Broadcast] -> {info['name']} (lang={info['lang']})")

                # Send TRANSCRIPT ONLY (no chat)
                await ws.send_json({
                    "type": "translation",
                    "timestamp": entry["timestamp"],
                    "original_text": entry["original_text"],
                    "translated_text": translated_text,
                    "lang": info["lang"],
                    "speaker": "host",
                    "audio": "",
                })

                # Send TTS audio
                if translated_text.strip():
                    asyncio.create_task(
                        generate_and_send_audio(ws, translated_text, info["lang"], info["name"])
                    )
            except Exception as e:
                print(f"[Live Broadcast] Failed for {info.get('name', '?')}: {e}")

        if room.host_ws:
            try:
                await room.host_ws.send_json({
                    "type": "processed",
                    "original_text": result["original"],
                    "translations": result["translations"],
                    "speaker": "host",
                    "total_audiences": len(room.audiences),
                })
            except Exception as e:
                print(f"[Live Broadcast] Host notify failed: {e}")

    except Exception as e:
        import traceback
        print(f"[Live] ERROR: {e}")
        traceback.print_exc()


async def process_hold_audio(room: Room, audio_bytes: bytes):
    """Process host HOLD audio → CHAT ONLY (no transcript entry)."""
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, process_audio, audio_bytes, room.host_lang)

        if not result["original"]:
            return

        print(f"[Hold] (host_lang={room.host_lang}) {result['original'][:80]}...")

        # Add to CHAT ONLY (not transcript)
        chat_msg = room.add_chat_message(
            sender_name="Host",
            sender_lang=room.host_lang,
            original_text=f"🎤 {result['original']}",
            translations={k: f"🎤 {v}" for k, v in result["translations"].items()},
        )

        audiences_snapshot = list(room.audiences.items())

        # Broadcast chat bubble to audiences
        for ws, info in audiences_snapshot:
            try:
                if ws.client_state.name != "CONNECTED":
                    continue
                translated = chat_msg["translations"].get(info["lang"], chat_msg["original_text"])
                await ws.send_json({
                    "type": "chat",
                    "id": chat_msg["id"],
                    "timestamp": chat_msg["timestamp"],
                    "sender_name": "Host",
                    "original_text": chat_msg["original_text"],
                    "translated_text": translated,
                    "sender_lang": room.host_lang,
                    "is_mine": False,
                })
            except Exception as e:
                print(f"[Hold] Audience send failed: {e}")

        # Echo back to host
        if room.host_ws:
            try:
                await room.host_ws.send_json({
                    "type": "chat",
                    "id": chat_msg["id"],
                    "timestamp": chat_msg["timestamp"],
                    "sender_name": "Host",
                    "original_text": chat_msg["original_text"],
                    "translated_text": chat_msg["original_text"],
                    "sender_lang": room.host_lang,
                    "is_mine": True,
                })
            except Exception:
                pass

    except Exception as e:
        import traceback
        print(f"[Hold] ERROR: {e}")
        traceback.print_exc()


async def process_audience_audio(room: Room, audio_bytes: bytes, sender_ws, sender_name: str, sender_lang: str):
    """Process audio from an audience member → CHAT ONLY."""
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, process_audio, audio_bytes, sender_lang)

        if not result["original"]:
            print(f"[Audience-Hold] No text from {sender_name}")
            return

        print(f"[Audience-Hold] {sender_name}: {result['original'][:80]}...")

        # Add to CHAT ONLY (not transcript)
        chat_msg = room.add_chat_message(
            sender_name=sender_name,
            sender_lang=sender_lang,
            original_text=f"🎤 {result['original']}",
            translations={k: f"🎤 {v}" for k, v in result["translations"].items()},
        )

        audiences_snapshot = list(room.audiences.items())

        for ws, info in audiences_snapshot:
            try:
                if ws.client_state.name != "CONNECTED":
                    continue
                translated = chat_msg["translations"].get(info["lang"], chat_msg["original_text"])
                await ws.send_json({
                    "type": "chat",
                    "id": chat_msg["id"],
                    "timestamp": chat_msg["timestamp"],
                    "sender_name": sender_name,
                    "original_text": chat_msg["original_text"],
                    "translated_text": translated,
                    "sender_lang": sender_lang,
                    "is_mine": info["name"] == sender_name,
                })
            except Exception:
                pass

        if room.host_ws:
            try:
                await room.host_ws.send_json({
                    "type": "chat",
                    "id": chat_msg["id"],
                    "timestamp": chat_msg["timestamp"],
                    "sender_name": sender_name,
                    "original_text": chat_msg["original_text"],
                    "translated_text": chat_msg["translations"].get(room.host_lang, chat_msg["original_text"]),
                    "sender_lang": sender_lang,
                    "is_mine": False,
                })
            except Exception:
                pass

    except Exception as e:
        import traceback
        print(f"[Audience-Hold] ERROR: {e}")
        traceback.print_exc()


async def generate_and_send_audio(ws: WebSocket, text: str, lang: str, name: str):
    """Generate TTS audio and send to a specific audience."""
    try:
        loop = asyncio.get_event_loop()
        audio_bytes_tts = await loop.run_in_executor(None, synthesize_speech, text, lang)

        if not audio_bytes_tts:
            return

        audio_b64 = base64.b64encode(audio_bytes_tts).decode('ascii')
        print(f"[TTS] Generated {len(audio_bytes_tts)} bytes for {lang} -> {name}")

        if ws.client_state.name == "CONNECTED":
            await ws.send_json({
                "type": "audio",
                "lang": lang,
                "audio": audio_b64,
            })
    except Exception as e:
        print(f"[TTS] Send failed for {name}: {e}")


# ============ CHAT MESSAGE HANDLER (TEXT) ============
async def process_chat_message(room: Room, text: str, sender_name: str, sender_lang: str, is_host: bool = False):
    """Process a TEXT chat message: translate, store, broadcast."""
    if not text.strip():
        return

    print(f"[Chat] {sender_name} ({sender_lang}): {text[:60]}")

    try:
        loop = asyncio.get_event_loop()
        src_nllb = LANG_MAP.get(sender_lang, LANG_MAP["en"])["nllb"]
        translations = await loop.run_in_executor(None, translate_all, text, src_nllb)
        translations[sender_lang] = text

        message = room.add_chat_message(
            sender_name=sender_name,
            sender_lang=sender_lang,
            original_text=text,
            translations=translations,
        )

        print(f"[Chat] Translations: {list(translations.keys())}")

        if not is_host and room.host_ws:
            try:
                await room.host_ws.send_json({
                    "type": "chat",
                    "id": message["id"],
                    "timestamp": message["timestamp"],
                    "sender_name": sender_name,
                    "original_text": text,
                    "translated_text": translations.get(room.host_lang, text),
                    "sender_lang": sender_lang,
                    "is_mine": False,
                })
            except Exception as e:
                print(f"[Chat] Host send failed: {e}")

        for ws, info in list(room.audiences.items()):
            try:
                translated = translations.get(info["lang"], text)
                await ws.send_json({
                    "type": "chat",
                    "id": message["id"],
                    "timestamp": message["timestamp"],
                    "sender_name": sender_name,
                    "original_text": text,
                    "translated_text": translated,
                    "sender_lang": sender_lang,
                    "is_mine": info["name"] == sender_name,
                })
            except Exception as e:
                print(f"[Chat] Audience send failed: {e}")

        if is_host and room.host_ws:
            try:
                await room.host_ws.send_json({
                    "type": "chat",
                    "id": message["id"],
                    "timestamp": message["timestamp"],
                    "sender_name": sender_name,
                    "original_text": text,
                    "translated_text": text,
                    "sender_lang": sender_lang,
                    "is_mine": True,
                })
            except Exception:
                pass

    except Exception as e:
        import traceback
        print(f"[Chat] ERROR: {e}")
        traceback.print_exc()


# ============ WEBSOCKET: AUDIENCE ============
@app.websocket("/ws/audience/{code}")
async def ws_audience(websocket: WebSocket, code: str):
    room = manager.get_room(code)
    if not room:
        await websocket.close(code=4004, reason="Room not found")
        return

    lang = websocket.query_params.get("lang", "en")
    name = websocket.query_params.get("name", "Anonymous")

    print(f"[Audience] Join attempt: name={name} lang={lang} room={code}")

    if lang not in LANG_MAP:
        print(f"[Audience] WARNING: '{lang}' not in LANG_MAP, defaulting to 'en'")
        lang = "en"

    await websocket.accept()
    room.audiences[websocket] = {"lang": lang, "name": name}
    print(f"[Audience] {name} joined room {code} (lang={lang})")

    try:
        history = room.get_history_for_audience(lang)
        chat_history = room.get_chat_for_audience(lang)

        await websocket.send_json({
            "type": "connected",
            "room_code": code,
            "lang": lang,
            "name": name,
            "history": history,
            "chat_history": chat_history,
            "total_audiences": len(room.audiences),
        })

        if room.host_ws:
            try:
                await room.host_ws.send_json({
                    "type": "audience_joined",
                    "name": name,
                    "lang": lang,
                    "total_audiences": len(room.audiences),
                })
            except Exception:
                pass

        while True:
            try:
                data = await websocket.receive()
            except RuntimeError:
                break

            if data.get("type") == "websocket.disconnect":
                break

            if "bytes" in data and data["bytes"]:
                audio_bytes = data["bytes"]
                print(f"[Audience] {name} sent {len(audio_bytes)} bytes audio (HOLD → chat)")
                asyncio.create_task(process_audience_audio(
                    room, audio_bytes, websocket, name, lang
                ))

            elif "text" in data and data["text"]:
                try:
                    msg = json.loads(data["text"])
                    if msg.get("type") == "ping":
                        await websocket.send_json({"type": "pong"})
                    elif msg.get("type") == "chat":
                        asyncio.create_task(process_chat_message(
                            room, msg.get("text", ""), name, lang, is_host=False
                        ))
                except Exception as e:
                    print(f"[Audience] Text error: {e}")

    except WebSocketDisconnect:
        print(f"[Audience] {name} disconnected")
    finally:
        room.audiences.pop(websocket, None)
        print(f"[Audience] {name} left room {code}. Remaining: {len(room.audiences)}")

        if room.host_ws:
            try:
                await room.host_ws.send_json({
                    "type": "audience_left",
                    "name": name,
                    "total_audiences": len(room.audiences),
                })
            except Exception:
                pass


@app.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    source_lang: str = Form("ru"),
):
    content = await audio.read()
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, process_audio, content, source_lang)
    return result