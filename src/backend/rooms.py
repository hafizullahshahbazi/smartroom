import json
import os
import time
import random
import string
from datetime import datetime
from typing import Optional

# Folder untuk simpan history
HISTORY_DIR = "/app/output/rooms"
os.makedirs(HISTORY_DIR, exist_ok=True)


def generate_room_code(length: int = 4) -> str:
    """Generate kode room unik 4 karakter (huruf+angka)."""
    chars = string.ascii_lowercase + string.digits
    return "".join(random.choices(chars, k=length))


class Room:
    def __init__(self, code: str):
        self.code = code
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        self.host_ws = None            # WebSocket host
        self.host_lang = "en"          # Host's speaking language (set when host connects)
        self.audiences = {}            # {websocket: {"lang": str, "name": str}}
        self.history = []              # list of transcript entries
        self.chat_history = []         # list of chat messages
        self.file_path = os.path.join(HISTORY_DIR, f"{code}.json")
        self.last_flush = time.time()
        self.flush_interval = 30       # write to file every 30 seconds
        self._pending_hold = False     # flag: next host audio is HOLD (chat only)

    def add_entry(self, original: str, source_lang: str, translations: dict, speaker: str = "host"):
        """Tambah kalimat baru ke transcript history."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "speaker": speaker,
            "original_text": original,
            "source_lang": source_lang,
            "translations": translations,
        }
        self.history.append(entry)
        self.last_activity = datetime.now()

        if time.time() - self.last_flush > self.flush_interval:
            self.save_to_file()
            self.last_flush = time.time()

        return entry

    def add_chat_message(self, sender_name: str, sender_lang: str, original_text: str, translations: dict):
        """Add a chat message to chat history."""
        message = {
            "id": len(self.chat_history) + 1,
            "timestamp": datetime.now().isoformat(),
            "sender_name": sender_name,
            "sender_lang": sender_lang,
            "original_text": original_text,
            "translations": translations,
        }
        self.chat_history.append(message)
        self.last_activity = datetime.now()

        if time.time() - self.last_flush > self.flush_interval:
            self.save_to_file()
            self.last_flush = time.time()

        return message

    def get_chat_for_audience(self, lang: str) -> list:
        """Get chat history filtered/translated for a specific language."""
        result = []
        for msg in self.chat_history:
            translated = msg["translations"].get(lang, "")
            # If no translation exists but sender spoke in this lang, use original
            if not translated and msg["sender_lang"] == lang:
                translated = msg["original_text"]
            result.append({
                "id": msg["id"],
                "timestamp": msg["timestamp"],
                "sender_name": msg["sender_name"],
                "original_text": msg["original_text"],
                "translated_text": translated or msg["original_text"],
            })
        return result

    def save_to_file(self):
        """Simpan history ke file JSON."""
        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump({
                    "room_code": self.code,
                    "created_at": self.created_at.isoformat(),
                    "last_activity": self.last_activity.isoformat(),
                    "host_lang": self.host_lang,
                    "total_entries": len(self.history),
                    "entries": self.history,
                    "chat_history": self.chat_history,
                }, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[Room {self.code}] Error saving file: {e}")

    def load_from_file(self):
        """Load history dari file kalau ada."""
        if not os.path.exists(self.file_path):
            return False
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.history = data.get("entries", [])
            self.chat_history = data.get("chat_history", [])
            self.host_lang = data.get("host_lang", "en")
            self.created_at = datetime.fromisoformat(data.get("created_at", datetime.now().isoformat()))
            print(f"[Room {self.code}] Loaded {len(self.history)} entries, {len(self.chat_history)} chat messages")
            return True
        except Exception as e:
            print(f"[Room {self.code}] Error loading file: {e}")
            return False

    def get_history_for_audience(self, lang: str) -> list:
        """Ambil history dalam format yang dibutuhkan audiens."""
        result = []
        for entry in self.history:
            result.append({
                "timestamp": entry["timestamp"],
                "speaker": entry.get("speaker", "host"),
                "original_text": entry["original_text"],
                "translated_text": entry["translations"].get(lang, ""),
            })
        return result

    def get_info(self) -> dict:
        """Info ringkas tentang room."""
        return {
            "code": self.code,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "host_connected": self.host_ws is not None,
            "host_lang": self.host_lang,
            "total_audiences": len(self.audiences),
            "total_entries": len(self.history),
            "total_chat_messages": len(self.chat_history),
            "audiences": [
                {"lang": info["lang"], "name": info.get("name", "Anonymous")}
                for info in self.audiences.values()
            ],
        }


class RoomManager:
    def __init__(self):
        self.rooms = {}  # {code: Room}

    def create_room(self) -> Room:
        """Buat room baru dengan kode unik."""
        for _ in range(10):
            code = generate_room_code()
            if code not in self.rooms:
                break
        else:
            raise Exception("Failed to generate unique room code")

        room = Room(code)
        room.load_from_file()
        self.rooms[code] = room
        print(f"[RoomManager] Room {code} created")
        return room

    def get_room(self, code: str) -> Optional[Room]:
        return self.rooms.get(code)

    def delete_room(self, code: str):
        room = self.rooms.get(code)
        if room:
            room.save_to_file()
            del self.rooms[code]
            print(f"[RoomManager] Room {code} deleted")

    def list_rooms(self) -> list:
        return [room.get_info() for room in self.rooms.values()]


# Instance global
manager = RoomManager()