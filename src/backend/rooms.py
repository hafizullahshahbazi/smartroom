"""
@file rooms.py
@brief Room and session management for SmartRoom
@details This module defines the core classes for managing SmartRoom
         sessions:
         - Entity (base class for participants)
         - Audience (derived from Entity — an audience member)
         - Room (a single session: host, audiences, history)
         - RoomManager (global manager for all rooms)
@author Hafizullah Shahbazi
@date 2026-09-24
@version 1.0.0
@copyright MIT License
@see main.py
@see pipeline.py
"""
import json
import os
import time
import random
import string
from datetime import datetime
from typing import Optional


##
# @brief Directory where room JSON files are stored
# @details Created automatically at module import. Each room is
#          persisted to a separate JSON file named after its code.
HISTORY_DIR = "/app/output/rooms"
os.makedirs(HISTORY_DIR, exist_ok=True)


##
# @brief Generate a unique 4-character room code
# @details Combines lowercase letters and digits to create a
#          short, human-readable code that is easy to share.
# @param length Number of characters in the code (default: 4)
# @return A random string of the requested length
# @return Example: "a3f9", "x7k2"
def generate_room_code(length: int = 4) -> str:
    chars = string.ascii_lowercase + string.digits
    return "".join(random.choices(chars, k=length))


# ============ BASE ENTITY (for demonstrating inheritance) ============

##
# @brief Base class for any participant in a room
# @details Represents a generic entity that has a display name.
#          Serves as the base class for Audience, demonstrating
#          OOP inheritance for educational purposes.
# @see Audience
class Entity:
    ##
    # @brief Constructor
    # @param name Display name of the entity
    def __init__(self, name: str):
        ## @brief Display name of the entity
        self.name = name

    ##
    # @brief Get the display name
    # @return The entity's name
    def get_name(self) -> str:
        return self.name


##
# @brief An audience member (participant who receives translations)
# @details Derived from Entity. Represents a user who joined a room
#          and is listening in a specific target language.
# @see Entity
class Audience(Entity):
    ##
    # @brief Constructor
    # @param name Display name
    # @param lang Target language code (e.g., "en", "fa", "ru", "zh", "es")
    def __init__(self, name: str, lang: str):
        super().__init__(name)
        ## @brief Target language code for translations
        self.lang = lang

    ##
    # @brief Get the audience's target language
    # @return Language code (e.g., "en", "fa")
    def get_lang(self) -> str:
        return self.lang


# ============ ROOM CLASS ============

##
# @brief Represents a single SmartRoom session
# @details Holds:
#          - The 4-character room code
#          - Host WebSocket connection
#          - Host language
#          - Audience WebSocket connections and their languages
#          - Transcript history
#          - Chat history
#          - Persistence path
#          The class handles automatic file flushing every 30 seconds
#          to avoid losing data.
# @see RoomManager
# @see Audience
class Room:
    ##
    # @brief Constructor for Room
    # @param code Unique 4-character room code
    def __init__(self, code: str):
        ## @brief Room code (4 chars)
        self.code = code
        ## @brief Creation timestamp
        self.created_at = datetime.now()
        ## @brief Last activity timestamp (updated on every action)
        self.last_activity = datetime.now()
        ## @brief WebSocket connection of the host (None if not connected)
        self.host_ws = None
        ## @brief Host's speaking language (set when host connects)
        self.host_lang = "en"
        ## @brief Dictionary of audience WebSockets -> {lang, name}
        self.audiences = {}
        ## @brief List of transcript entries
        self.history = []
        ## @brief List of chat messages
        self.chat_history = []
        ## @brief File path for JSON persistence
        self.file_path = os.path.join(HISTORY_DIR, f"{code}.json")
        ## @brief Timestamp of the last file flush
        self.last_flush = time.time()
        ## @brief Interval (in seconds) between automatic file flushes
        self.flush_interval = 30
        ## @brief Flag: next host audio is a HOLD (chat only, not transcript)
        self._pending_hold = False

    ##
    # @brief Add a sentence to the transcript history
    # @details Stores the original text, source language, and
    #          translations into all target languages. Triggers
    #          file flush if enough time has elapsed.
    # @param original Original text (in host's language)
    # @param source_lang Language code of the original text
    # @param translations Dictionary mapping language codes to translations
    # @param speaker Speaker name (default: "host")
    # @return The created entry dictionary
    def add_entry(self, original: str, source_lang: str, translations: dict, speaker: str = "host"):
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

    ##
    # @brief Add a chat message to the chat history
    # @details Chat messages can be text or voice (with a "🎤" prefix).
    #          They are automatically translated into all target languages.
    # @param sender_name Display name of the sender
    # @param sender_lang Language code of the sender
    # @param original_text Original text
    # @param translations Dictionary mapping language codes to translations
    # @return The created message dictionary
    def add_chat_message(self, sender_name: str, sender_lang: str, original_text: str, translations: dict):
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

    ##
    # @brief Get chat history in a specific language
    # @details If a translation is missing but the sender spoke the
    #          requested language, the original text is used as fallback.
    # @param lang Target language code
    # @return List of chat messages, each with translated_text
    def get_chat_for_audience(self, lang: str) -> list:
        result = []
        for msg in self.chat_history:
            translated = msg["translations"].get(lang, "")
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

    ##
    # @brief Save room state to a JSON file
    # @details Writes the full room state (code, timestamps, host lang,
    #          transcript, and chat) to self.file_path.
    # @note Errors are logged but do not raise exceptions
    def save_to_file(self):
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

    ##
    # @brief Load room state from a JSON file
    # @details Restores history, chat, host language, and creation date
    #          from self.file_path if the file exists.
    # @return True if data was loaded, False otherwise
    # @note Errors are logged but do not raise exceptions
    def load_from_file(self):
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

    ##
    # @brief Get transcript history for a specific audience language
    # @param lang Target language code
    # @return List of entries with translated_text per requested language
    def get_history_for_audience(self, lang: str) -> list:
        result = []
        for entry in self.history:
            result.append({
                "timestamp": entry["timestamp"],
                "speaker": entry.get("speaker", "host"),
                "original_text": entry["original_text"],
                "translated_text": entry["translations"].get(lang, ""),
            })
        return result

    ##
    # @brief Get a compact summary of the room
    # @return Dictionary with room metadata (code, timestamps, counts,
    #         audience list, and host connection status)
    def get_info(self) -> dict:
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


# ============ ROOM MANAGER ============

##
# @brief Global manager for all active rooms
# @details Maintains a dictionary of all rooms by code. Handles room
#          creation (with collision-safe code generation), lookup,
#          deletion, and listing.
# @see Room
class RoomManager:
    ##
    # @brief Constructor — initializes an empty room dictionary
    def __init__(self):
        ## @brief Dictionary mapping room codes to Room objects
        self.rooms = {}

    ##
    # @brief Create a new room with a unique code
    # @details Tries up to 10 times to generate a non-colliding code.
    #          Attempts to load existing state from file if available.
    # @return The newly created Room object
    # @throw Exception If a unique code cannot be generated after 10 tries
    def create_room(self) -> Room:
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

    ##
    # @brief Get a room by its code
    # @param code The 4-character room code
    # @return The Room object, or None if not found
    def get_room(self, code: str) -> Optional[Room]:
        return self.rooms.get(code)

    ##
    # @brief Delete a room from the manager
    # @details Saves the room to disk first, then removes it from memory.
    # @param code The room code to delete
    def delete_room(self, code: str):
        room = self.rooms.get(code)
        if room:
            room.save_to_file()
            del self.rooms[code]
            print(f"[RoomManager] Room {code} deleted")

    ##
    # @brief List all active rooms with their metadata
    # @return List of dictionaries (one per room)
    def list_rooms(self) -> list:
        return [room.get_info() for room in self.rooms.values()]


##
# @brief Global singleton instance of RoomManager
# @details Imported by main.py to manage all rooms across the app.
manager = RoomManager()