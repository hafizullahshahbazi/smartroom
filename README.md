<div align="center">

# 🎙️ SmartRoom

### Interactive Real-Time Room Translation System

*Break language barriers. Speak your language. Everyone hears theirs — instantly.*

**Status:** ✅ Working &nbsp;•&nbsp; **License:** MIT &nbsp;•&nbsp; **Python:** 3.11 &nbsp;•&nbsp; **Docker:** Ready

[Features](#-features) • [Architecture](#-architecture) • [Tech Stack](#-tech-stack) • [Installation](#-installation) • [Team](#-team)

</div>

---

## 📖 About the Project

**SmartRoom** is a web-based platform that lets people speaking **different native languages** communicate in real time inside shared virtual rooms. Speak or type in your own language — every participant sees and hears your message in *their* language.

Built to solve a real problem: in multilingual meetings, lectures, and group calls, everything ends up in one shared language (usually English), and quieter voices get lost. SmartRoom removes that friction.

> 💡 **The twist:** SmartRoom runs on modest hardware — just **4 GB RAM and 2 CPUs** — proving that real-time translation doesn't require a server farm.

---

## ✨ Features

- 🌍 **5 Languages supported** — English, Persian (Farsi), Russian, Chinese, Spanish
- 🎙️ **Live microphone mode** — continuous speech recognition with automatic sentence detection
- 🎤 **Push-to-talk** — send voice messages as WhatsApp-style chat bubbles
- 💬 **Real-time text chat** with automatic multilingual translation
- 🔊 **Voice-to-voice translation** — hear the message in your language via neural TTS
- 🌙 **Dark / Light theme** — remembered across sessions
- 📝 **Transcript export** — download as TXT or JSON
- ⚡ **Sub-3-second text latency**, **~10-second voice latency**
- 🐳 **Fully containerized** with Docker Compose

---

## 🏗️ Architecture
```text
┌──────────────────┐        WebSocket        ┌──────────────────────┐
│                  │◄───────────────────────►│                      │
│  Frontend        │                         │  FastAPI Backend     │
│  (HTML/CSS/JS)   │◄───────────────────────►│  - WebSocket Server  │
│                  │                         │  - Room Manager      │
│  • MediaRecorder │                         │  • Chat / Transcript │
│  • Web Audio API │                         │                      │
│  • WebSocket     │                         └──────────┬───────────┘
└──────────────────┘                                    │
                                                        │
                                    ┌───────────────────▼────────────────────┐
                                    │      ML Pipeline (pipeline.py)         │
                                    ├────────────────────────────────────────┤
                                    │  1. Whisper (STT)          → Text      │
                                    │  2. NLLB-200 (CTranslate2) → Translate │
                                    │  3. Edge-TTS               → Audio     │
                                    └────────────────────────────────────────┘
    ```



