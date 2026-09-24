<div align="center">

# 🎙️ SmartRoom

### Interactive Real-Time Room Translation System

*Break language barriers. Speak your language. Everyone hears theirs — instantly.*

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Working-success)]()

[Features](#-features) • [Demo](#-demo) • [Tech Stack](#-tech-stack) • [Installation](#-installation) • [Team](#-team)

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

## 📸 Demo

### Host — Choose your language
![Host Landing](docs/screenshots/host-landing.png)

### Host — Live conversation
![Host Dashboard](docs/screenshots/host-dashboard.png)

### Audience — Join and receive translations
![Audience Join](docs/screenshots/audience-join.png)

> 📷 *Add your own screenshots to `docs/screenshots/` and they'll show up here.*

---

## 🏗️ Architecture
