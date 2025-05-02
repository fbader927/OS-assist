import os, time, base64, signal, asyncio, threading, struct, io
from datetime import datetime

import requests
from PIL import Image, ImageGrab, ImageFile
import pyaudio
import speech_recognition as sr
from PyQt5 import QtCore, QtGui, QtWidgets

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("Please set the GROQ_API_KEY environment variable.")

GROQ_API_BASE = "https://api.groq.com/openai/v1"
DEFAULT_HEADERS = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}
HEADERS_FORM = {"Authorization": f"Bearer {GROQ_API_KEY}"}

MODEL_VISION = "meta-llama/llama-4-scout-17b-16e-instruct"
MODEL_STT    = "whisper-large-v3-turbo"
MODEL_TTS    = "playai-tts"
VOICE_NAME   = "Arista-PlayAI"          # valid PlayAI voice

# retain full conversation context
MAX_TOKENS_CHAT = 256

# ------------------------------------------------------------
# Globals
# ------------------------------------------------------------

ImageFile.LOAD_TRUNCATED_IMAGES = True
exit_event = threading.Event()
stop_listening_event = threading.Event()
stop_event = threading.Event()
audio_lock = threading.Lock()
audio_thread = None
current_playback_stream = None
p = pyaudio.PyAudio()

# persistent HTTP session (for chat & TTS)
session = requests.Session()
session.headers.update(DEFAULT_HEADERS)

img_dir = os.path.join(os.path.expanduser("~"), "LLM_Vision_OS", "images")
os.makedirs(img_dir, exist_ok=True)
img_path = os.path.join(img_dir, "screenshot.jpg")

screenshot_interval = 2  # seconds
conversation_history = []

# ------------------------------------------------------------
# Groq helpers
# ------------------------------------------------------------

def groq_chat_completion(messages, temperature=0.7):
    resp = session.post(
        f"{GROQ_API_BASE}/chat/completions",
        json={
            "model": MODEL_VISION,
            "messages": messages,
            "max_tokens": MAX_TOKENS_CHAT,
            "temperature": temperature,
            "stream": False
        },
        timeout=45
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()

def transcribe_audio_whisper(wav_bytes: bytes) -> str:
    # Use requests.post here to avoid carrying over session's JSON header
    resp = requests.post(
        f"{GROQ_API_BASE}/audio/transcriptions",
        headers=HEADERS_FORM,
        data={"model": MODEL_STT, "language": "en"},
        files={"file": ("audio.wav", wav_bytes, "audio/wav")},
        timeout=45
    )
    resp.raise_for_status()
    return resp.json()["text"].strip()

# ------------------------------------------------------------
# TTS
# ------------------------------------------------------------

def synthesize_speech(text: str) -> bytes:
    resp = session.post(
        f"{GROQ_API_BASE}/audio/speech",
        json={
            "model": MODEL_TTS,
            "input": text,
            "voice": VOICE_NAME,
            "response_format": "wav"
        },
        timeout=45
    )
    resp.raise_for_status()
    return resp.content

def _wav_sr(wav: bytes) -> int:
    return struct.unpack_from("<I", wav, 24)[0]

def play_wav(wav: bytes, chunk=2048):
    global audio_thread, stop_event, current_playback_stream

    def _play():
        stop_event.clear()
        try:
            sr_rate = _wav_sr(wav)
            pcm = wav[44:]
            stream = p.open(format=pyaudio.paInt16, channels=1, rate=sr_rate, output=True)
            current_playback_stream = stream
            for i in range(0, len(pcm), chunk):
                if stop_event.is_set():
                    break
                stream.write(pcm[i:i+chunk])
            stream.stop_stream()
            stream.close()
        except Exception as exc:
            print(f"Audio play error: {exc}")

    with audio_lock:
        if audio_thread and audio_thread.is_alive():
            stop_event.set()
            audio_thread.join()
        audio_thread = threading.Thread(target=_play, daemon=True)
        audio_thread.start()

# ------------------------------------------------------------
# Vision
# ------------------------------------------------------------

def vision_analyze_image(path: str) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    content = [
        {"type": "text", "text": "Describe this screenshot briefly."},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
    ]
    return groq_chat_completion([{"role": "user", "content": content}])

# ------------------------------------------------------------
# GUI
# ------------------------------------------------------------

class LLMVisionApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.screenshot_thread = None
        self.listener_thread   = None
        self.listening         = False
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("LLM Vision OS (Groq)")
        self.setGeometry(100, 100, 800, 600)
        self.setStyleSheet("background-color:#2E2E2E; color:#FFFFFF;")

        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        vbox = QtWidgets.QVBoxLayout(central)

        # Top controls
        hbox = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton("Start")
        self.start_btn.setStyleSheet("background-color:#444;color:#fff;")
        self.start_btn.clicked.connect(self.toggle_listen)

        self.export_btn = QtWidgets.QPushButton("Export log")
        self.export_btn.setStyleSheet("background-color:#444;color:#fff;")
        self.export_btn.clicked.connect(self.export_log)

        self.interval_input = QtWidgets.QLineEdit()
        self.interval_input.setPlaceholderText("Screenshot Interval (s)")
        self.interval_input.setValidator(QtGui.QIntValidator(1, 60))
        self.interval_input.setFixedWidth(180)
        self.interval_input.setStyleSheet("background-color:#333;color:#fff;")

        hbox.addWidget(self.start_btn)
        hbox.addWidget(self.export_btn)
        hbox.addWidget(self.interval_input)
        vbox.addLayout(hbox)

        # Output areas
        self.image_output  = QtWidgets.QTextEdit(readOnly=True)
        self.image_output.setStyleSheet("background-color:#333;color:#fff;")
        self.speech_output = QtWidgets.QTextEdit(readOnly=True)
        self.speech_output.setStyleSheet("background-color:#333;color:#fff;")

        vbox.addWidget(QtWidgets.QLabel("Image Analysis Output:"))
        vbox.addWidget(self.image_output)
        vbox.addWidget(QtWidgets.QLabel("Conversation:"))
        vbox.addWidget(self.speech_output)

    # ---------- GUI helpers ----------
    def update_image_output(self, text: str):
        ts = datetime.now().strftime("%H:%M:%S")
        QtCore.QMetaObject.invokeMethod(
            self.image_output,
            "append",
            QtCore.Qt.QueuedConnection,
            QtCore.Q_ARG(str, f"[{ts}] {text}")
        )

    def update_speech_output(self, text: str, user: bool=False):
        ts = datetime.now().strftime("%H:%M:%S")
        role = "User" if user else "Assistant"
        QtCore.QMetaObject.invokeMethod(
            self.speech_output,
            "append",
            QtCore.Qt.QueuedConnection,
            QtCore.Q_ARG(str, f"\n[{ts}] {role}: {text}\n")
        )

    # ---------- Buttons ----------
    def toggle_listen(self):
        global screenshot_interval
        if self.interval_input.text():
            screenshot_interval = int(self.interval_input.text())
        if self.listening:
            self.start_btn.setText("Start")
            stop_audio()
            exit_event.set(); stop_listening_event.set()
            if self.screenshot_thread:
                self.screenshot_thread.join(timeout=5)
            if self.listener_thread:
                self.listener_thread.join(timeout=5)
            exit_event.clear(); stop_listening_event.clear()
            self.screenshot_thread = self.listener_thread = None
        else:
            self.start_btn.setText("Stop")
            self.screenshot_thread = threading.Thread(target=run_screenshot_loop, daemon=True)
            self.screenshot_thread.start()
            self.listener_thread   = threading.Thread(target=listen_to_microphone, daemon=True)
            self.listener_thread.start()
        self.listening = not self.listening

    def export_log(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Log", "", "Text Files (*.txt);;All Files (*)"
        )
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write("Image Analysis Output:\n")
                f.write(self.image_output.toPlainText())
                f.write("\n\nConversation:\n")
                f.write(self.speech_output.toPlainText())
            QtWidgets.QMessageBox.information(self, "Export Successful", f"Log exported to {path}")

# ------------------------------------------------------------
# Screenshot + Vision loop
# ------------------------------------------------------------

async def take_screenshot_loop():
    while not exit_event.is_set():
        try:
            img = ImageGrab.grab().resize((640, 360))
            with io.BytesIO() as buf:
                img.save(buf, format="JPEG", quality=60)
                with open(img_path, "wb") as f:
                    f.write(buf.getvalue())
            text = vision_analyze_image(img_path)
            app.update_image_output(text)
            conversation_history.append({"role": "assistant", "content": text})
        except Exception as exc:
            print("Vision error:", exc)
        for _ in range(screenshot_interval * 10):
            if exit_event.is_set():
                return
            await asyncio.sleep(0.1)

def run_screenshot_loop():
    try:
        asyncio.run(take_screenshot_loop())
    except Exception as exc:
        print("Screenshot loop exception:", exc)

# ------------------------------------------------------------
# Microphone + STT
# ------------------------------------------------------------

def listen_to_microphone():
    recognizer, mic = sr.Recognizer(), sr.Microphone()
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 0.6  # more responsive
    with mic as src:
        recognizer.adjust_for_ambient_noise(src, duration=0.5)
    print("Listening …")

    def callback(_, audio: sr.AudioData):
        if stop_listening_event.is_set():
            return False
        try:
            stop_audio()
            text = transcribe_audio_whisper(audio.get_wav_data())
            if text:
                app.update_speech_output(text, True)
                if text.strip().lower() == "exit":
                    exit_event.set()
                    os._exit(0)
                asyncio.run(process_user_request(text))
        except Exception as exc:
            print("STT error:", exc)

    stop_fn = recognizer.listen_in_background(mic, callback, phrase_time_limit=6)
    try:
        while not exit_event.is_set() and not stop_listening_event.is_set():
            time.sleep(0.1)
    finally:
        stop_fn(wait_for_stop=False)

# ------------------------------------------------------------
# LLM → TTS pipeline
# ------------------------------------------------------------

async def process_user_request(user_input: str):
    conversation_history.append({"role": "user", "content": user_input})
    t0 = time.time()
    try:
        reply = groq_chat_completion(conversation_history)
        conversation_history.append({"role": "assistant", "content": reply})
        app.update_speech_output(reply)
        play_wav(synthesize_speech(reply))
        print(f"Total pipeline time: {time.time() - t0:.2f}s")
    except Exception as exc:
        print("LLM / TTS error:", exc)

# ------------------------------------------------------------
# Misc
# ------------------------------------------------------------

def stop_audio():
    global current_playback_stream
    stop_event.set()
    if current_playback_stream:
        current_playback_stream.stop_stream()

signal.signal(signal.SIGINT, lambda *_: (print("Exiting…"), exit_event.set(), os._exit(0)))

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    global app
    qapp = QtWidgets.QApplication([])
    app = LLMVisionApp()
    app.show()
    qapp.exec_()

if __name__ == "__main__":
    main()
