import google.generativeai as genai
import os
import time
import threading
import speech_recognition as sr
from PIL import Image, ImageGrab, ImageFile
import pyaudio
import asyncio
import signal
import requests
import json
import base64
import numpy as np
from datetime import datetime
from PyQt5 import QtCore, QtGui, QtWidgets

ImageFile.LOAD_TRUNCATED_IMAGES = True  # To handle truncated image files

# Fetch the API key from environment variables
api_key = os.getenv('GOOGLE_API_KEY')

# Configure the Generative AI model
genai.configure(api_key=api_key)
model = genai.GenerativeModel(model_name='gemini-1.5-flash-latest')

# Start the chat session
chat = model.start_chat(history=[])

# Define the image path
img_path = r'C:\Users\fbb92\OneDrive\Desktop\Projects\test\geminiOS\vision\img\img1.png'

# Event to signal when to stop the threads
exit_event = threading.Event()
stop_listening_event = threading.Event()

screenshot_interval = 2  # Default screenshot interval in seconds


class LLMVisionApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.screenshot_thread = None
        self.listener_thread = None

    def init_ui(self):
        self.setWindowTitle("LLM Vision OS")
        self.setGeometry(100, 100, 800, 600)
        self.setStyleSheet("background-color: #2E2E2E; color: #FFFFFF;")

        central_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(central_widget)

        layout = QtWidgets.QVBoxLayout(central_widget)

        top_layout = QtWidgets.QHBoxLayout()

        self.start_button = QtWidgets.QPushButton("Start", self)
        self.start_button.setStyleSheet(
            "background-color: #444444; color: #FFFFFF; font-size: 14px; padding: 5px;")
        self.start_button.clicked.connect(self.toggle_listen)

        self.export_button = QtWidgets.QPushButton("Export log", self)
        self.export_button.setStyleSheet(
            "background-color: #444444; color: #FFFFFF; font-size: 14px; padding: 5px;")
        self.export_button.clicked.connect(self.export_log)

        self.interval_input = QtWidgets.QLineEdit(self)
        self.interval_input.setPlaceholderText("Screenshot Interval (s)")
        self.interval_input.setStyleSheet(
            "background-color: #333333; color: #FFFFFF; font-size: 14px; padding: 5px;")
        self.interval_input.setFixedWidth(200)
        self.interval_input.setValidator(QtGui.QIntValidator(1, 60, self))

        top_layout.addWidget(self.start_button)
        top_layout.addWidget(self.export_button)
        top_layout.addWidget(self.interval_input)

        layout.addLayout(top_layout)

        self.image_output = QtWidgets.QTextEdit(self)
        self.image_output.setReadOnly(True)
        self.image_output.setStyleSheet(
            "background-color: #333333; color: #FFFFFF; font-size: 14px;")

        self.speech_output = QtWidgets.QTextEdit(self)
        self.speech_output.setReadOnly(True)
        self.speech_output.setStyleSheet(
            "background-color: #333333; color: #FFFFFF; font-size: 14px;")

        layout.addWidget(QtWidgets.QLabel("Image Analysis Output:", self))
        layout.addWidget(self.image_output)
        layout.addWidget(QtWidgets.QLabel("Speech Synthesis Output:", self))
        layout.addWidget(self.speech_output)

        self.listening = False

    def toggle_listen(self):
        global screenshot_interval

        if self.interval_input.text():
            screenshot_interval = int(self.interval_input.text())

        if self.listening:
            self.start_button.setText("Start")
            stop_audio()
            if self.screenshot_thread:
                exit_event.set()
                self.screenshot_thread.join()
                exit_event.clear()
            if self.listener_thread:
                stop_listening_event.set()
                self.listener_thread.join()
                stop_listening_event.clear()
        else:
            self.start_button.setText("Stop")
            self.screenshot_thread = threading.Thread(
                target=run_screenshot_loop)
            self.screenshot_thread.daemon = True
            self.screenshot_thread.start()
            self.listener_thread = threading.Thread(
                target=listen_to_microphone)
            self.listener_thread.start()
        self.listening = not self.listening

    def update_image_output(self, text):
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_text = f"[{timestamp}] {text}"
        QtCore.QMetaObject.invokeMethod(
            self.image_output, "append", QtCore.Qt.QueuedConnection, QtCore.Q_ARG(str, formatted_text))

    def update_speech_output(self, text, is_user=False):
        timestamp = datetime.now().strftime("%H:%M:%S")
        prefix = "User" if is_user else "Assistant"
        formatted_text = f"\n[{timestamp}] {prefix}: {text}\n"
        QtCore.QMetaObject.invokeMethod(
            self.speech_output, "append", QtCore.Qt.QueuedConnection, QtCore.Q_ARG(str, formatted_text))

    def export_log(self):
        options = QtWidgets.QFileDialog.Options()
        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export Log", "",
                                                             "Text Files (*.txt);;All Files (*)", options=options)
        if file_name:
            with open(file_name, 'w') as f:
                f.write(self.speech_output.toPlainText())
            QtWidgets.QMessageBox.information(
                self, "Export Successful", f"Log exported to {file_name}")


# Create a global reference to the GUI application instance
app = None


async def take_screenshot():
    global screenshot_interval
    while not exit_event.is_set():
        try:
            screenshot = ImageGrab.grab()
            screenshot = screenshot.resize((800, 450))  # Downscale the image
            screenshot.save(img_path)
            await preprocess_image(img_path)
        except Exception as e:
            print(f"Error taking screenshot: {e}")
        # Take a screenshot every screenshot_interval seconds
        await asyncio.sleep(screenshot_interval)


def run_screenshot_loop():
    asyncio.run(take_screenshot())


async def preprocess_image(img_path):
    start_time = time.time()
    try:
        img = Image.open(img_path)
        response = await asyncio.to_thread(
            model.generate_content,
            ["This image is for preprocessing:", img],
            generation_config=genai.GenerationConfig(
                temperature=0.7
            ),
            safety_settings={
                "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
                "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
                "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
                "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE"
            }
        )
        print(f"Preprocessed image output: {response.text}")
        app.update_image_output(response.text)
        # Send the preprocessed image context to the chat
        chat.send_message(f"Image analysis: {response.text}")
    except Exception as e:
        print(f"Error during preprocessing: {e}")
    end_time = time.time()
    print(f"Time for image analysis: {end_time - start_time:.2f} seconds")

# Global variables for audio playback
stop_event = threading.Event()
audio_lock = threading.Lock()
audio_thread = None
current_playback_stream = None
p = pyaudio.PyAudio()


def listen_to_microphone():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    print("Listening...")

    def callback(recognizer, audio):
        if stop_listening_event.is_set():
            return False
        try:
            stop_audio()  # Stop any playing audio immediately when speech is detected
            print("Processing...")
            text = recognizer.recognize_google(audio)
            print(f"Recognized: {text}")
            # Add this line to show user's speech
            app.update_speech_output(text, is_user=True)
            if text.strip().lower() == 'exit':
                exit_event.set()
                os._exit(0)
                return False
            asyncio.run(process_request(text))
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            print(
                f"Could not request results from Google Speech Recognition service; {e}")

    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
    stop_listening = recognizer.listen_in_background(microphone, callback)

    try:
        while not exit_event.is_set() and not stop_listening_event.is_set():
            time.sleep(0.1)
    finally:
        # Use wait_for_stop to stop listening immediately
        stop_listening(wait_for_stop=False)


async def synthesize_text(text):
    start_synthesize_time = time.time()

    url = f"https://texttospeech.googleapis.com/v1/text:synthesize?key={
        api_key}"

    headers = {
        "Content-Type": "application/json; charset=utf-8"
    }

    data = {
        "input": {
            "text": text
        },
        "voice": {
            "languageCode": "en-US",
            "name": "en-US-Journey-F"
        },
        "audioConfig": {
            "audioEncoding": "LINEAR16",
            "sampleRateHertz": 24000
        }
    }

    response = await asyncio.to_thread(requests.post, url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        audio_content = response.json()["audioContent"]
        audio_buffer = base64.b64decode(audio_content)
        audio_buffer = audio_buffer[44:]  # Remove the first 44 bytes
        print("Speech synthesized and streaming started")
    else:
        print(f"Speech synthesis failed with status code {
              response.status_code}")
        print(response.text)
        return None

    end_synthesize_time = time.time()
    print(f"Time for synthesis: {
          end_synthesize_time - start_synthesize_time:.2f} seconds")

    return audio_buffer


def play_audio_from_buffer(audio_buffer):
    global audio_thread, stop_event, current_playback_stream

    def play_audio_internal():
        stop_event.clear()
        try:
            stream = p.open(format=pyaudio.paInt16,
                            channels=1,
                            rate=24000,
                            output=True)
            current_playback_stream = stream
            chunk_size = 1024
            for i in range(0, len(audio_buffer), chunk_size):
                if stop_event.is_set():
                    break
                stream.write(audio_buffer[i:i+chunk_size])
            stream.stop_stream()
            stream.close()
        except Exception as e:
            print(f"Error playing the buffer: {e}")

    with audio_lock:
        if audio_thread and audio_thread.is_alive():
            stop_event.set()
            audio_thread.join()
        audio_thread = threading.Thread(target=play_audio_internal)
        audio_thread.start()


def stop_audio():
    global stop_event, current_playback_stream
    stop_event.set()
    if current_playback_stream:
        current_playback_stream.stop_stream()


async def process_request(user_input):
    try:
        start_time = time.time()

        # Add system instruction as part of the conversation flow
        system_instruction = (
            "You are an assistant helping the user with their tasks. "
            "You will provide relevant information based on the user's queries and the context from the latest image data. "
            "Ensure responses are natural and conversational. Do not use emojis and do not mention pictures or images."
            "DO NOT MENTION IMAGES"
            "DO NOT MENTION USERS"
            "DO NOT MENTION ASTERISK OR RESPOND WITH THE CHARACTER '*' "
            "ONLY ANSWER QUESTIONS AS DIRECTED"
            "DO NOT RETURN THE ASTERISK CHARACTER. '*' TRIGGERS MY OCD"
            "DO NOT START WITH 'IT SEEMS LIKE' OR 'BASED ON THE IMAGE ANALYSIS'"
            "START EACH CONVERSATION WITH THE ASSUMPTION YOU UNDERSTAND ALL THE CONTEXT OF WHATS BEING SHOWN"
            "IGNORE TERMINALS"
            "DO NOT ASK QUESTIONS"
            "ALWAYS USE CHAT HISTORY AS CONTEXT. THIS IS YOUR MEMORY"
            "RESPONSES SHOULD ALWAYS BE BRIEF UNLESS TOLD OTHERWISE. TWO SENTENCES OR LESS"
        )
        chat.send_message(system_instruction)

        # Generate the main response
        response = chat.send_message(user_input)

        # Print the response
        print(response.text)
        app.update_speech_output(response.text)

        # Synthesize text and get audio buffer
        start_speech_time = time.time()
        audio_buffer = await synthesize_text(response.text)

        # Measure the end time right after synthesis
        end_time = time.time()

        # Play the synthesized text from the buffer
        if audio_buffer:
            play_audio_from_buffer(audio_buffer)

        # Calculate and print the elapsed time
        elapsed_time = end_time - start_time
        speech_time = start_speech_time - start_time
        print(f"Total request time: {elapsed_time:.2f} seconds")
        print(f"Time from user input to speech synthesis start: {
              speech_time:.2f} seconds")

    except Exception as e:
        print(f"Error during processing: {e}")


def signal_handler(sig, frame):
    print("Exiting program due to keyboard interrupt...")
    exit_event.set()
    os._exit(0)  # Forcefully exit


signal.signal(signal.SIGINT, signal_handler)


def main():
    global app
    try:
        qapp = QtWidgets.QApplication([])

        app = LLMVisionApp()
        app.show()

        qapp.exec_()

    except KeyboardInterrupt:
        print("Exiting program due to keyboard interrupt...")
        exit_event.set()
        os._exit(0)  # Forcefully exit


if __name__ == "__main__":
    main()
