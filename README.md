🎙️ Audio Transcriber with LangChain Agent & TTS

An interactive Streamlit app that lets you record your voice, transcribe it using Google Speech Recognition, process the transcribed text with a LangChain-powered agent, and generate a spoken response using Coqui TTS.

✨ Features
Record audio directly in the browser
Transcribe speech to text using Google Speech Recognition
Automatically generate intelligent responses using a LangChain agent and Wikipedia tools
Convert text responses back to speech with Coqui TTS (Tacotron2-DDC model)
GPU support for TTS generation if available
Clean, responsive Streamlit UI with real-time feedback
🔧 Tech Stack
Streamlit – for the interactive UI
LangChain – to handle intelligent text processing via LLMs and tools
Coqui TTS – for high-quality text-to-speech audio generation
SpeechRecognition (Google) – for speech-to-text conversion

To run the project, set up your keys on a .env file and run the command 'streamlit run main.py'

![Alt text](Screenshot.png?raw=true "Optional Title")
