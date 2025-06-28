import os
import streamlit as st
from st_audiorec import st_audiorec
import speech_recognition as sr
import textwrap
from langchain_community.llms import OpenAI
from langchain.agents import initialize_agent, AgentType
from langchain_community.agent_toolkits.load_tools import load_tools
from dotenv import load_dotenv
import types
from TTS.api import TTS as CoquiTTS
import torch

os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

# Load environment variables
load_dotenv()

# Streamlit page config
st.set_page_config(page_title="Audio Transcriber + TTS")
if isinstance(torch.classes, types.ModuleType):
    torch.classes.__path__ = []  # Prevent Streamlit from trying to walk this non-existent path

# Load Coqui TTS
@st.cache_resource(show_spinner="Loading Coqui TTS model...")
def load_coqui_tts():
    return CoquiTTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=torch.cuda.is_available())

coqui_tts = load_coqui_tts()


def langchain_agent(text):
    llm = OpenAI(temperature=0.5)
    tools = load_tools(["wikipedia"], llm=llm)
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=False)
    return agent.run(text)


def text_to_speech_coqui(text):
    coqui_tts.tts_to_file(text=text, file_path="coqui_tts_out.wav")
    with open("coqui_tts_out.wav", "rb") as audio_file:
        return audio_file.read()


def audio_transcriber_demo_app():
    st.title('🎙️ Audio Transcriber + LangChain Agent + Coqui TTS')

    wav_audio_data = st_audiorec()

    if wav_audio_data is not None:
        st.subheader("📥 Recorded Audio")
        st.audio(wav_audio_data, format='audio/wav')

        with open("temp.wav", "wb") as f:
            f.write(wav_audio_data)

        recognizer = sr.Recognizer()

        try:
            with sr.AudioFile("temp.wav") as source:
                audio_data = recognizer.record(source)
                text = recognizer.recognize_google(audio_data)

                st.subheader("📝 Transcription")
                st.text(textwrap.fill(text, width=80))

                response = langchain_agent(text)

                st.subheader("🤖 LangChain Answer")
                st.text(textwrap.fill(response, width=80))

                with st.spinner("Generating speech with Coqui TTS..."):
                    tts_audio = text_to_speech_coqui(response)

                st.subheader("🎧 AI Answer Audio")
                st.audio(tts_audio, format='audio/wav')

        except sr.UnknownValueError:
            st.error("❌ Could not understand the audio.")
        except sr.RequestError as e:
            st.error(f"❌ Could not request results: {e}")


if __name__ == '__main__':
    audio_transcriber_demo_app()
