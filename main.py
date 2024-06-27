# pip install streamlit-audiorec
import streamlit as st
from st_audiorec import st_audiorec
import speech_recognition as sr
import textwrap
from subprocess import Popen, PIPE
from langchain.llms import OpenAI
from langchain.agents import load_tools, initialize_agent, AgentType
from dotenv import load_dotenv


load_dotenv()

st.set_page_config(page_title="streamlit_audio_recorder")


def langchain_agent(text):
    llm = OpenAI(temperature=0.5)

    tools = load_tools(["wikipedia"], llm=llm)
    agent = initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )
    result = agent.run(text)
    return result

def text_to_speech(text):
    pwd = load_dotenv("pwd")
    command = f'echo {pwd} | sudo -S docker run --platform linux/amd64 -v /Users/raylandmagalhaes/spx-data:/data --rm msftspeech/spx spx synthesize --text "{text}" --audio output my-sample.wav'
    with Popen(command, shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE, text=True) as process:
        # Wait for the command to complete and collect its output
        stdout, stderr = process.communicate()
        # Optionally, you can check the exit code and print the output
        if process.returncode == 0:
            print('Command succeeded:')
            print(stdout)
            audio_file = open('/Users/raylandmagalhaes/spx-data/my-sample.wav', 'rb')
            audio_bytes = audio_file.read()
            return audio_bytes
        else:
            print('Command failed:')
            print(stderr)


def audio_transcriber_demo_app():

    st.title('Audio transcriber')

    wav_audio_data = st_audiorec()

    col_info, col_space = st.columns([0.8, 0.6])
    with col_info:
        st.write('\n')  # add vertical spacer
        st.write('\n')  # add vertical spacer

    if wav_audio_data is not None:
        st.subheader("Audio recorded")
        # display audio data as received on the Python side
        col_playback, col_space = st.columns([0.58, 0.42])
        with col_playback:
            st.audio(wav_audio_data, format='audio/wav')

        # # Write the bytes data to a temporary WAV file

        with open("temp.wav", "wb") as f:
            f.write(wav_audio_data)

        # Initialize the recognizer
        recognizer = sr.Recognizer()

        # Transcribe the audio file
        try:
            with sr.AudioFile("temp.wav") as source:
                audio_data = recognizer.record(source)
                text = recognizer.recognize_google(audio_data)
                st.subheader("Transcription: ")
                st.text(textwrap.fill(text, width=80))
                #Generating answer
                response = langchain_agent(text)
                st.subheader("Answer: ")
                st.text(textwrap.fill(response, width=80))
                tts_audio = text_to_speech(response)
                st.audio(tts_audio, format='audio/wav')

        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")



if __name__ == '__main__':
    # call main function
    audio_transcriber_demo_app()
