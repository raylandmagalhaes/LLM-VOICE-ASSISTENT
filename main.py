# pip install streamlit-audiorec
import streamlit as st
from st_audiorec import st_audiorec
import speech_recognition as sr
import textwrap
from langchain.llms import OpenAI
from langchain.agents import load_tools, initialize_agent, AgentType
from dotenv import load_dotenv
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
import torch


load_dotenv()


st.set_page_config(page_title="streamlit_audio_recorder")

#Text to speech model loading
device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"
if torch.xpu.is_available():
    device = "xpu"
torch_dtype = torch.float16 if device != "cpu" else torch.float32

model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler_tts_mini_v0.1").to(device, dtype=torch_dtype)
tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler_tts_mini_v0.1")


def langchain_agent(text):
    llm = OpenAI(temperature=0.5)

    tools = load_tools(["wikipedia"], llm=llm)
    agent = initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )
    result = agent.run(text)
    return result


def text_to_speech(text):
    prompt = text
    description = "A female speaker with a slightly low-pitched voice delivers her words quite expressively, in a very confined sounding environment with clear audio quality. She speaks at a normal pace."

    input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
    prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids).to(torch.float32)
    audio_arr = generation.cpu().numpy().squeeze()
    sf.write("parler_tts_out.wav", audio_arr, model.config.sampling_rate)
    audio_file = open('parler_tts_out.wav', 'rb')
    audio_bytes = audio_file.read()
    return audio_bytes



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
