# Use a pipeline as a high-level helper
import os
import streamlit as st 
from dotenv import load_dotenv
load_dotenv()
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import requests
import tensorflow as tf
from dotenv import load_dotenv
from transformers import pipeline
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import Ollama
from PIL import Image
from IPython.display import Audio


def img_to_text(img):
    pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")
    response=pipe(img)
    return response[0]['generated_text']


def generate_story(text):
    template="""You are a good story teller.You will be provided with the few lines of text based on which you will
    generate the good story,the story should not be more than 30 words and don't write anything else except the task.
    
    CONTEXT: {text}
    STORY:
    """
    
    prompt=PromptTemplate(template=template,input_variables=['text'])
    llm=Ollama(base_url="http://localhost:11434",model='llama3')
    chain=LLMChain(llm=llm,prompt=prompt)
    story=chain.predict(text=text)
    return story
    


def text_to_audio(text):
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": f"Bearer {os.getenv('2nd_hug_token')}"}
    payload=text
    response = requests.post(API_URL, headers=headers, json=payload)
    aud=response.content
    audio_file_path="audio.flac"
    with open(audio_file_path,"wb") as f:
        f.write(aud)
    return audio_file_path
    
    


if __name__ == '__main__':
    # a=img_to_text('9dd32abd4e5185c1b102ce2dcfd723d0.jpg')
    # print(a)
    # b=generate_story(a)
    # print(str(b))
    # text_to_audio(str(b))
    st.title("Story generator from image")
    st.header("This website is able to generate a random story related to the picture which the user uploads in it.")
    uploaded_file=st.file_uploader("Upload your file here",type=['jpg','png','jpeg'])
    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        btn=st.button("Click button to get generated story")
        if btn:
            img_text=img_to_text(uploaded_file.name)
            st.write(img_text)
            story=generate_story(img_text)
            st.write(str(story))
            audio_file=text_to_audio(str(story))
            st.audio(audio_file)
            
            
