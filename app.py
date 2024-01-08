import time, os
import logging
import streamlit as st
import numpy as np
import librosa, librosa.display
import matplotlib.pyplot as plt
from PIL import Image
# from settings import IMAGE_DIR, DURATION, WAVE_OUTPUT_FILE
# from src.sound import sound
from src.model import Model
# from setup_logging import setup_logging
import streamlit as st

# TODO: move constants to settings file
# Constants
DATA_PATH = "data/audio/44100/" # path to folder containing audio files

# Predefined files
PREDEFINED_FILES = [
    "5-259169-A-5.wav", 
    "5-156999-C-19.wav", 
    "3-243726-A-43.wav", 
    "2-82455-A-23.wav",
    "1-59513-A-0.wav"
]

def main():
    audio_data = None
    file = None
    title = "Sound Classification"
    st.title(title)
    # image = Image.open(os.path.join(IMAGE_DIR, 'app_guitar.jpg'))
    # st.image(image, use_column_width=True)

     # Add a select box for the predefined files with an additional option for file upload
    selected_file = st.selectbox("Choose a predefined file or upload your own", ["Upload your own"] + PREDEFINED_FILES)

    # If a predefined file is selected, use it
    if selected_file != "Upload your own":
        with open(os.path.join(DATA_PATH, selected_file), 'rb') as f:
            audio_data = f.read()
            file = selected_file
            # process the audio data with your machine learning model
    else:
        uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "flac"])
        if uploaded_file is not None:
            file = uploaded_file.name
            audio_data = uploaded_file.read()
            # process the audio data with your machine learning model
    
    if audio_data is not None:
        try:
            st.write("Playing")
            st.audio(audio_data, format='audio/wav')
        except:
            st.write("Please record sound first")

        if st.button('Classify'):
            model = Model()
            model.load_model()
            with st.spinner("Classifying the chord"):
                sound_class = model.predict(file)
                st.success("Classification completed")

            st.write("### The sound is ", sound_class)
            # if chord == 'N/A':
            #     st.write("Please record sound first")
            # st.write("\n")

        # Add a placeholder
        if st.button('Display Spectrogram'):
            st.write("spectrogram")
            # if os.path.exists(WAVE_OUTPUT_FILE):
            #     spectrogram, format = get_spectrogram(type='mel')
            #     display(spectrogram, format)
            # else:
            #     st.write("Please record sound first")

if __name__ == '__main__':
    main()