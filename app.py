import time, os
import logging
import streamlit as st
import numpy as np
import librosa, librosa.display
import matplotlib.pyplot as plt
from PIL import Image
# from settings import IMAGE_DIR, DURATION, WAVE_OUTPUT_FILE
# from src.sound import sound
# from src.model import CNN
# from setup_logging import setup_logging
import streamlit as st

def main():
    title = "Guitar Chord Recognition"
    st.title(title)
    # image = Image.open(os.path.join(IMAGE_DIR, 'app_guitar.jpg'))
    # st.image(image, use_column_width=True)

    uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "flac"])

    if uploaded_file is not None:
        audio_data = uploaded_file.read()
        # process the audio data with your machine learning model

    if st.button('Play'):
        # sound.play()
        try:
            st.write("Play")
            # audio_file = open(WAVE_OUTPUT_FILE, 'rb')
            # audio_bytes = audio_file.read()
            # st.audio(audio_bytes, format='audio/wav')
        except:
            st.write("Please record sound first")

    if st.button('Classify'):
        # cnn = init_model()
        # with st.spinner("Classifying the chord"):
        #     chord = cnn.predict(WAVE_OUTPUT_FILE, False)
        st.success("Classification completed")
        # st.write("### The recorded chord is **", chord + "**")
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