import os
import io
import streamlit as st
import librosa
import numpy as np
import matplotlib.pyplot as plt
from lite_model import TFLiteModel

DATA_PATH = "data/samples/" # path to folder containing audio files

# Predefined files
PREDEFINED_FILES = [
    "birds.wav", 
    "car.wav", 
    "dog.wav", 
    "coughing.wav",
    "cat.wav",
    "typing.wav",
    "drinking.wav"
]

def main():
    audio_data = None
    title = "Sound Classification"
    st.title(title)

     # Add a select box for the predefined files with an additional option for file upload
    selected_file = st.selectbox("Choose a predefined file or upload your own", ["Upload your own"] + PREDEFINED_FILES)

    # If a predefined file is selected, use it
    if selected_file != "Upload your own":
        with open(os.path.join(DATA_PATH, selected_file), 'rb') as f:
            audio_data = io.BytesIO(f.read())
            audio_signal, sample_rate = librosa.load(audio_data)
    else:
        uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "flac"])
        if uploaded_file is not None:
            audio_data = io.BytesIO(uploaded_file.read())
            audio_signal, sample_rate = librosa.load(audio_data)
    
    if audio_data is not None:
        try:
            st.write("Playing")
            st.audio(audio_data, format='audio/wav')
        except:
            st.write("Please record sound first")

        # Add a placeholder
        if st.button('Display Spectrogram'):
            st.write("spectrogram")

            mel_spec = librosa.feature.melspectrogram(y=audio_signal , sr=sample_rate ,  n_fft=2048, hop_length=512)
            mel_spec = librosa.power_to_db(mel_spec, ref=np.max)  #visualizing mel_spectrogram directly gives black image. So, coverting from power_to_db is required

            # Create a new figure and set its size
            plt.figure(figsize=(10, 4))

            # Display the mel_spec array as an image with a colormap
            plt.imshow(mel_spec, aspect='auto', cmap='inferno')

            # Remove the axes for a cleaner look
            plt.axis('off')

            # Save the figure to a BytesIO object
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)

            st.image(buf, caption='Mel Spectrogram', use_column_width=True)

        if st.button('Classify'):
            model = TFLiteModel()
            with st.spinner("Classifying the sound"):
                sound_class = model.predict(audio_signal, sample_rate)
                st.success("Classification completed")

            st.write("### The sound is ", sound_class)
        

if __name__ == '__main__':
    main()