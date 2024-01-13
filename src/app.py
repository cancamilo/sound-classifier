import os
import io
import streamlit as st
import librosa
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

        if st.button('Classify'):
            model = TFLiteModel()
            with st.spinner("Classifying the sound"):
                sound_class = model.predict(audio_signal, sample_rate)
                st.success("Classification completed")

            st.write("### The sound is ", sound_class)


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