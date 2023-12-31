import tensorflow as tf
import numpy as np
import librosa

DATA_PATH = "data/audio/44100/" # path to folder containing audio files

class_dict = {
  'dog': 0,
 'chirping_birds': 1,
 'thunderstorm': 2,
 'keyboard_typing': 3,
 'car_horn': 4,
 'drinking_sipping': 5,
 'rain': 6,
 'breathing': 7,
 'coughing': 8,
 'cat': 9}

inverted_dict = {value: key for key, value in class_dict.items()}

class Model:
    def __init__(self) -> None:
        pass

    def preprocess(self, audio_file_path):
        X = []
        signal, sr = librosa.load(DATA_PATH+audio_file_path)
        mfcc_ = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
        X.append(mfcc_)
        return X
    
    def feature_to_tf(self, feature):
        X = np.array(feature)
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
        return X

    def load_model(self):
        # Load the model
        self.model = tf.keras.models.load_model("model/model.h5")

    def predict(self, audio_file_path):
        X = self.preprocess(audio_file_path)
        X = self.feature_to_tf(X)
        prediction = self.model.predict(X)

        # Get the predicted class
        predicted_class = np.argmax(prediction)
        return inverted_dict[predicted_class]

