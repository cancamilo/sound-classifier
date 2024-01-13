import numpy as np
import librosa
# import tensorflow.lite as tflite
import tflite_runtime.interpreter as tflite


DATA_PATH = "data/samples/" # path to folder containing audio files

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

class TFLiteModel:

    def __init__(self) -> None:
        self.interpreter = tflite.Interpreter(model_path='model/sound-model.tflite')
        self.interpreter.allocate_tensors()

        self.input_index = self.interpreter.get_input_details()[0]['index']
        self.output_index = self.interpreter.get_output_details()[0]['index']

    def preprocess(self, audio_signal, sr):
        X = []
        mfcc_ = librosa.feature.mfcc(y=audio_signal, sr=sr, n_mfcc=13)
        X.append(mfcc_)
        return X
    
    def feature_to_tf(self, feature):
        X = np.array(feature)
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
        return X

    def predict(self, audio_signal, sr):
        X = self.preprocess(audio_signal, sr)
        X = self.feature_to_tf(X)

        self.interpreter.set_tensor(self.input_index, X)
        self.interpreter.invoke()
        preds = self.interpreter.get_tensor(self.output_index)

        # Get the predicted class
        predicted_class = np.argmax(preds)
        return inverted_dict[predicted_class]