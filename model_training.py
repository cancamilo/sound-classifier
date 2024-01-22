# %%
# Script to train the sound classification model from scratch

# %%
import pandas as pd
import numpy as np
import tensorflow as tf
import librosa
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
from sklearn.model_selection import train_test_split

CSV_FILE_PATH = "data/esc50.csv"  # path of csv file
DATA_PATH = "data/audio/44100/" # path to folder containing audio files
df = pd.read_csv(CSV_FILE_PATH)

# from the full data set with 40 different classes, select the classes you want to filter. 

class_selection = [
    "thunderstorm",
    "rain",
    "sea_weaves",
    "dog",
    "cat",
    "chirping_birds",    
    "breathing",
    "keyboard_typing",
    "coughing",
    "drinking_sipping",
    "car_horn"
]

df_sel = df[df["category"].isin(class_selection)]

# Map each category to a new target column
classes = df_sel['category'].unique()
class_dict = {i:x for x,i in enumerate(classes)}
df_sel.loc[:, 'target'] = df_sel['category'].map(class_dict)

# %%
# define data augmentation functions

def add_noise(data, scale=0.05):
    noise = np.random.normal(0, scale, len(data))
    audio_noisy = data + noise
    return audio_noisy
    
def pitch_shifting(data, sr=16000):
    sr  = sr
    bins_per_octave = 12
    pitch_pm = 2
    pitch_change =  pitch_pm * 2*(np.random.uniform())   
    data = librosa.effects.pitch_shift(data.astype('float64'),  sr=sr, n_steps=pitch_change, 
                                          bins_per_octave=bins_per_octave)
    return data

def random_shift(data):
    timeshift_fac = 0.2 *2*(np.random.uniform()-0.5)  # up to 20% of length
    start = int(data.shape[0] * timeshift_fac)
    if (start > 0):
        data = np.pad(data,(start,0),mode='constant')[0:data.shape[0]]
    else:
        data = np.pad(data,(0,-start),mode='constant')[0:data.shape[0]]
    return data

def volume_scaling(data):
    dyn_change = np.random.uniform(low=1.5,high=2.5)
    data = data * dyn_change
    return data
    
def time_stretching(data, rate=1.5):
    input_length = len(data)
    streching = data.copy()
    streching = librosa.effects.time_stretch(streching, rate=rate)
    
    if len(streching) > input_length:
        streching = streching[:input_length]
    else:
        streching = np.pad(streching, (0, max(0, input_length - len(streching))), "constant")
    return streching

# %%
# Data peparing helper functions

def augment_df(df):
    """
    Apply variouys augmentation strategies the signals provided in the df
    """
    totals = []
    
    for i, row in df.iterrows():
        df_temp = pd.DataFrame()
        signal , sr = librosa.load(DATA_PATH+row["filename"])
        aug_signals = {
            "original": signal,
            "noised": add_noise(signal, 0.005),
            "pitch_shift": pitch_shifting(signal),
            "random_shifted": random_shift(signal),
            "vol_scaled": volume_scaling(signal),
            "time_stretched": time_stretching(signal)
        }

        df_temp = df_temp._append([row]*len(aug_signals),ignore_index=True)

        df_temp["signal"] = aug_signals.values()
        df_temp["type"] = aug_signals.keys()
        
        totals.append(df_temp)
            
    return pd.concat(totals)

def load_signals(df):
    """
    Given a df with references to audio files, load each of the file to a numpy array 
    and returned a new dataframe with the loaded signals
    """
    df["signal"] = df["filename"].apply(lambda x: librosa.load(DATA_PATH+x)[0])
    return df

def df_to_tf(df):
    """
    Given a dataframe with audio signals as numpy arrays, apply the mfcc transformation and categorize the label classes.
    and reshape into X and y for the CNN model. 
    """
    sr = 22050
    X , y = [] , []
    for _, data in df.iterrows():
        mfcc_ = librosa.feature.mfcc(y=data["signal"], sr=sr, n_mfcc=13)
        X.append(mfcc_)
        y.append(data["target"])

    # convert list to numpy array
    X = np.array(X) 
    y = np.array(y)

    #one-hot encoding the target
    y = tf.keras.utils.to_categorical(y , num_classes=10)

    # our tensorflow model takes input as (no_of_sample , height , width , channel).
    # here X has dimension (no_of_sample , height , width).
    # So, the below code will reshape it to (no_of_sample , height , width , 1).
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    return X, y

# %%
from keras.layers import Dropout, BatchNormalization

# Define the CNN architecture
def create_model():
    INPUTSHAPE = (13,216,1)
    model = models.Sequential([
                          layers.Conv2D(16 , (3,3),activation = 'relu',padding='valid', input_shape = INPUTSHAPE),
                          BatchNormalization(),
                          layers.Conv2D(64, (3,3), activation='relu',padding='valid'),
                          BatchNormalization(),
                          layers.Conv2D(32, (3,3), activation='relu',padding='valid'),
                          BatchNormalization(),
                          layers.GlobalAveragePooling2D(),
                          Dropout(0.5),
                          layers.Dense(32 , activation = 'relu'),
                          Dropout(0.5),
                          layers.Dense(10 , activation = 'softmax')
    ])
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = 'acc')
    return model


print("Preparing data for model training...\n")
# %%
df_train, df_val = train_test_split(df_sel, test_size=0.2, random_state=2023)

# augment only the training data
df_train_aug = augment_df(df_train)
df_val = load_signals(df_val)

# %%
X_train, y_train = df_to_tf(df_train_aug)
X_val, y_val = df_to_tf(df_val)

# %%
# Modeling
INPUTSHAPE = (13,216,1)
LOGDIR = "logs"
CPKT = "cpkt/"

#this callback is used to prevent overfitting.
callback_1 = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0.01, patience=60, verbose=0, mode='auto',
    baseline=None, restore_best_weights=False
)

#this checkpoint saves the best weights of model at every epoch
callback_2 = tf.keras.callbacks.ModelCheckpoint(
    CPKT, monitor='val_loss', verbose=0, save_best_only=True,
    save_weights_only=True, mode='auto', save_freq='epoch', options=None
)
model = create_model()
print("Model summary:\n")
print(model.summary())

print("Training...\n")
model.fit(X_train,y_train,
        validation_data=(X_val,y_val),
        epochs=90,
        callbacks = [callback_1 , callback_2])

model.save("model/model.h5")
print("model saved as model/model.h5")


