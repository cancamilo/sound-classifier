from multiprocessing import Pool
import pathos.pools as pp
import scipy.io.wavfile as sci_wav
import bc_utils as U
import pandas as pd
import numpy as np
import threading
import random
import keras
import os


class ESC50(keras.utils.Sequence):
    """This class is shipped with a generator yielding audio from ESC10 or
    ESC50. You may specify the folds you want to used

    eg:
    train = ESC50(folds=[1,2,3])
    train.data_gen.next()

    Parameters
    ----------
    folds : list of integers
        The folds you want to load

    only_ESC10 : boolean
        Wether to use ESC10 instead of ESC50
    """
    def __init__(self,
                 csv_path = '../meta/esc50.csv',
                 wav_dir = '../audio',
                 dest_dir = None,
                 batch_size=16,
                 only_ESC10=False,
                 folds=[1,2],
                 randomize=True,
                 audio_rate=44100,
                 strongAugment=False,
                 pad=0,
                 inputLength=0,
                 random_crop=False,
                 mix=False,
                 normalize=False):
        '''Initialize the generator

        Parameters
        ----------
        csv_path : str
            Path of the CSV file
        wav_dir : str
            path of the wav files
        dest_dir : str
            Directory where the sub-sampled wav are stored
        only_ESC10: Bool
            Wether to use ESC10 instead of ESC50
        randomize: Bool
            Randomize samples 
        audio_rate: int
            Audio rate of our samples
        strongAugment: Bool 
           rAndom scale and put gain in audio input 
        pad: int
            Add padding before and after audio signal
        inputLength: float
            Time in seconds of the audio input
        random_crop: Bool
            Perform random crops
        normalize: int
            Value used to normalize input
        mix: Bool
            Wether to mix samples or not (between classes learning)
        '''
        # file paths
        self.csv_path = csv_path
        self.wav_dir = wav_dir
        self.dest_dir = (dest_dir if dest_dir 
                                  else os.path.join(wav_dir, str(audio_rate)))

        # Batch options
        self.batch_size = batch_size
        self.n_classes = 50

        # Preprocessing options
        self.audio_rate = audio_rate
        self.randomize = randomize
        self.audio_rate = audio_rate
        self.strongAugment = strongAugment
        self.pad = pad 
        self.inputLength = inputLength
        self.random_crop = random_crop
        self.normalize = normalize
        self.mix = mix

        # Inner thingy
        self.df = pd.read_csv(self.csv_path)
        self.df[self.df.fold.isin(folds)]
        self._init_fId()
        self._preprocess_setup()

    def _init_fId(self):
        '''
        init the list of file indexes
        '''
        self.fIdsA = list(self.df.index)
        self.fIdsB = list(self.df.index)
        if self.randomize:
            random.shuffle(self.fIdsA)
            random.shuffle(self.fIdsB)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.fIdsA) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        batch_fIdA = self.fIdsA[index*self.batch_size: (index+1)*self.batch_size]
        batch_fIdB = self.fIdsB[index*self.batch_size: (index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(batch_fIdA, batch_fIdB)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self._init_fId()

    def fId_to_sound(self, fId):
        '''
        file idx to preprocess audio sample and label
        '''
        fname = self.df.filename[fId]
        sound = self.fname_to_wav(fname)
        sound = self.preprocess(sound)
        label = self.df.target[fId]
        return sound, label

    def fname_to_wav(self, fname):
        """Retrive wav data from fname
        """
        U.change_audio_rate(fname, self.wav_dir, self.audio_rate, self.dest_dir)
        fpath = os.path.join(self.dest_dir, fname)
        wav_freq, wav_data = sci_wav.read(fpath)
        return wav_data

    def __data_generation(self, batch_fIdA, batch_fIdB):
        'Generates data containing batch_size samples'
        nPool = 4 if self.mix else 1
        pool = pp.ProcessPool(nPool)
        sounds_lbs = pool.map(self._generate_sample, batch_fIdA, batch_fIdB)
        sounds, labels = zip(*sounds_lbs)
        sounds = np.array(sounds)
        labels = np.array(labels)
        if len(labels.shape) == 1:
            labels = keras.utils.to_categorical(labels, self.n_classes)

        return sounds, labels

    def _generate_sample(self, fIdA, fIdB):
        '''
        Takes 2 audio fileidx, preprocess them, mix them (if needed)
        '''
        sound1, label1 = self.fId_to_sound(fIdA)
        sound2, label2 = self.fId_to_sound(fIdB)
        if self.n_classes == 10:
            lbl_indexes = {0:0,  1:1,  10:2, 11:3, 12:4, 
                           20:5, 21:6, 38:7, 40:8, 41:9}
            label1 = lbl_indexes[label1]
            label2 = lbl_indexes[label2]

        if self.mix:  # Mix two examples
            r = np.array(random.random())
            sound = U.mix(sound1, sound2, r, self.audio_rate) # delays
            sound = sound.astype(np.float32)
            eye = np.eye(self.n_classes)
            label = (eye[label1] * r + eye[label2] * (1 - r))
            label = label.astype(np.float32)

        else:
            sound, label = sound1, label1

        if self.strongAugment:
            sound = U.random_gain(6)(sound).astype(np.float32)

        sound = sound[:, np.newaxis]

        return sound, label

    def _preprocess_setup(self):
        """Apply desired pre_processing to the input
        """
        self.preprocess_funcs = []
        if self.strongAugment:
            self.preprocess_funcs.append(U.random_scale(1.25))

        if self.pad > 0:
            self.preprocess_funcs.append(U.padding(self.pad))
        
        if self.random_crop:
            self.preprocess_funcs.append(
                U.random_crop(int(self.inputLength * self.audio_rate)))

        if self.normalize is True:
            self.preprocess_funcs.append(U.normalize(32768.0))

    def preprocess(self, audio):
        """Apply desired pre_processing to the input

        Parameters
        ----------
        audio: array 
            audio signal to be preprocess
        """
        for f in self.preprocess_funcs:
            audio = f(audio)

        return audio


class ESC10(ESC50):
    def __init__(self,
                 csv_path='../meta/esc50.csv',
                 wav_dir='../audio',
                 dest_dir=None,
                 batch_size=16,
                 only_ESC10=False,
                 folds=[1,2],
                 randomize=True,
                 audio_rate=44100,
                 strongAugment=False,
                 pad=0,
                 inputLength=0,
                 random_crop=False,
                 mix=False,
                 normalize=False):
        super(ESC10, self).__init__(
                   csv_path=csv_path,
                   wav_dir=wav_dir,
                   dest_dir=dest_dir,
                   batch_size=batch_size,
                   only_ESC10=only_ESC10,
                   folds=folds,
                   randomize=randomize,
                   audio_rate=audio_rate,
                   strongAugment=strongAugment,
                   pad=pad,
                   inputLength=inputLength,
                   random_crop=random_crop,
                   mix=mix,
                   normalize=normalize)
        self.df = self.df[self.df['esc10']] 
        self.n_classes = 10
        self._init_fId()


if __name__ == '__main__':
    a = ESC10(mix=True)
    from time import time; start = time()
    print(a[0])
    print(time()-start)

