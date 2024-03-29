from __future__ import print_function
import os
from collections import defaultdict
import scipy.io.wavfile
import imageio
import numpy as np
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE
CLASS_NAMES = np.array(["True" , "False"])
IMG_WIDTH = 64
IMG_HEIGHT = 64


class FSDD_:
    """Summary

    Attributes:
        file_paths (TYPE): Description
        recording_paths (TYPE): Description
    """

    def __init__(self, data_dir):
        """Initializes the FSDD data helper which is used for fetching FSDD data.

        :param data_dir: The directory where the audiodata is located.
        :return: None

        Args:
            data_dir (TYPE): Description
        """

        # A dict containing lists of file paths, where keys are the label and vals.
        self.recording_paths = defaultdict(list)
        file_paths = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
        self.file_paths = file_paths

        for digit in range(0, 10):
            # fetch all the file paths that start with this digit
            digit_paths = [os.path.join(data_dir, f) for f in file_paths if f[0] == str(digit)]
            self.recording_paths[digit] = digit_paths

    @staticmethod
    def get_lable(file_name):
        return file_name[0]

    @classmethod
    def get_spectrograms(cls, data_dir=None):
        """

        Args:
            data_dir (string): Path to the directory containing the spectrograms.

        Returns:
            (spectrograms, labels): a tuple of containing lists of spectrograms images(as numpy arrays) and their corresponding labels as strings
        """
        spectrograms = []
        labels = []

        if data_dir is None:
            data_dir = os.path.dirname(__file__) + '/../spectrograms'

        file_paths = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f)) and '.png' in f]

        if len(file_paths) == 0:
            raise Exception('There are no files in the spectrogram directory. Make sure to run the spectrogram.py before calling this function.')

        for file_name in file_paths:
            label = cls.get_lable(file_name)
            if label is None:
                continue
            spectrogram = imageio.imread(data_dir + '/' + file_name , pilmode = 'F' )
            spectrogram = np.reshape(spectrogram , (64,64,1))
            spectrograms.append(spectrogram)
            labels.append(label)

        print("total num of entris is: " + str(FSDD.num_of_entries))
        print("total num of true statments is: " + str(FSDD.true_rec))
        print("total num of false statments is: " + str(FSDD.false_rec))
        return np.array(spectrograms), np.array(labels)

class FSDD(FSDD_):
    import utils.read_data as read_data
    csv_file = "data/DeceptionDB/description.csv"
    data_labels = read_data.parst_data_labels(csv_file)
    del read_data
    num_of_entries = 0
    true_rec = 0
    false_rec = 0
    def __init__(self, data_dir ):
        super(FSDD, self).__init__(data_dir)

    #overide
    @staticmethod
    def get_lable(file_name):
        entry = next((x for x in FSDD.data_labels if x["filename"].replace("wav","png") == file_name) , None)
        if entry is None:
            return
        FSDD.num_of_entries = FSDD.num_of_entries + 1
        if entry["istruestatment"] == "True":    # [is lie , is true]
            FSDD.true_rec = FSDD.true_rec + 1
            return 0
        else:
            FSDD.false_rec = FSDD.false_rec + 1
            return 1

