# !/usr/bin/env python

import csv
from shutil import copyfile
from pathlib import Path
import xml.etree.ElementTree as ET
import os
import pandas as pd
import os.path
from os import path

audio_dir = "C:/Users/noaiz/Desktop/Thesis/cheat_detector_keras-master/data/DeceptionDB/DATA/"
predicted_as_true_dir = "C:/Users/noaiz/Desktop/Thesis/cheat_detector_keras-master/data/predicted_as_true_dir/"
predicted_as_false_dir = "C:/Users/noaiz/Desktop/Thesis/cheat_detector_keras-master/data/predicted_as_false_dir/"
# os.rename("path/to/current/file.foo", "path/to/new/destination/for/file.foo")
# shutil.move("path/to/current/file.foo", "path/to/new/destination/for/file.foo")
# os.replace("path/to/current/file.foo", "path/to/new/destination/for/file.foo")
# DATA_PATH = "/home/evgeny/code_projects/cheat_detector_model/data/CheatGameLogs"
nn_path = "C:/Users/noaiz/Desktop/Thesis/cheat_detector_keras-master/predict_nn.csv"
cnn_path = "C:/Users/noaiz/Desktop/Thesis/cheat_detector_keras-master/predict_cnn.csv/"
rnn_path = "C:/Users/noaiz/Desktop/Thesis/cheat_detector_keras-master/predict_rnn.csv/"
# DeceptionDB_csv_path = "/home/evgeny/code_projects/cheat_detector_model/data/DeceptionDB/description.csv"
# DeceptionDB_csv_path = "/home/evgeny/code_projects/cheat_detector_model/data/description.csv"
DeceptionDB_csv_path = "C:/Users/noaiz/Desktop/Thesis/cheat_detector_keras-master/data/DeceptionDB/DB_description.csv"
DeceptionDB_path = "C:/Users/noaiz/Desktop/Thesis/cheat_detector_keras-master/data/DeceptionDB/DATA/"


def main():
    # os.mkdir(predicted_as_true_dir)
    # os.mkdir(predicted_as_false_dir)


    counterF = 0
    counterT = 0
    counterN = 0
    df = pd.read_csv(nn_path, usecols=["file-name","predicted-label"])
    names = df["file-name"]
    preds = df["predicted-label"]
    for audio_filename, pred in zip(names, preds):
        # print(claim)
        try:
            if path.exists(audio_dir + audio_filename):

                if pred == 1:
                    counterT += 1
                    copyfile(DeceptionDB_path + audio_filename, predicted_as_true_dir + audio_filename)
                elif pred == 0:
                    counterF += 1
                    copyfile(DeceptionDB_path + audio_filename, predicted_as_false_dir + audio_filename)
            else:
                print(audio_filename)
                counterN += 1
        except:
            print(audio_filename, pred)
    print("True:", counterT)
    print("False:", counterF)
    print("Nothing:", counterN)
    print("Sum:", (counterT + counterF))


if __name__ == "__main__":
    main()
