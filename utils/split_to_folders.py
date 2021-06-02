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
true_dir = "C:/Users/noaiz/Desktop/Thesis/cheat_detector_keras-master/data/true_claims/"
false_dir = "C:/Users/noaiz/Desktop/Thesis/cheat_detector_keras-master/data/false_claims/"
heb_dir = "C:/Users/noaiz/Desktop/Thesis/cheat_detector_keras-master/data/heb_claims/"
eng_dir = "C:/Users/noaiz/Desktop/Thesis/cheat_detector_keras-master/data/eng_claims/"
hebT_dir = "C:/Users/noaiz/Desktop/Thesis/cheat_detector_keras-master/data/heb_claims/TRUE/"
engT_dir = "C:/Users/noaiz/Desktop/Thesis/cheat_detector_keras-master/data/eng_claims/TRUE/"
hebF_dir = "C:/Users/noaiz/Desktop/Thesis/cheat_detector_keras-master/data/heb_claims/FALSE/"
engF_dir = "C:/Users/noaiz/Desktop/Thesis/cheat_detector_keras-master/data/eng_claims/FALSE/"

Results_csv_path = "C:/Users/noaiz/Desktop/Thesis/cheat_detector_keras-master/results.csv"
claimed_true_dir = "C:/Users/noaiz/Desktop/Thesis/cheat_detector_keras-master/data/claimed_true_claims/"
claimed_false_dir = "C:/Users/noaiz/Desktop/Thesis/cheat_detector_keras-master/data/claimed_false_claims/"
audio_clean_dir = "C:/Users/noaiz/Desktop/claims/DATA/"
true_clean_dir = "C:/Users/noaiz/Desktop/claims/DATA/true_claims/"
false_clean_dir = "C:/Users/noaiz/Desktop/claims/DATA/false_claims/"
# os.rename("path/to/current/file.foo", "path/to/new/destination/for/file.foo")
# shutil.move("path/to/current/file.foo", "path/to/new/destination/for/file.foo")
# os.replace("path/to/current/file.foo", "path/to/new/destination/for/file.foo")
# DATA_PATH = "/home/evgeny/code_projects/cheat_detector_model/data/CheatGameLogs"
DATA_PATH = "C:/Users/noaiz/Desktop/Thesis/CheatGameLogs"
DeceptionDB_path = "C:/Users/noaiz/Desktop/Thesis/cheat_detector_keras-master/data/DeceptionDB/"
# DeceptionDB_csv_path = "/home/evgeny/code_projects/cheat_detector_model/data/DeceptionDB/description.csv"
# DeceptionDB_csv_path = "/home/evgeny/code_projects/cheat_detector_model/data/description.csv"
DeceptionDB_csv_path = "C:/Users/noaiz/Desktop/Thesis/cheat_detector_keras-master/data/DeceptionDB/DB_description.csv"

def outOfGame():
    # os.mkdir(claimed_true_dir)
    # os.mkdir(claimed_false_dir)

    counterF = 0
    counterT = 0
    counterN = 0
    df = pd.read_csv(Results_csv_path, usecols=["file_name", "Claimed"])
    names = df["file_name"]
    claims = df.Claimed
    for audio_filename, claim in zip(names, claims):
        try:
            if path.exists(DeceptionDB_path + audio_filename+".wav"):
                i = 1
                original = audio_filename+".wav"
                audio_name = audio_filename + "_"
                audio_filename = audio_name + str(i) + ".wav"
                while path.exists(claimed_false_dir + audio_filename) or path.exists(claimed_true_dir + audio_filename) :
                    i += 1
                    audio_filename = audio_name + str(i) + ".wav"
                if audio_filename.find("wav") == -1:
                    audio_filename+=".wav"
                if claim:
                    counterT += 1
                    copyfile(DeceptionDB_path + original, claimed_true_dir + audio_filename)
                elif not claim:
                    counterF += 1
                    copyfile(DeceptionDB_path + original, claimed_false_dir + audio_filename)
            else:
                print(audio_filename)
                counterN += 1
        except:
            print(audio_filename, claim)
    print("True:", counterT)
    print("False:", counterF)
    print("Nothing:", counterN)
    print("Sum:", (counterT + counterF))

def main():

    outOfGame()
    # # os.mkdir(true_dir)
    # # os.mkdir(false_dir)

    ## TRUE-FALSE (DATA)
    # os.mkdir(heb_dir)
    # os.mkdir(eng_dir)
    # counterF = 0
    # counterT = 0
    # counterN = 0
    # df = pd.read_csv(DeceptionDB_csv_path, usecols=["file_name", "IsTrueClaim"])
    # names = df["file_name"]
    # claims = df.IsTrueClaim
    # for audio_filename, claim in zip(names, claims):
    #     # print(claim)
    #     try:
    #         if path.exists(audio_dir + audio_filename):
    #
    #             if claim:
    #                 counterT += 1
    #                 copyfile(DeceptionDB_path + audio_filename, true_dir + audio_filename)
    #             elif not claim:
    #                 counterF += 1
    #                 copyfile(DeceptionDB_path + audio_filename, false_dir + audio_filename)
    #         else:
    #             print(audio_filename)
    #             counterN += 1
    #     except:
    #         print(audio_filename, claim)
    # print("True:", counterT)
    # print("False:", counterF)
    # print("Nothing:", counterN)
    # print("Sum:", (counterT + counterF))


    # ## TRUE-FALSE (CLEANED-DATA)
    # # os.mkdir(true_clean_dir)
    # # os.mkdir(false_clean_dir)
    # counterF = 0
    # counterT = 0
    # counterN = 0
    # df = pd.read_csv(DeceptionDB_csv_path, usecols=["file_name", "IsTrueClaim"])
    # names = df["file_name"]
    # claims = df.IsTrueClaim
    # for audio_filename, claim in zip(names, claims):
    #     # print(claim)
    #     try:
    #         if path.exists(audio_clean_dir + audio_filename):
    #
    #             if claim:
    #                 counterT += 1
    #                 copyfile(DeceptionDB_path + audio_filename, true_clean_dir + audio_filename)
    #             elif not claim:
    #                 counterF += 1
    #                 copyfile(DeceptionDB_path + audio_filename, false_clean_dir + audio_filename)
    #         else:
    #             print(audio_filename)
    #             counterN += 1
    #     except:
    #         print(audio_filename, claim)
    # print("True:", counterT)
    # print("False:", counterF)
    # print("Nothing:", counterN)
    # print("Sum:", (counterT + counterF))
#
# #LANGUAGE
#     # os.mkdir(true_dir)
#     # os.mkdir(false_dir)
#     # os.mkdir(heb_dir)
#     # os.mkdir(eng_dir)
#     counterF = 0
#     counterT = 0
#     counterN = 0
#     df = pd.read_csv(DeceptionDB_csv_path)
#     names = df["file_name"]
#     lans = df[df.columns[4]]
#     print(lans)
#     for audio_filename, lan in zip(names, lans):
#         # print(claim)
#         try:
#             if path.exists(audio_dir + audio_filename):
#
#                 if "Israel" in lan:
#                     counterT += 1
#                     copyfile(DeceptionDB_path + audio_filename, heb_dir + audio_filename)
#                 else:
#                     counterF += 1
#                     copyfile(DeceptionDB_path + audio_filename, eng_dir + audio_filename)
#             else:
#                 print(audio_filename)
#                 counterN += 1
#         except:
#             print(audio_filename, lan)
#     print("HEBREW:", counterT)
#     print("WNGLISH:", counterF)
#     print("Nothing:", counterN)
#     print("Sum:", (counterT + counterF))
#
# #
#     counterF = 0
#     counterT = 0
#     counterN = 0
#     df = pd.read_csv(DeceptionDB_csv_path, usecols=["file_name", "IsTrueClaim"])
#     names = df["file_name"]
#     claims = df.IsTrueClaim
#     for audio_filename, claim in zip(names, claims):
#         # print(claim)
#         try:
#             if path.exists(eng_dir + audio_filename):
#
#                 if claim:
#                     counterT += 1
#                     copyfile(DeceptionDB_path + audio_filename, engT_dir + audio_filename)
#                 elif not claim:
#                     counterF += 1
#                     copyfile(DeceptionDB_path + audio_filename, engF_dir + audio_filename)
#             else:
#                 print(audio_filename)
#                 counterN += 1
#         except:
#             print(audio_filename, claim)
#     print("True:", counterT)
#     print("False:", counterF)
#     print("Nothing:", counterN)
#     print("Sum:", (counterT + counterF))
#
#     df = pd.read_csv(DeceptionDB_csv_path, usecols=["file_name", "IsTrueClaim"])
#     names = df["file_name"]
#     claims = df.IsTrueClaim
#     for audio_filename, claim in zip(names, claims):
#         # print(claim)
#         try:
#             if path.exists(heb_dir + audio_filename):
#
#                 if claim:
#                     counterT += 1
#                     copyfile(DeceptionDB_path + audio_filename, hebT_dir + audio_filename)
#                 elif not claim:
#                     counterF += 1
#                     copyfile(DeceptionDB_path + audio_filename, hebF_dir + audio_filename)
#             else:
#                 print(audio_filename)
#                 counterN += 1
#         except:
#             print(audio_filename, claim)
#     print("True:", counterT)
#     print("False:", counterF)
#     print("Nothing:", counterN)
#     print("Sum:", (counterT + counterF))

if __name__ == "__main__":
  main()
