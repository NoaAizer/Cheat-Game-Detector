
import numpy as np
from datetime import datetime

from keras import layers
from keras import models

import librosa.display

import matplotlib.pyplot as plt

import pandas as pd
import librosa
import librosa.display

import random
import os

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras_preprocessing.image import ImageDataGenerator

from keras.layers import Dense, Flatten, Dropout
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras import optimizers

SEED = 1

def CNN_model():
    # Building our model
    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(64, 64, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(2, activation='softmax'))

    # Compiling using adam and categorical crossentropy+
    opt = optimizers.Adam()
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=1, mode='auto')
    model.summary()
    return model , early_stop


def NN_model():
    model = Sequential()

    model.add(Dense(193, input_shape=(193,), activation='relu'))
    model.add(Dropout(0.1))

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(2, activation='softmax'))

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=1, mode='min')
    return model, early_stop


def RNN_model():
    model = models.Sequential()
    model.add(layers.Conv2D(1, (3, 3), padding='same', activation='relu', bias_initializer="he_normal",
                            kernel_initializer='he_normal', input_shape=(64, 64, 3)))
    for layer in model.layers:
        print(layer.output_shape)
    model.add(layers.Reshape((64, 64)))
    model.add(layers.SimpleRNN(128, input_shape=(64, 64), activation='relu', bias_initializer="random_uniform"))
    model.add(layers.Dense(64, activation='relu', bias_initializer="random_uniform"))
    model.add(layers.Reshape((64, 1)))
    model.add(layers.SimpleRNN(32, activation='relu'))
    model.add(layers.Dense(2, activation='softmax', bias_initializer="random_uniform"))
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=1, mode='auto')
    model.summary()
    return model ,early_stop


def images(files):
    # We define the audiofile from the rows of the dataframe when we iterate through
    # every row of our dataframe for train, val and test
    # audiofile = os.path.join(os.path.abspath('new_test_true_false') + '/' + str(files.file))#origin
    # audiofile = os.path.join(os.path.abspath('data') + '/claimed/' + str(files.file))  # out_of_game
    audiofile = os.path.join(os.path.abspath('data') + '/DeceptionDB/DATA/' + str(files.file)) #VAD



    # Loading the image with no sample rate to use the original sample rate and
    # kaiser_fast to make the speed faster according to a blog post about it (on references)
    X, sample_rate = librosa.load(audiofile, sr=None, res_type='kaiser_fast')

    # Setting the size of the image
    fig = plt.figure(figsize=[1, 1])

    # This is to get rid of the axes and only get the picture
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)

    # This is the melspectrogram from the decibels with a linear relationship
    # Setting min and max frequency to account for human voice frequency
    S = librosa.feature.melspectrogram(y=X, sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max), x_axis='time', y_axis='mel', fmin=50, fmax=280)

    # Here we choose the path and the name to save the file, we will change the path when
    # using the function for train, val and test to make the function easy to use and output
    # the images in different folders to use later with a generator
    name = files.file

    file = 'voice_images_test_new/' + str(name) + '.jpg'

    # Here we finally save the image file choosing the resolution
    plt.savefig(file, dpi=500, bbox_inches='tight', pad_inches=0)

    # Here we close the image because otherwise we get a warning saying that the image stays
    # open and consumes memory
    plt.close()


# gc.collect()


def extract_features(files):
    # Sets the name to be the path to where the file is in my computer
    # file_name = os.path.join(os.path.abspath('new_test_true_false') + '/' + str(files.file)) #origin
    file_name = os.path.join(os.path.abspath('data') + '/DeceptionDB/' + str(files.file)) #VAD
    #file_name = os.path.join(os.path.abspath('data') + '/claimed/' + str(files.file)) #claimed
    #file_name = 'C:/Users/noaiz/Desktop/claims/DATA/' + str(files.file) # VAD

    # Loads the audio file as a floating point time series and assigns the default sample rate
    # Sample rate is set to 22050 by default
    X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')

    # Generate Mel-frequency cepstral coefficients (MFCCs) from a time series
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)

    # Generates a Short-time Fourier transform (STFT) to use in the chroma_stft
    stft = np.abs(librosa.stft(X))

    # Computes a chromagram from a waveform or power spectrogram.
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)

    # Computes a mel-scaled spectrogram.
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)

    # Computes spectral contrast
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)

    # Computes the tonal centroid features (tonnetz)
    tonnetz = 0
    try:
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
                                                  sr=sample_rate).T, axis=0)
    except:
        print(file_name)

    # We add also the classes of each file as a label at the end
    label = files.label

    return mfccs, chroma, mel, contrast, tonnetz, label


def read_data():
    #### -------------- All (after cleaning - 959) - True/False claims - This is the good one! --------------
    # list the files
    filelist = os.listdir('data/true_claims')
    # read them into pandas
    df_male = pd.DataFrame(filelist)
    # Adding the 1 label to the dataframe representing male
    df_male['label'] = '1'
    # Renaming the column name to file
    df_male = df_male.rename(columns={0: 'file'})

    # Checking for a file that gets automatically generated and we need to drop
    df_male[df_male['file'] == '.DS_Store']

    filelist = os.listdir('data/false_claims')
    # read them into pandas
    df_female = pd.DataFrame(filelist)
    df_female['label'] = '0'
    df_female = df_female.rename(columns={0: 'file'})

    df_female[df_female['file'] == '.DS_Store']
    # Dropping the system file
    # df_female.drop(981, inplace=True)
    # df_female = df_female.reset_index(drop=True)

    df = pd.concat([df_female, df_male], ignore_index=True)

    # Randomizing our files to be able to split into train, validation and test
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    df_train = df[:767]
    df_train['label'].value_counts(normalize=True)
    df_validation = df[767:959]
    df_validation['label'].value_counts(normalize=True)
    return df_train, df_validation

    # ### -------------- Original- True/False claims (789) --------------
    # # list the files
    # filelist = os.listdir('data/true_claims')
    # # read them into pandas
    # df_male = pd.DataFrame(filelist)
    # # Adding the 1 label to the dataframe representing male
    # df_male['label'] = '1'
    # # Renaming the column name to file
    # df_male = df_male.rename(columns={0: 'file'})
    #
    # # Checking for a file that gets automatically generated and we need to drop
    # df_male[df_male['file'] == '.DS_Store']
    #
    # filelist = os.listdir('data/false_claims')
    # # read them into pandas
    # df_female = pd.DataFrame(filelist)
    # df_female['label'] = '0'
    # df_female = df_female.rename(columns={0: 'file'})
    #
    # df_female[df_female['file'] == '.DS_Store']
    # # Dropping the system file
    # # df_female.drop(981, inplace=True)
    # # df_female = df_female.reset_index(drop=True)
    #
    # df = pd.concat([df_female, df_male], ignore_index=True)
    #
    # # Randomizing our files to be able to split into train, validation and test
    # df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    # df_train = df[:892]
    # df_train['label'].value_counts(normalize=True)
    # df_validation = df[892:1127]
    # df_validation['label'].value_counts(normalize=True)
    # return df_train, df_validation

    # #### -------------- All (before cleaning - 1127) - True/False claims (I have 1120) --------------
    # # list the files
    # filelist = os.listdir('data/true_claims')
    # # read them into pandas
    # df_male = pd.DataFrame(filelist)
    # # Adding the 1 label to the dataframe representing male
    # df_male['label'] = '1'
    # # Renaming the column name to file
    # df_male = df_male.rename(columns={0: 'file'})
    #
    # # Checking for a file that gets automatically generated and we need to drop
    # df_male[df_male['file'] == '.DS_Store']
    #
    # filelist = os.listdir('data/false_claims')
    # # read them into pandas
    # df_female = pd.DataFrame(filelist)
    # df_female['label'] = '0'
    # df_female = df_female.rename(columns={0: 'file'})
    #
    # df_female[df_female['file'] == '.DS_Store']
    # # Dropping the system file
    # # df_female.drop(981, inplace=True)
    # # df_female = df_female.reset_index(drop=True)
    #
    # df = pd.concat([df_female, df_male], ignore_index=True)
    #
    # # Randomizing our files to be able to split into train, validation and test
    # df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    # df_train = df[:631]
    # df_train['label'].value_counts(normalize=True)
    # df_validation = df[631:789]
    # df_validation['label'].value_counts(normalize=True)
    # return df_train, df_validation

    # # -------------- list the files -ENGLISH- only!! --------------
    # filelist = os.listdir('data/eng_claims/TRUE')
    # # read them into pandas
    # df_male = pd.DataFrame(filelist)
    # # Adding the 1 label to the dataframe representing male
    # df_male['label'] = '1'
    # # Renaming the column name to file
    # df_male = df_male.rename(columns={0: 'file'})
    #
    # # Checking for a file that gets automatically generated and we need to drop
    # df_male[df_male['file'] == '.DS_Store']
    #
    # filelist = os.listdir('data/eng_claims/FALSE')
    # # read them into pandas
    # df_female = pd.DataFrame(filelist)
    # df_female['label'] = '0'
    # df_female = df_female.rename(columns={0: 'file'})
    #
    # df_female[df_female['file'] == '.DS_Store']
    # # Dropping the system file
    # # df_female.drop(981, inplace=True)
    # # df_female = df_female.reset_index(drop=True)
    #
    # df = pd.concat([df_female, df_male], ignore_index=True)
    #
    # # Randomizing our files to be able to split into train, validation and test
    # df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    # df_train = df[:448]
    # df_train['label'].value_counts(normalize=True)
    # df_validation = df[448:561]
    # df_validation['label'].value_counts(normalize=True)

    # # -------------- list the files- HEBREW only --------------
    # filelist = os.listdir('data/heb_claims/TRUE')
    # # read them into pandas
    # df_male = pd.DataFrame(filelist)
    # # Adding the 1 label to the dataframe representing male
    # df_male['label'] = '1'
    # # Renaming the column name to file
    # df_male = df_male.rename(columns={0: 'file'})
    #
    # # Checking for a file that gets automatically generated and we need to drop
    # df_male[df_male['file'] == '.DS_Store']
    #
    # filelist = os.listdir('data/heb_claims/FALSE')
    # # read them into pandas
    # df_female = pd.DataFrame(filelist)
    # df_female['label'] = '0'
    # df_female = df_female.rename(columns={0: 'file'})
    #
    # df_female[df_female['file'] == '.DS_Store']
    # # Dropping the system file
    # # df_female.drop(981, inplace=True)
    # # df_female = df_female.reset_index(drop=True)
    #
    # df = pd.concat([df_female, df_male], ignore_index=True)
    #
    # # Randomizing our files to be able to split into train, validation and test
    # df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    # cv = KFold(n_splits=5, random_state=1, shuffle=True)
    # df_train = df[:318]
    # df_train['label'].value_counts(normalize=True)
    # df_validation = df[318:398]
    # df_validation['label'].value_counts(normalize=True)

    # -------------- list the files - train- hebrew , test- english --------------
    # filelist_train_true = os.listdir('data/heb_claims/TRUE')
    # # read them into pandas
    # df_male = pd.DataFrame(filelist_train_true)
    # # Adding the 1 label to the dataframe representing male
    # df_male['label'] = '1'
    # # Renaming the column name to file
    # df_male = df_male.rename(columns={0: 'file'})
    #
    # # Checking for a file that gets automatically generated and we need to drop
    # df_male[df_male['file'] == '.DS_Store']
    #
    # filelist_train_false = os.listdir('data/heb_claims/FALSE')
    # # read them into pandas
    # df_female = pd.DataFrame(filelist_train_false)
    # df_female['label'] = '0'
    # df_female = df_female.rename(columns={0: 'file'})
    #
    # df_female[df_female['file'] == '.DS_Store']
    # # Dropping the system file
    # # df_female.drop(981, inplace=True)
    # # df_female = df_female.reset_index(drop=True)
    #
    # df = pd.concat([df_female, df_male], ignore_index=True)
    #
    # # Randomizing our files to be able to split into train, validation and test
    # df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    # cv = KFold(n_splits=5, random_state=1, shuffle=True)
    # df_train = df
    # df_train['label'].value_counts(normalize=True)
    # # /////////////////////////////
    # filelist_test_true = os.listdir('data/eng_claims/TRUE')
    # # read them into pandas
    # df_male2 = pd.DataFrame(filelist_test_true)
    # # Adding the 1 label to the dataframe representing male
    # df_male2['label'] = '1'
    # # Renaming the column name to file
    # df_male2 = df_male2.rename(columns={0: 'file'})
    #
    # # Checking for a file that gets automatically generated and we need to drop
    # df_male2[df_male2['file'] == '.DS_Store']
    #
    # filelist_test_false = os.listdir('data/eng_claims/FALSE')
    # # read them into pandas
    # df_female2 = pd.DataFrame(filelist_test_false)
    # df_female2['label'] = '0'
    # df_female2 = df_female2.rename(columns={0: 'file'})
    #
    # df_female2[df_female2['file'] == '.DS_Store']
    # # Dropping the system file
    # # df_female.drop(981, inplace=True)
    # # df_female = df_female.reset_index(drop=True)
    #
    # df2 = pd.concat([df_female2, df_male2], ignore_index=True)
    #
    # # Randomizing our files to be able to split into train, validation and test
    # df2 = df2.sample(frac=1, random_state=SEED).reset_index(drop=True)
    # cv = KFold(n_splits=5, random_state=1, shuffle=True)
    # df_validation = df2
    # df_validation['label'].value_counts(normalize=True)

    # # -------------- list the files - train- hebrew& english , test- hebrew- in the game scope --------------
    # filelist_train_true = os.listdir('data/heb_claims/TRUE')
    # filelist_train_trueENG = os.listdir('data/eng_claims/TRUE')
    # # read them into pandas
    # df_male = pd.DataFrame(filelist_train_true)
    # df_maleENG = pd.DataFrame(filelist_train_trueENG)
    # # Adding the 1 label to the dataframe representing male
    # df_male['label'] = '1'
    # df_maleENG['label'] = '1'
    # # Renaming the column name to file
    # df_male = df_male.rename(columns={0: 'file'})
    # df_maleENG = df_maleENG.rename(columns={0: 'file'})
    #
    # # Checking for a file that gets automatically generated and we need to drop
    # df_male[df_male['file'] == '.DS_Store']
    # df_maleENG[df_maleENG['file'] == '.DS_Store']
    #
    # filelist_train_false = os.listdir('data/heb_claims/FALSE')
    # filelist_train_falseENG = os.listdir('data/eng_claims/FALSE')
    # # read them into pandas
    # df_female = pd.DataFrame(filelist_train_false)
    # df_female['label'] = '0'
    # df_female = df_female.rename(columns={0: 'file'})
    #
    # df_female[df_female['file'] == '.DS_Store']
    #
    # df_femaleENG = pd.DataFrame(filelist_train_falseENG)
    # df_femaleENG['label'] = '0'
    # df_femaleENG = df_femaleENG.rename(columns={0: 'file'})
    #
    # df_femaleENG[df_femaleENG['file'] == '.DS_Store']
    # # Dropping the system file
    # # df_female.drop(981, inplace=True)
    # # df_female = df_female.reset_index(drop=True)
    #
    # df = pd.concat([df_female, df_male], ignore_index=True)  # HEBREW- TRAIN
    # dfENG = pd.concat([df_femaleENG, df_maleENG], ignore_index=True)  # ENGLISH
    # dfENG = dfENG.sample(frac=1, random_state=SEED).reset_index(drop=True)
    # df = pd.concat([df, dfENG[:318]], ignore_index=True)
    # # Randomizing our files to be able to split into train, validation and test
    # df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)  #train
    # df_train = df
    # df_train['label'].value_counts(normalize=True)
    # df_validation = dfENG[318:398]
    # df_validation['label'].value_counts(normalize=
    #
    # # --------------list the files - train- hebrew& english , test- english- in the game scope--------------
    # filelist_train_true = os.listdir('data/heb_claims/TRUE')
    # filelist_train_trueENG = os.listdir('data/eng_claims/TRUE')
    # # read them into pandas
    # df_male = pd.DataFrame(filelist_train_true)
    # df_maleENG = pd.DataFrame(filelist_train_trueENG)
    # # Adding the 1 label to the dataframe representing male
    # df_male['label'] = '1'
    # df_maleENG['label'] = '1'
    # # Renaming the column name to file
    # df_male = df_male.rename(columns={0: 'file'})
    # df_maleENG = df_maleENG.rename(columns={0: 'file'})
    #
    # # Checking for a file that gets automatically generated and we need to drop
    # df_male[df_male['file'] == '.DS_Store']
    # df_maleENG[df_maleENG['file'] == '.DS_Store']
    #
    # filelist_train_false = os.listdir('data/heb_claims/FALSE')
    # filelist_train_falseENG = os.listdir('data/eng_claims/FALSE')
    # # read them into pandas
    # df_female = pd.DataFrame(filelist_train_false)
    # df_female['label'] = '0'
    # df_female = df_female.rename(columns={0: 'file'})
    #
    # df_female[df_female['file'] == '.DS_Store']
    #
    # df_femaleENG = pd.DataFrame(filelist_train_falseENG)
    # df_femaleENG['label'] = '0'
    # df_femaleENG = df_femaleENG.rename(columns={0: 'file'})
    #
    # df_femaleENG[df_femaleENG['file'] == '.DS_Store']
    # # Dropping the system file
    # # df_female.drop(981, inplace=True)
    # # df_female = df_female.reset_index(drop=True)
    #
    # df = pd.concat([df_female, df_male], ignore_index=True)  # HEBREW- TRAIN
    # dfENG = pd.concat([df_femaleENG, df_maleENG], ignore_index=True)  # ENGLISH
    # dfENG = dfENG.sample(frac=1, random_state=SEED).reset_index(drop=True)
    # df = pd.concat([df, dfENG[:369]], ignore_index=True)
    # # Randomizing our files to be able to split into train, validation and test
    # df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)  # train
    # df_train = df
    # df_train['label'].value_counts(normalize=True)
    # df_validation = dfENG[369:561]
    # df_validation['label'].value_counts(normalize=True)
    return df_train, df_validation


def make_jpg(files):
    return str(files.split('.')[0]) + '.wav.jpg'


def cnn_preprocess(train, val):
    train.apply(images, axis=1);
    val.apply(images, axis=1);

    train['file'] = train['file'].apply(make_jpg)
    val['file'] = val["file"].apply(make_jpg)
    # Rescaling the images as usual to feed into the CNN
    datagen = ImageDataGenerator(rescale=1. / 255.)

    train_generator = datagen.flow_from_dataframe(
        dataframe=train,
        directory="voice_images_test_new",
        x_col="file",
        y_col="label",
        shuffle=False,
        class_mode="categorical",
        target_size=(64, 64))

    val_generator = datagen.flow_from_dataframe(
        dataframe=val,
        directory="voice_images_test_new",
        x_col="file",
        y_col="label",
        shuffle=False,
        class_mode="categorical",
        target_size=(64, 64))

    #plt.close()
    # gc.collect()!!!!!
    return train_generator, val_generator


def nn_preprocess_step(df, name=""):
    startTime = datetime.now()
    print("Starting featureextraction at: ", startTime)
    features_label = df.apply(extract_features, axis=1)
    print("done extracting. took: ", datetime.now() - startTime)

    # Saving the numpy array because it takes a long time to extract the features
    np.save(name, features_label)

    # loading the features
    features_label = np.load(name + '.npy', allow_pickle=True)
    # We create an empty list where we will concatenate all the features into one long feature
    # for each file to feed into our neural network
    features = []
    labels = []
    for i in range(0, len(features_label)):
        try:
            features.append(np.concatenate((features_label[i][0], features_label[i][1],
                                            features_label[i][2], features_label[i][3],
                                            features_label[i][4]), axis=0))
            labels.append(features_label[i][5])
        except:
            print("feature ", i, "didnt work")

    print(len(features))
    print(len(labels))

    np.unique(labels, return_counts=True)

    # Setting our X as a numpy array to feed into the neural network
    X = np.array(features)
    # Setting our y
    y = np.array(labels)

    # Hot encoding y
    lb = LabelEncoder()
    y = to_categorical(lb.fit_transform(y))

    return X, y


def nn_preprocess(train, val):
    X_train, y_train = nn_preprocess_step(train, "train_features")
    X_val, y_val = nn_preprocess_step(val, "val_features")

    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_val = ss.transform(X_val)

    return X_train, y_train, X_val, y_val


def trainCNN(df_tarin, df_val):
    checkpoint = ModelCheckpoint("CNN_best_model.hdf5", monitor='val_accuracy', verbose=1,
                                 save_best_only=True, mode='auto', period=1)
    cnn_train_generator, cnn_val_generator = cnn_preprocess(df_tarin.copy(), df_val.copy())
    cnn_model_ , es_cb = CNN_model()
    cnn_history = cnn_model_.fit_generator(generator=cnn_train_generator,
                                           steps_per_epoch=24,
                                           validation_data=cnn_val_generator,
                                           validation_steps=5,
                                           epochs=100,
                                           callbacks=[es_cb, checkpoint]
                                           )


    # # HEBREW only
    # cnn_history = cnn_model_.fit_generator(generator=cnn_train_generator,
    #                                        steps_per_epoch=10,
    #                                        validation_data=cnn_val_generator,
    #                                        validation_steps=3,
    #                                        epochs=200)

    ## ENGLISH only
    # cnn_history = cnn_model_.fit_generator(generator=cnn_train_generator,
    #                                        steps_per_epoch=14,
    #                                        validation_data=cnn_val_generator,
    #                                        validation_steps=4,
    #                                        epochs=100)

    # # Train- Hebrew Test- English
    # cnn_history = cnn_model_.fit_generator(generator=cnn_train_generator,
    #                                        steps_per_epoch=13,
    #                                        validation_data=cnn_val_generator,
    #                                        validation_steps=7,
    #                                        epochs=100)
    #
    #

    # # Train- English test- English
    # cnn_history = cnn_model_.fit_generator(generator=cnn_train_generator,
    #                                        steps_per_epoch=20,
    #                                        validation_data=cnn_val_generator,
    #                                        validation_steps=4,
    #                                        epochs=150)


    cnn_train_accuracy = cnn_history.history['accuracy']
    cnn_val_accuracy = cnn_history.history['val_accuracy']

    plt.plot(cnn_train_accuracy, label='CNN Training Accuracy', color='black')
    plt.plot(cnn_val_accuracy, label='CNN Validation Accuracy', color='red')

    preds_cnn = cnn_model_.predict_generator(cnn_val_generator)
    predicted_labels = np.argmax(preds_cnn, axis=1)
    predic_cnn = pd.DataFrame(preds_cnn)
    predic_cnn.insert(0, 'predicted-label', predicted_labels)
    predic_cnn.insert(0, 'real-label', cnn_val_generator.labels)
    predic_cnn.insert(0, 'file-name', df_val.file.to_numpy())
   # predic_cnn = pd.concat([predic_cnn, df_val], axis=1)
    predic_cnn.to_csv('predict_cnn.csv', index=False)
    cnn_model_.save("CNN_end")
    return cnn_history


def trainCRNN(df_tarin, df_val):
    checkpoint = ModelCheckpoint("CRNN_best_model.hdf5", monitor='val_accuracy', verbose=1,
                                 save_best_only=True, mode='auto', period=1)
    crnn_train_generator, crnn_val_generator = cnn_preprocess(df_tarin.copy(), df_val.copy())
    crnn_model, es_cb = RNN_model()
    print("training crnn")
    # crnn_history = crnn_model.fit_generator(generator=crnn_train_generator,
    #                 steps_per_epoch=28,
    #                 validation_data=crnn_val_generator,
    #                 validation_steps=7,
    #                 epochs=100)
    # crnn_history = crnn_model.fit_generator(generator=crnn_train_generator,
    #                                         steps_per_epoch=20,
    #                                         validation_data=crnn_val_generator,
    #                                         validation_steps=5,
    #                                         epochs=100)
    # BEST!!
    crnn_history = crnn_model.fit_generator(generator=crnn_train_generator,
                                            steps_per_epoch=18,
                                            validation_data=crnn_val_generator,
                                            validation_steps=5,
                                            epochs=100,
                                            callbacks= [es_cb, checkpoint]
                                            )

    # crnn_history = crnn_model.fit_generator(generator=crnn_train_generator, ENGLISH
    #                                         steps_per_epoch=14,
    #                                         validation_data=crnn_val_generator,
    #                                         validation_steps=4,
    #                                         epochs=100)

    # #HEBREW
    # crnn_history = crnn_model.fit_generator(generator=crnn_train_generator,
    #                                         steps_per_epoch=14,
    #                                         validation_data=crnn_val_generator,
    #                                         validation_steps=4,
    #                                         epochs=100)

    # # train- HEBREW test- english
    # crnn_history = crnn_model.fit_generator(generator=crnn_train_generator,
    #                                         steps_per_epoch=13,
    #                                         validation_data=crnn_val_generator,
    #                                         validation_steps=7,
    #                                         epochs=100)
    #


    # train- HEBREW & ENGLISH test- ENGLISH
    # crnn_history = crnn_model.fit_generator(generator=crnn_train_generator,
    #                                         steps_per_epoch=24,
    #                                         validation_data=crnn_val_generator,
    #                                         validation_steps=5,
    #                                         epochs=150)
    crnn_train_accuracy = crnn_history.history['accuracy']
    crnn_val_accuracy = crnn_history.history['val_accuracy']
    plt.plot(crnn_train_accuracy, label='CRNN Training Accuracy', color='green')
    plt.plot(crnn_val_accuracy, label='CRNN Validation Accuracy', color='brown')
    print("CRNN:" + str(max(crnn_val_accuracy)))

    preds_rnn = crnn_model.predict_generator(crnn_val_generator)
    predicted_labels = np.argmax(preds_rnn, axis=1)
    predic_rnn = pd.DataFrame(preds_rnn)
    predic_rnn.insert(0, 'predicted-label', predicted_labels)
    predic_rnn.insert(0,'real-label',crnn_val_generator.labels)
    predic_rnn.insert(0, 'file-name', df_val.file.to_numpy())
    #predic_rnn = pd.concat([predic_rnn, df_val], axis=1)
    predic_rnn.to_csv('predict_rnn.csv', index=False)
    crnn_model.save("CRNN_end")
    return crnn_history


def trainNN(df_tarin, df_val):
    checkpoint = ModelCheckpoint("NN_best_model.hdf5", monitor='val_accuracy', verbose=1,
                                 save_best_only=True, mode='auto', period=1)
    X_train, y_train, X_val, y_val = nn_preprocess(df_tarin.copy(), df_val.copy())
    nn_model_, es_cb = NN_model()
    nn_history = nn_model_.fit(X_train, y_train, batch_size=256, epochs=100,
                        validation_data=(X_val, y_val),
                        callbacks=[es_cb,checkpoint])



    nn_train_accuracy = nn_history.history['accuracy']
    nn_val_accuracy = nn_history.history['val_accuracy']
    plt.plot(nn_train_accuracy, label='NN Training Accuracy', color='#185fad')
    plt.plot(nn_val_accuracy, label='NN Validation Accuracy', color='orange')
    # plt.title("NN")
    # plt.show()

    predic_nn = nn_model_.predict_proba(X_val)
    predicted_labels = np.argmax(predic_nn, axis=1)
    predic_nn = pd.DataFrame(predic_nn)

    predic_nn.insert(0, 'predicted-label', predicted_labels)
    predic_nn.insert(0,'real-label',np.argmax(y_val, axis=1))
    predic_nn.insert(0, 'file-name', df_val.file.to_numpy())


    predic_nn.to_csv('predict_nn.csv', index=False)
    nn_model_.save('NN_end')
    print("max score: ", max(nn_val_accuracy))
    return


def main():
    random.seed(SEED);
    df_tarin, df_val = read_data()
    trainNN(df_tarin, df_val)
    trainCRNN(df_tarin, df_val)
    trainCNN(df_tarin, df_val)


    # Set title
    plt.title('Training and Validation Accuracy by Epoch', fontsize=25)
    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel('Categorical Crossentropy', fontsize=18)
    plt.xticks(range(0, 100, 5), range(0, 100, 5))

    plt.legend(fontsize=18)
    plt.show()


if __name__ == "__main__":
    main()
