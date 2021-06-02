import os
import random
from random import seed

import numpy
import pandas as pd
from keras_preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np

import portfolio_model


def get_results(model,X,Y,test):
    print('Confusion Matrix')
    y_pred = model.predict(X[test], verbose=0)
    labels= np.unique(y_pred.argmax(axis=1))
    cm = confusion_matrix(Y[test].argmax(axis=1), y_pred.argmax(axis=1), labels=labels)
    print(pd.DataFrame(cm, index=labels, columns=labels))
    Y_test = np.argmax(Y[test], axis=1)
    y_pred = np.argmax(y_pred, axis=1)
    from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
    print(f"Accuracy: {round(accuracy_score(Y_test, y_pred), 2)}")
    print(f"Precision: {round(precision_score(Y_test, y_pred), 2)}")
    print(f"Recall: {round(recall_score(Y_test, y_pred), 2)}")
    print(f"F1_score: {round(f1_score(Y_test, y_pred), 2)}\n\n")
    return cm

def NN(df):

    X, Y = portfolio_model.nn_preprocess_step(df, "test_features")
    ss = StandardScaler()
    X = ss.fit_transform(X)

    # model_path: str = "NN_end"
    # model = load_model(model_path, compile=True, custom_objects=None)

    # define 5-fold cross validation test harness
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    cvscores = []
    models = [[0,0],[0,0]]
    for train, test in kfold.split(X, np.argmax(Y, axis=1)):
        model, _ = portfolio_model.NN_model()
        model.fit(X[train], Y[train], epochs=1000, batch_size=40,verbose=0)
        # evaluate the model
        scores = model.evaluate(X[test], Y[test], verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
        cvscores.append(scores[1] * 100)
        models += get_results(model,X,Y,test)

    print(models)
    TP = models[0][0]
    FP=models[1][0]
    FN=models[0][1]
    TN = models[1][1]
    print(f"Accuracy: {(TP+TN)/(TP+TN+FP+FN)}")
    print(f"Precision: {(TP)/(TP+FP)}")
    print(f"Recall: {TP/(TP+FN)}")
    print(f"F1_score: {(2*TP/(TP+FN)*(TP)/(TP+FP))/(TP/(TP+FN)+(TP)/(TP+FP))}\n\n")
    print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))


def NN_2_lan(df_train , df_test):

    X, Y = portfolio_model.nn_preprocess_step(df_train, "train_features")
    ss = StandardScaler()
    X = ss.fit_transform(X)

    X2, Y2 = portfolio_model.nn_preprocess_step(df_test, "test_features")
    ss = StandardScaler()
    X2 = ss.fit_transform(X2)

    # model_path: str = "NN_end"
    # model = load_model(model_path, compile=True, custom_objects=None)

    # define 5-fold cross validation test harness
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    cvscores = []
    models = [[0,0],[0,0]]
    for train, test in kfold.split(X2,np.argmax(Y2, axis=1)):
        print(test.size)
        model, _ = portfolio_model.NN_model()
        model.fit(X2[train], Y2[train], epochs=1000, batch_size=400,verbose=0)
        # evaluate the model
        scores = model.evaluate(X[test], Y[test], verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
        cvscores.append(scores[1] * 100)
        models += get_results(model,X,Y,test)

    print(models)
    TP = models[0][0]
    FP=models[1][0]
    FN=models[0][1]
    TN = models[1][1]
    print(f"Accuracy: {(TP+TN)/(TP+TN+FP+FN)}")
    print(f"Precision: {(TP)/(TP+FP)}")
    print(f"Recall: {TP/(TP+FN)}")
    print(f"F1_score: {(2*TP/(TP+FN)*(TP)/(TP+FP))/(TP/(TP+FN)+(TP)/(TP+FP))}\n\n")
    print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))

def concatenate_per_row(A, B):

    m1 = len(A)
    m2 = len(B)
    n=2
    out = np.zeros((m1,n),dtype=A.dtype)
    out[:,:1] = A[:,None]
    out[:,1:] = np.array(B).reshape(len(B),1)
    return out
def CNN(df):
   # df.apply(portfolio_model.images, axis=1);
    df['file'] = df["file"].apply(portfolio_model.make_jpg)
    # Rescaling the images as usual to feed into the CNN
    datagen = ImageDataGenerator(rescale=1. / 255.)
    # df_generator = datagen.flow_from_dataframe(
    #     dataframe=df,
    #     directory="voice_images_test_new",
    #     x_col="file",
    #     y_col="label",
    #     shuffle=False,
    #     class_mode="categorical",
    #     target_size=(64, 64))
    # define 5-fold cross validation test harness
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    #kfold.get_n_splits(df_generator)
    X= df.file
    Y= df.label
    # X = np.array(df_generator.filepaths)
    # Y = np.array(df_generator.labels)

    cvscores = []
    models = [[0,0],[0,0]]

    # print(df_generator.filepaths)
    # X = df_generator.filepaths
    # Y = df_generator.labels
    for train, test in kfold.split(X,Y):
        # trainData = X[train]
        # testData = X[test]
        # trainLabels = Y[train]
        # testLabels = Y[test]
        #trainGenerator = Generator(trainData, trainLabels, batchSize=24, imageSize=(64, 64), augment=True)
        # valGenerator = ImageDataGenerator(testData, testLabels, batchSize=5, imageSize=(64, 64), augment=False)
        #train = concatenate_per_row(trainData, trainLabels)
        #test_df = concatenate_per_row(testData,testLabels)

        testData = datagen.flow_from_dataframe(
            dataframe=df.iloc[test],
            directory="voice_images_test_new",
            x_col="file",
            y_col="label",
            shuffle=False,
            class_mode="categorical",
            target_size=(64, 64))

        trainData = datagen.flow_from_dataframe(
            dataframe=df.iloc[train],
            directory="voice_images_test_new",
            x_col="file",
            y_col="label",
            shuffle=False,
            class_mode="categorical",
            target_size=(64, 64))

        model, _ = portfolio_model.CNN_model()
        model.fit_generator(
            trainData,
            steps_per_epoch=len(trainData),
            epochs=1000,
            validation_data=testData,
            validation_steps=len(testData))

        # scores =model.fit_generator(generator=X[train],
        #                          steps_per_epoch=24,
        #                          validation_data=X[test],
        #                          validation_steps=5,
        #                          epochs=100,
        #                          )
        # evaluate the model

        scores = model.evaluate(testData, verbose=2)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
        cvscores.append(scores[1] * 100)
        print('Restored model, accuracy: {:5.2f}%'.format(100 * scores[1]))
        y_pred = model.predict_generator(testData)
        y_pred = np.argmax(y_pred, axis=1)
        y_test = testData.labels
        print('Confusion Matrix')
        labels = [0,1]
        cm = confusion_matrix(testData.labels, y_pred, labels=labels)
        models += cm
        print(pd.DataFrame(cm, index=labels, columns=labels))
        from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
        print(f"Accuracy: {round(accuracy_score(y_test, y_pred), 2)}")
        print(f"Precision: {round(precision_score(y_test, y_pred), 2)}")
        print(f"Recall: {round(recall_score(y_test, y_pred), 2)}")
        print(f"F1_score: {round(f1_score(y_test, y_pred), 2)}")
    print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))


    print(models)
    TP = models[0][0]
    FP = models[1][0]
    FN = models[0][1]
    TN = models[1][1]
    print(f"Accuracy: {(TP + TN) / (TP + TN + FP + FN)}")
    print(f"Precision: {(TP) / (TP + FP)}")
    print(f"Recall: {TP / (TP + FN)}")
    print(f"F1_score: {(2 * TP / (TP + FN) * (TP) / (TP + FP)) / (TP / (TP + FN) + (TP) / (TP + FP))}\n\n")
    print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))

def CRNN(df):
   # df.apply(portfolio_model.images, axis=1);
    df['file'] = df["file"].apply(portfolio_model.make_jpg)
    # Rescaling the images as usual to feed into the CNN
    datagen = ImageDataGenerator(rescale=1. / 255.)

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    X= df.file
    Y= df.label
    cvscores = []
    models = [[0,0],[0,0]]

    for train, test in kfold.split(X,Y):
        testData = datagen.flow_from_dataframe(
            dataframe=df.iloc[test],
            directory="voice_images_test_new",
            x_col="file",
            y_col="label",
            shuffle=False,
            class_mode="categorical",
            target_size=(64, 64))

        trainData = datagen.flow_from_dataframe(
            dataframe=df.iloc[train],
            directory="voice_images_test_new",
            x_col="file",
            y_col="label",
            shuffle=False,
            class_mode="categorical",
            target_size=(64, 64))

        model, _ = portfolio_model.RNN_model()
        model.fit_generator(
            trainData,
            steps_per_epoch=len(trainData),
            epochs=1000,
            validation_data=testData,
            validation_steps=len(testData))

        scores = model.evaluate(testData, verbose=2)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
        cvscores.append(scores[1] * 100)
        print('Restored model, accuracy: {:5.2f}%'.format(100 * scores[1]))
        y_pred = model.predict_generator(testData)
        y_pred = np.argmax(y_pred, axis=1)
        y_test = testData.labels
        print('Confusion Matrix')
        labels = [0, 1]
        cm = confusion_matrix(testData.labels, y_pred, labels=labels)
        models += cm
        print(pd.DataFrame(cm, index=labels, columns=labels))
        from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
        print(f"Accuracy: {round(accuracy_score(y_test, y_pred), 2)}")
        print(f"Precision: {round(precision_score(y_test, y_pred), 2)}")
        print(f"Recall: {round(recall_score(y_test, y_pred), 2)}")
        print(f"F1_score: {round(f1_score(y_test, y_pred), 2)}")


    print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))

    print(models)
    TP = models[0][0]
    FP = models[1][0]
    FN = models[0][1]
    TN = models[1][1]
    print(f"Accuracy: {(TP + TN) / (TP + TN + FP + FN)}")
    print(f"Precision: {(TP) / (TP + FP)}")
    print(f"Recall: {TP / (TP + FN)}")
    print(f"F1_score: {(2 * TP / (TP + FN) * (TP) / (TP + FP)) / (TP / (TP + FN) + (TP) / (TP + FP))}\n\n")
    print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))

def read_data():

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
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    return df


def read_data_out_of_game():
    # list the files
    filelist = os.listdir('data/claimed_true_claims')
    # read them into pandas
    df_male = pd.DataFrame(filelist)
    # Adding the 1 label to the dataframe representing male
    df_male['label'] = '1'
    # Renaming the column name to file
    df_male = df_male.rename(columns={0: 'file'})

    # Checking for a file that gets automatically generated and we need to drop
    df_male[df_male['file'] == '.DS_Store']

    filelist = os.listdir('data/claimed_false_claims')
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
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df['label'].value_counts(normalize=True)
    return df


def read_data_heb_heb():
    # list the files
    filelist = os.listdir('data/heb_claims/TRUE/')
    # read them into pandas
    df_male = pd.DataFrame(filelist)
    # Adding the 1 label to the dataframe representing male
    df_male['label'] = '1'
    # Renaming the column name to file
    df_male = df_male.rename(columns={0: 'file'})

    # Checking for a file that gets automatically generated and we need to drop
    df_male[df_male['file'] == '.DS_Store']

    filelist = os.listdir('data/heb_claims/FALSE/')
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
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df['label'].value_counts(normalize=True)
    return df


def read_data_eng_eng():
    # list the files
    filelist = os.listdir('data/eng_claims/TRUE/')
    # read them into pandas
    df_male = pd.DataFrame(filelist)
    # Adding the 1 label to the dataframe representing male
    df_male['label'] = '1'
    # Renaming the column name to file
    df_male = df_male.rename(columns={0: 'file'})

    # Checking for a file that gets automatically generated and we need to drop
    df_male[df_male['file'] == '.DS_Store']

    filelist = os.listdir('data/eng_claims/FALSE/')
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
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df['label'].value_counts(normalize=True)
    return df

def read_data_eng_heb():


    filelist_train_true = os.listdir('data/heb_claims/TRUE')
    # read them into pandas
    df_male = pd.DataFrame(filelist_train_true)
    # Adding the 1 label to the dataframe representing male
    df_male['label'] = '1'
    # Renaming the column name to file
    df_male = df_male.rename(columns={0: 'file'})

    # Checking for a file that gets automatically generated and we need to drop
    df_male[df_male['file'] == '.DS_Store']

    filelist_train_false = os.listdir('data/heb_claims/FALSE')
    # read them into pandas
    df_female = pd.DataFrame(filelist_train_false)
    df_female['label'] = '0'
    df_female = df_female.rename(columns={0: 'file'})

    df_female[df_female['file'] == '.DS_Store']
    # Dropping the system file
    # df_female.drop(981, inplace=True)
    # df_female = df_female.reset_index(drop=True)

    df = pd.concat([df_female, df_male], ignore_index=True)

    # Randomizing our files to be able to split into train, validation and test
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    cv = KFold(n_splits=5, random_state=1, shuffle=True)
    df_train = df
    df_train['label'].value_counts(normalize=True)
    # /////////////////////////////
    filelist_test_true = os.listdir('data/eng_claims/TRUE')
    # read them into pandas
    df_male2 = pd.DataFrame(filelist_test_true)
    # Adding the 1 label to the dataframe representing male
    df_male2['label'] = '1'
    # Renaming the column name to file
    df_male2 = df_male2.rename(columns={0: 'file'})

    # Checking for a file that gets automatically generated and we need to drop
    df_male2[df_male2['file'] == '.DS_Store']

    filelist_test_false = os.listdir('data/eng_claims/FALSE')
    # read them into pandas
    df_female2 = pd.DataFrame(filelist_test_false)
    df_female2['label'] = '0'
    df_female2 = df_female2.rename(columns={0: 'file'})

    df_female2[df_female2['file'] == '.DS_Store']
    # Dropping the system file
    # df_female.drop(981, inplace=True)
    # df_female = df_female.reset_index(drop=True)

    df2 = pd.concat([df_female2, df_male2], ignore_index=True)

    # Randomizing our files to be able to split into train, validation and test
    df2 = df2.sample(frac=1, random_state=42).reset_index(drop=True)
    cv = KFold(n_splits=5, random_state=1, shuffle=True)
    df_validation = df2
    df_validation['label'].value_counts(normalize=True)
    return df_train ,df_validation



def read_data2(): #clean
    true_clean_dir = "C:/Users/noaiz/Desktop/claims/DATA/true_claims/"
    false_clean_dir = "C:/Users/noaiz/Desktop/claims/DATA/false_claims/"
    # list the files
    filelist = os.listdir(true_clean_dir)
    # read them into pandas
    df_male = pd.DataFrame(filelist)
    # Adding the 1 label to the dataframe representing male
    df_male['label'] = '0'
    # Renaming the column name to file
    df_male = df_male.rename(columns={0: 'file'})

    # Checking for a file that gets automatically generated and we need to drop
    df_male[df_male['file'] == '.DS_Store']

    filelist = os.listdir(false_clean_dir)
    # read them into pandas
    df_female = pd.DataFrame(filelist)
    df_female['label'] = '1'
    df_female = df_female.rename(columns={0: 'file'})

    df_female[df_female['file'] == '.DS_Store']
    # Dropping the system file
    # df_female.drop(981, inplace=True)
    # df_female = df_female.reset_index(drop=True)

    df = pd.concat([df_female, df_male], ignore_index=True)
    print(df.shape)
    return df


def main():
    random.seed(1234);
    df = read_data_out_of_game()
   # df_train ,df_test = read_data_eng_heb()

    #NN
    print("---------------------  NN ---------------------")
    NN(df)

    # # NN
    print("---------------------  CNN  ---------------------")
    CNN(df)

    # # NN
    print("---------------------  CRNN  ---------------------")
    CRNN(df)


if __name__ == "__main__":
    main()
