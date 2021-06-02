import numpy as np
import pandas as pd

# Read in the results from our CNN
preds_cnn = pd.read_csv('predict_cnn.csv')
# Read in the results from our Dense NN
preds_nn = pd.read_csv('predict_nn.csv')
preds_rnn = pd.read_csv('predict_rnn.csv')

# Convering the results to arrays
preds_cnn_array = preds_cnn.to_numpy()
preds_nn_array = preds_nn.to_numpy()
preds_rnn_array = preds_rnn.to_numpy()

# Adding the porcentages of predictions together
preds = (preds_cnn_array + preds_nn_array + preds_rnn_array)
features_label = np.load('val_features.npy', allow_pickle=True)

features_df = pd.DataFrame(features_label)
features_df.to_csv('features_df.csv', index=False)

test = pd.read_csv('features_df.csv')
# Creating an empty list to store the values where the predictions are the maximum out
# of all the 10 possible values
p = []
cnn_p = []
nn_p = []
rnn_p = []
for i in range(0, len(preds)):
    p.append(np.where(preds[i] == max(preds[i])))
    cnn_p.append(np.where(preds_cnn_array[i] == max(preds_cnn_array[i])))
    nn_p.append(np.where(preds_nn_array[i] == max(preds_nn_array[i])))
    rnn_p.append(np.where(preds_rnn_array[i] == max(preds_rnn_array[i])))

# Creating an empty list to store the values in a clean list
predictions = []
for i in range(0, len(preds)):
    predictions.append(p[i][0][0])


s = [10,20,30,40,50,60,70,80,90,100,110,120,130,140,146]
for samples in range(len(s)):
    Pcorect = 0
    cnn_corect = 0
    nn_corect = 0
    rnn_corect = 0
    for i in range(s[samples]):
        if int(test.values[i][-1][-3]) == int(p[i][0][0]):
            Pcorect += 1
        if int(test.values[i][-1][-3]) == int(cnn_p[i][0][0]):
            cnn_corect += 1
        if int(test.values[i][-1][-3]) == int(nn_p[i][0][0]):
            nn_corect += 1
        if int(test.values[i][-1][-3]) == int(rnn_p[i][0][0]):
            rnn_corect += 1
    print("joint model at " , s[samples] , "samples: " , round(Pcorect/s[samples],2))
    print("cnn model at " , s[samples] , "samples: " , round(cnn_corect/s[samples],2))
    print("NN model at " , s[samples] , "samples: " , round(nn_corect/s[samples],2))
    print("RNN model at " , s[samples] , "samples: " , round(rnn_corect/s[samples],2))
#print("test:")
#for i in range(0,142):
 #   print(test.values[i][-1][-3])



