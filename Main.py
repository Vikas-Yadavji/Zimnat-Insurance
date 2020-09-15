# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 22:54:28 2020

@author: vikas
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
import random
from keras.models import load_model
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K
from sklearn.preprocessing import StandardScaler
from keras.optimizers import Adam

def onehot_encoder(feature1, feature2):
    # integer encoding
    feature = np.concatenate((np.array(feature1), np.array(feature2)),
                             axis=0)
    label_encoder = LabelEncoder()
    label_encoder.fit(feature)
    
    feature_integer_encoded = label_encoder.transform(feature)
    # binary encoding
    binary_encoder = OneHotEncoder(sparse=False)
    feature_integer_encoded = feature_integer_encoded.reshape(
        len(feature_integer_encoded),1)
    feature_onehot_encoded = binary_encoder.fit_transform(
        feature_integer_encoded)
    
    return feature_onehot_encoded[0:len(feature1),:]

def form_features(df1, df2, flag):
    
    if flag == 1:
        df1, df2 = df2, df1
        
    df = pd.DataFrame({'day': [1],
                       'month': [7],
                       'year': [2020]})
    df = pd.concat([df]*len(df1['join_date']), ignore_index=True)
    date = pd.to_datetime(df)
    
    
    df1['join_date'] = pd.to_datetime(df1['join_date'])
    join = date.sub(df1['join_date'], axis=0)
    join = np.array(join/np.timedelta64(1, 'D')).reshape(-1,1)
    
    
    sex = df1['sex']
    sex = np.array((sex=='F').astype('int')).reshape(-1,1)
    
    
    birth_year = df1['birth_year']
    age = np.array(2020 - birth_year).reshape(-1,1)
    
    
    ## One-hot encoding
    marital_onehot_encoded = onehot_encoder(df1['marital_status'],
                                            df2['marital_status'])
    
    branch_onehot_encoded = onehot_encoder(df1['branch_code'],
                                           df2['branch_code'])
    
    occ_onehot_encoded = onehot_encoder(df1['occupation_code'],
                                        df2['occupation_code'])
    
    # print(occ_onehot_encoded.shape)
    
    occ_cat_onehot_encoded = onehot_encoder(df1['occupation_category_code'],
                                            df2['occupation_category_code'])
    
    ## Merge Features
    features = np.concatenate((join,sex,age,marital_onehot_encoded,
                        branch_onehot_encoded, occ_onehot_encoded,
                        occ_cat_onehot_encoded), axis=1)
    
    return features

def Remove1s(labels):
    
    for i in range(len(labels)):
        nz_ind = list(np.nonzero(labels[i])[0])
        if np.size(nz_ind) > 1:
            ind = random.randrange(0,np.size(nz_ind))
            labels[i,nz_ind[ind]] = 0
    
    return labels
    
    
def NN_Model(in_dim, classes, final_act):
    model = Sequential()
    
    model.add(Dense(100, input_dim=in_dim, activation='relu'))
    model.add(BatchNormalization())
    
    model.add(Dense(100))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    
    model.add(Dense(classes))
    model.add(Activation(final_act))
    
    return model

df_train = pd.read_csv('Data/Train.csv')
df_test = pd.read_csv('Data/Test.csv')

Y_train = df_train[df_train.columns[-21:]]
Y_train = Y_train.to_numpy()

Y_test = df_test[df_test.columns[-21:]]
Y_test = Y_test.to_numpy()

df_train['marital_status'] = df_train['marital_status'].str.upper()
df_test['marital_status'] = df_test['marital_status'].str.upper()

#%% Training Features

train_features1 = form_features(df_train,df_test,0)

#%% Testing Features

test_features1 = form_features(df_train,df_test,1)

#%% Randomly make zero in train labels

train_features2 = Remove1s(Y_train)
test_features2 = Remove1s(Y_test)
        
X_train = np.concatenate((train_features1,train_features2), axis=1)
X_test = np.concatenate((test_features1,test_features2), axis=1)

#%% Remove Nan

ind1 = np.argwhere(np.isnan(X_train))
X_train = np.delete(X_train, list(ind1[:,0]), 0)
Y_train = np.delete(Y_train, list(ind1[:,0]), 0)

ind2 = np.argwhere(np.isnan(X_test))
X_test = np.delete(X_test, list(ind2[:,0]), 0)
Y_test = np.delete(Y_test, list(ind2[:,0]), 0)

#%% Standardization

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.fit_transform(X_test)
#%% NN Architecture

# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions
EPOCHS = 25
INIT_LR = 1e-3
BS = 240

print("[INFO] compiling model...")
model = NN_Model(295, 21, 'softmax')

# initialize the optimizer (SGD is sufficient)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

# compile the model using binary cross-entropy rather than
# categorical cross-entropy -- this may seem counterintuitive for
# multi-label classification, but keep in mind that the goal here
# is to treat each output label as an independent Bernoulli
# distribution
model.compile(loss="kullback_leibler_divergence", optimizer=opt,
              metrics=["accuracy"])

# train the network
print("[INFO] training network...")
H = model.fit(x=X_train_scaled, y=Y_train, batch_size=BS, epochs=EPOCHS,
              verbose=1)

# save the model to disk
print("[INFO] serializing network...")
model.save('Zindi.model', save_format='h5')

#%% Prediction

model = load_model('Zindi.model')
proba = model.predict(X_test_scaled)

#%% Saving

res = proba.flatten()
res = np.around(res, decimals=4)

Result = pd.DataFrame(res)

csv = pd.read_csv('Data/SampleSubmission.csv')
csv['Label'] = Result
csv.to_csv('SampleSubmission.csv',index=False)

#%% Plots

plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_accuracy")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")