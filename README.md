# Credit_Card_Fraud_Detection

[ ]
# import the important libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import time

from sklearn.metrics import classification_report,accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
[ ]
data = pd.read_csv('/content/drive/MyDrive/creditcard.csv',sep=',')
[ ]
from google.colab import drive
drive.mount('/content/drive')
Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).
[ ]
data.head()

[ ]
# check the shape of the dataset
data.shape
(284807, 31)
[ ]
# we will check if there are any null values in the dataset
data.isnull().sum()
Time      0
V1        0
V2        0
V3        0
V4        0
V5        0
V6        0
V7        0
V8        0
V9        0
V10       0
V11       0
V12       0
V13       0
V14       0
V15       0
V16       0
V17       0
V18       0
V19       0
V20       0
V21       0
V22       0
V23       0
V24       0
V25       0
V26       0
V27       0
V28       0
Amount    0
Class     0
dtype: int64
[ ]
data.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 284807 entries, 0 to 284806
Data columns (total 31 columns):
 #   Column  Non-Null Count   Dtype  
---  ------  --------------   -----  
 0   Time    284807 non-null  float64
 1   V1      284807 non-null  float64
 2   V2      284807 non-null  float64
 3   V3      284807 non-null  float64
 4   V4      284807 non-null  float64
 5   V5      284807 non-null  float64
 6   V6      284807 non-null  float64
 7   V7      284807 non-null  float64
 8   V8      284807 non-null  float64
 9   V9      284807 non-null  float64
 10  V10     284807 non-null  float64
 11  V11     284807 non-null  float64
 12  V12     284807 non-null  float64
 13  V13     284807 non-null  float64
 14  V14     284807 non-null  float64
 15  V15     284807 non-null  float64
 16  V16     284807 non-null  float64
 17  V17     284807 non-null  float64
 18  V18     284807 non-null  float64
 19  V19     284807 non-null  float64
 20  V20     284807 non-null  float64
 21  V21     284807 non-null  float64
 22  V22     284807 non-null  float64
 23  V23     284807 non-null  float64
 24  V24     284807 non-null  float64
 25  V25     284807 non-null  float64
 26  V26     284807 non-null  float64
 27  V27     284807 non-null  float64
 28  V28     284807 non-null  float64
 29  Amount  284807 non-null  float64
 30  Class   284807 non-null  int64  
dtypes: float64(30), int64(1)
memory usage: 67.4 MB
[ ]
# we can check that how many values are present in the 'class' having values as 0 or 1
data['Class'].value_counts()
0    284315
1       492
Name: Class, dtype: int64
[ ]
# dividing the dataframe into fraud and non fraud data
non_fraud=data[data['Class']==0]
fraud=data[data['Class']==1]
[ ]
non_fraud.shape, fraud.shape
((284315, 31), (492, 31))
[ ]
# now we are going to select the 492 non-fraud entries from the dataframe 
non_fraud=non_fraud.sample(fraud.shape[0])
non_fraud.shape
(492, 31)
[ ]
data=fraud.append(non_fraud, ignore_index=True)
data

[ ]
# now let us again check the value counts
data.Class.value_counts()
1    492
0    492
Name: Class, dtype: int64
[ ]
# now dividing the dataframe into dependent and independent varaible
X=data.drop(['Class'], axis=1)
y=data.Class

# check the shape
X.shape, y.shape
((984, 30), (984,))
[ ]
# we will divide the dataset into training and testing dataset
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0, stratify=y)

# check the shape again
X_train.shape,X_test.shape,y_train.shape,y_test.shape
((787, 30), (197, 30), (787,), (197,))
[ ]
X_train

[ ]
# scaler
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.fit_transform(X_test)
[ ]
X_train,y_train
(array([[-0.29562417,  0.64269153, -0.46780112, ..., -0.04336194,
         -0.04065981, -0.42830623],
        [ 1.10281608,  0.78198454, -0.60376879, ..., -0.1180752 ,
         -0.19090662, -0.36760358],
        [-1.36519008, -2.11585979,  1.23296428, ...,  2.49707402,
         -2.45803882, -0.47451425],
        ...,
        [-1.4929067 , -0.39915821,  0.29827688, ..., -1.35462981,
          1.57840753, -0.32345153],
        [-0.21979243,  0.62503272, -0.4795398 , ..., -0.03792906,
         -0.08127206, -0.40306914],
        [ 1.25069102,  0.79320879, -0.55081022, ..., -0.15488753,
         -0.27204396, -0.4705051 ]]), 845    0
 898    0
 45     1
 536    0
 739    0
       ..
 202    1
 711    0
 31     1
 882    0
 563    0
 Name: Class, Length: 787, dtype: int64)
[ ]
y_train=y_train.to_numpy()
y_test=y_test.to_numpy()
[ ]
X_train.shape
(787, 30)
[ ]
X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],1)
X_test=X_test.reshape(X_test.shape[0],X_test.shape[1],1)

# check the shape again
X_train.shape, X_test.shape
((787, 30, 1), (197, 30, 1))
[ ]
# import the libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,Conv1D,BatchNormalization,Dropout,MaxPool1D
[ ]
# import model
model=Sequential()
# layers
model.add(Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=X_train[0].shape))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Conv1D(filters=64, kernel_size=2, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# build ANN
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))

[ ]
model.summary()
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv1d (Conv1D)             (None, 29, 32)            96        
                                                                 
 batch_normalization (BatchN  (None, 29, 32)           128       
 ormalization)                                                   
                                                                 
 dropout (Dropout)           (None, 29, 32)            0         
                                                                 
 conv1d_1 (Conv1D)           (None, 28, 64)            4160      
                                                                 
 batch_normalization_1 (Batc  (None, 28, 64)           256       
 hNormalization)                                                 
                                                                 
 dropout_1 (Dropout)         (None, 28, 64)            0         
                                                                 
 flatten (Flatten)           (None, 1792)              0         
                                                                 
 dense (Dense)               (None, 64)                114752    
                                                                 
 dropout_2 (Dropout)         (None, 64)                0         
                                                                 
 dense_1 (Dense)             (None, 1)                 65        
                                                                 
=================================================================
Total params: 119,457
Trainable params: 119,265
Non-trainable params: 192
_________________________________________________________________
[ ]
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
[ ]
%%time
# fitting the model
history=model.fit(X_train,y_train,epochs=20,validation_data=(X_test,y_test), verbose=1)
Epoch 1/20
25/25 [==============================] - 2s 21ms/step - loss: 0.3995 - accuracy: 0.8501 - val_loss: 0.5155 - val_accuracy: 0.9188
Epoch 2/20
25/25 [==============================] - 0s 11ms/step - loss: 0.1833 - accuracy: 0.9428 - val_loss: 0.4809 - val_accuracy: 0.8832
Epoch 3/20
25/25 [==============================] - 0s 11ms/step - loss: 0.1826 - accuracy: 0.9403 - val_loss: 0.4898 - val_accuracy: 0.6091
Epoch 4/20
25/25 [==============================] - 0s 11ms/step - loss: 0.1686 - accuracy: 0.9416 - val_loss: 0.4821 - val_accuracy: 0.6041
Epoch 5/20
25/25 [==============================] - 0s 11ms/step - loss: 0.1663 - accuracy: 0.9390 - val_loss: 0.4898 - val_accuracy: 0.6091
Epoch 6/20
25/25 [==============================] - 0s 13ms/step - loss: 0.1621 - accuracy: 0.9416 - val_loss: 0.5837 - val_accuracy: 0.5076
Epoch 7/20
25/25 [==============================] - 0s 11ms/step - loss: 0.1518 - accuracy: 0.9428 - val_loss: 0.4645 - val_accuracy: 0.6599
Epoch 8/20
25/25 [==============================] - 0s 11ms/step - loss: 0.1284 - accuracy: 0.9568 - val_loss: 0.4442 - val_accuracy: 0.7107
Epoch 9/20
25/25 [==============================] - 0s 12ms/step - loss: 0.1287 - accuracy: 0.9504 - val_loss: 0.3247 - val_accuracy: 0.8782
Epoch 10/20
25/25 [==============================] - 0s 11ms/step - loss: 0.1428 - accuracy: 0.9479 - val_loss: 0.3238 - val_accuracy: 0.8934
Epoch 11/20
25/25 [==============================] - 0s 11ms/step - loss: 0.1208 - accuracy: 0.9619 - val_loss: 0.3805 - val_accuracy: 0.8122
Epoch 12/20
25/25 [==============================] - 0s 11ms/step - loss: 0.1285 - accuracy: 0.9581 - val_loss: 0.2430 - val_accuracy: 0.9086
Epoch 13/20
25/25 [==============================] - 0s 11ms/step - loss: 0.1167 - accuracy: 0.9543 - val_loss: 0.2805 - val_accuracy: 0.9086
Epoch 14/20
25/25 [==============================] - 0s 11ms/step - loss: 0.1076 - accuracy: 0.9619 - val_loss: 0.2758 - val_accuracy: 0.9137
Epoch 15/20
25/25 [==============================] - 0s 11ms/step - loss: 0.1078 - accuracy: 0.9619 - val_loss: 0.2383 - val_accuracy: 0.9289
Epoch 16/20
25/25 [==============================] - 0s 11ms/step - loss: 0.1175 - accuracy: 0.9606 - val_loss: 0.2436 - val_accuracy: 0.9137
Epoch 17/20
25/25 [==============================] - 0s 11ms/step - loss: 0.1155 - accuracy: 0.9593 - val_loss: 0.2247 - val_accuracy: 0.9340
Epoch 18/20
25/25 [==============================] - 0s 11ms/step - loss: 0.0910 - accuracy: 0.9670 - val_loss: 0.2109 - val_accuracy: 0.9391
Epoch 19/20
25/25 [==============================] - 0s 11ms/step - loss: 0.1054 - accuracy: 0.9644 - val_loss: 0.2121 - val_accuracy: 0.9340
Epoch 20/20
25/25 [==============================] - 0s 11ms/step - loss: 0.1045 - accuracy: 0.9619 - val_loss: 0.2208 - val_accuracy: 0.9239
CPU times: user 8.46 s, sys: 733 ms, total: 9.19 s
Wall time: 7.69 s
[ ]
test_scores= model.evaluate(X_test, y_test, verbose=2)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])

7/7 - 0s - loss: 0.2208 - accuracy: 0.9239 - 33ms/epoch - 5ms/step
Test loss: 0.2208133041858673
Test accuracy: 0.9238578677177429
[ ]
# plot
def plot_learningcurve(history,epochs):
  epoch=range(1,epochs+1)
  # accuracy
  plt.plot(epoch, history.history['accuracy'])
  plt.plot(epoch, history.history['val_accuracy'])
  plt.title('Model accuracy')
  plt.xlabel('epoch')
  plt.ylabel('accuracy')
  plt.legend(['train','val'], loc='upper left')
  plt.show()

  # loss
  plt.plot(epoch, history.history['loss'])
  plt.plot(epoch, history.history['val_loss'])
  plt.title('Model loss')
  plt.xlabel('epoch')
  plt.ylabel('loss')
  plt.legend(['train','val'], loc='upper left')
  plt.show()
[ ]
plot_learningcurve(history,20)

[ ]
y_pred = model.predict(X_test)
[ ]
y_pred
array([[1.21230811e-01],
       [1.00000000e+00],
       [1.00000000e+00],
       [1.00000000e+00],
       [6.74509764e-01],
       [1.00000000e+00],
       [1.65150523e-01],
       [1.00000000e+00],
       [1.00000000e+00],
       [1.39102936e-02],
       [1.00000000e+00],
       [3.22641432e-02],
       [2.50424743e-01],
       [1.00000000e+00],
       [3.94855440e-02],
       [9.99996543e-01],
       [6.07289970e-01],
       [1.59009099e-02],
       [9.99660134e-01],
       [1.67220831e-03],
       [1.00000000e+00],
       [1.00000000e+00],
       [1.00000000e+00],
       [1.00000000e+00],
       [9.99996901e-01],
       [9.99995053e-01],
       [3.58997881e-02],
       [1.48467422e-02],
       [2.13860571e-02],
       [1.00000000e+00],
       [1.74663365e-02],
       [2.42969394e-02],
       [5.22581458e-01],
       [2.63169706e-02],
       [9.99923229e-01],
       [2.27031112e-03],
       [3.23798954e-02],
       [1.00000000e+00],
       [1.22088194e-03],
       [9.99310732e-01],
       [1.00000000e+00],
       [2.86404788e-02],
       [9.82892632e-01],
       [2.51883268e-01],
       [1.69798613e-01],
       [9.99999166e-01],
       [9.70738530e-02],
       [9.99999762e-01],
       [7.69406557e-04],
       [4.18735743e-02],
       [1.71814382e-01],
       [1.30313694e-01],
       [1.00000000e+00],
       [1.08890951e-01],
       [1.00000000e+00],
       [1.06007159e-01],
       [1.56531334e-02],
       [1.00000000e+00],
       [1.00000000e+00],
       [3.41293454e-01],
       [4.35280800e-03],
       [7.38363862e-02],
       [4.84459996e-02],
       [2.57583261e-02],
       [4.29290235e-02],
       [9.99995589e-01],
       [1.60868555e-01],
       [3.84671390e-02],
       [1.00000000e+00],
       [9.00986791e-02],
       [1.00000000e+00],
       [9.99905705e-01],
       [1.00000000e+00],
       [1.55145228e-02],
       [9.99999881e-01],
       [9.99851346e-01],
       [8.29993188e-02],
       [1.33237302e-01],
       [7.46875107e-02],
       [9.99990940e-01],
       [9.91697907e-01],
       [1.00000000e+00],
       [1.00000000e+00],
       [4.88304883e-01],
       [2.71999836e-02],
       [9.95900989e-01],
       [2.55167484e-04],
       [1.00000000e+00],
       [9.99987960e-01],
       [1.00000000e+00],
       [7.14928508e-02],
       [6.39638305e-03],
       [6.53484464e-02],
       [9.99999106e-01],
       [6.83755577e-02],
       [9.99988556e-01],
       [1.00000000e+00],
       [1.00000000e+00],
       [9.99994636e-01],
       [9.99999046e-01],
       [1.98952526e-01],
       [7.50249922e-02],
       [1.00000000e+00],
       [9.42879915e-02],
       [2.99738467e-01],
       [1.44511461e-02],
       [3.95509958e-01],
       [1.00000000e+00],
       [9.99998927e-01],
       [4.20884490e-02],
       [2.96902657e-02],
       [1.00000000e+00],
       [2.20403969e-02],
       [1.64490640e-02],
       [9.99906361e-01],
       [3.17210853e-02],
       [1.00000000e+00],
       [9.99999881e-01],
       [4.35385108e-03],
       [1.00000000e+00],
       [9.97525454e-03],
       [2.99483538e-04],
       [4.48360234e-01],
       [5.71041405e-01],
       [1.00000000e+00],
       [4.28649783e-03],
       [9.42310691e-03],
       [1.00000000e+00],
       [1.00000000e+00],
       [9.99993920e-01],
       [2.85163224e-02],
       [1.56654447e-01],
       [8.69588852e-02],
       [3.52372229e-02],
       [1.48425996e-02],
       [1.00000000e+00],
       [9.99978542e-01],
       [8.03363323e-03],
       [2.02195048e-02],
       [9.83124077e-02],
       [7.16655612e-01],
       [7.63625979e-01],
       [2.17309892e-02],
       [1.00000000e+00],
       [2.87450850e-02],
       [3.52847874e-02],
       [9.99996841e-01],
       [3.06716561e-03],
       [2.79986858e-01],
       [1.52163893e-01],
       [1.84861541e-01],
       [1.06293321e-01],
       [1.00000000e+00],
       [1.59446597e-02],
       [3.13815475e-03],
       [3.03068101e-01],
       [1.00000000e+00],
       [3.37390602e-01],
       [1.00000000e+00],
       [1.25588298e-01],
       [5.19692898e-04],
       [3.49322081e-01],
       [9.87910867e-01],
       [4.66297865e-02],
       [9.99535561e-01],
       [1.00000000e+00],
       [4.27314222e-01],
       [9.92122889e-01],
       [1.00000000e+00],
       [9.99994278e-01],
       [1.00000000e+00],
       [9.99972820e-01],
       [9.12602246e-02],
       [2.33349204e-02],
       [5.32450378e-02],
       [1.03550255e-02],
       [4.81307507e-04],
       [6.03663921e-03],
       [1.14972323e-01],
       [1.00000000e+00],
       [1.24038368e-01],
       [8.20700228e-02],
       [2.06211746e-01],
       [3.18765402e-01],
       [1.00000000e+00],
       [3.08312953e-01],
       [1.00000000e+00],
       [9.99236763e-01],
       [2.89412796e-01],
       [9.99998569e-01],
       [3.40108275e-02],
       [1.02397203e-02],
       [6.63675368e-02],
       [1.33478343e-02],
       [3.89207006e-02],
       [1.00000000e+00],
       [9.99998569e-01]], dtype=float32)
