
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None 

from math import sqrt
# Keras specific
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

import os
import tensorflow
import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES']='1'

Window_size=8
num_steps =4

def lstm_data_transform(x_data, y_data, num_steps=Window_size):#fun3
    x_data=np.array(x_data)
    y_data=np.array(y_data)

    # Prepare the list for the transformed data
    X, y = list(), list()
    # Loop of the entire data set
    for i in range(x_data.shape[0]):
        # compute a new (sliding window) index
        end_ix = i + num_steps
        # if index is larger than the size of the dataset, we stop
        if end_ix >= x_data.shape[0]:
            break
        # Get a sequence of data for x
        seq_X = x_data[i:end_ix]
        # Get only the last element of the sequency for y
        seq_y = y_data[end_ix]
        # Append the list with sequencies
        X.append(seq_X)
        y.append(seq_y)
    # Make final arrays
    x_array = np.array(X)
    y_array = np.array(y)
    return x_array, y_array

def windowsfun(df,i,num_steps):#fun2
    x_data = df.loc[:][df.columns[i]].values
    #num_steps=5
    X=list()
    for i in range(df.shape[0]):
        end_ix =i+num_steps
        if end_ix>=df.shape[0]:
            break
        seq_x = x_data[i:end_ix]
        X.append(seq_x.flatten())
    return pd.DataFrame(X)

def FinalDf(df,num_steps):#Fun1
    master_list =list()
    x_df =df.drop(['GPS_VxF'],axis=1)
    for i in range(len(x_df.columns)):
        kk = windowsfun(x_df,i,num_steps)
        master_list.append(kk)
    updatedDf=pd.concat(master_list,axis=1)
    updatedDf=updatedDf.rename(columns={x:y for x,y in zip(updatedDf.columns,range(0,len(updatedDf.columns)))})
    updatedDf['GPS_VxF']=df[['GPS_VxF']].values[num_steps:]
    return updatedDf

Train_path="/home/aesicd_42/Desktop/tejas/Hyundai_project/DATA/"


def getListOfFiles(dirName):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles

files= getListOfFiles(Train_path)
Cons_X = list()
cons_y = list()

files_xls = [f for f in files if f[-4:] == '.csv']
df = pd.DataFrame()
df_new =pd.DataFrame()
for f in files_xls:

    data = pd.read_csv(f)
    #df = data.loc[data['Brake_by_Driver']>0.5]
    df =data

    #df.loc[df['A'] > 2, 'B'] 
    #df=data
    MAX_Vwhl_F = df[['Vwhl_FL', 'Vwhl_FR','Vwhl_RL','Vwhl_RR']].max(axis=1)
    df['MAX_Vwhl'] = MAX_Vwhl_F
    
    MIN_Vwhl_R = df[['Vwhl_FL', 'Vwhl_FR','Vwhl_RL','Vwhl_RR']].min(axis=1)
    df['MIN_Vwhl'] = MIN_Vwhl_R

    MAX_MIN = df['MAX_Vwhl']/df['MIN_Vwhl']
    df['MAX_MIN'] = MAX_MIN
    
    MAX_MIN_Diff=df['MAX_Vwhl']-df['MIN_Vwhl']
    df['MAX_MIN_Diff'] = MAX_MIN_Diff
    
    Rear_Mean_Vwhl = df[['Vwhl_RL', 'Vwhl_RR']].mean(axis=1)
    df['Rear_Mean_Vwhl'] = Rear_Mean_Vwhl
    
    Front_Max_Vwhl = df[['Vwhl_FL', 'Vwhl_FR']].mean(axis=1)
    df['Front_Max_Vwhl'] = Front_Max_Vwhl

    

    add=df['MAX_MIN']+df['Rear_Mean_Vwhl']
    df['add']=add
    df['SMA50'] = df['Rear_Mean_Vwhl'].rolling(50).mean()
    
    df = df[['Vwhl_FL', 'Vwhl_FR', 'Vwhl_RL', 'Vwhl_RR', 'MAX_MIN', 'Rear_Mean_Vwhl','Front_Max_Vwhl','SMA50','GPS_VxF']]
    df.dropna(inplace=True)


    Filtered_df= FinalDf(df,num_steps)
    x=Filtered_df.drop(['GPS_VxF'],axis=1)
    y =Filtered_df[['GPS_VxF']]

    x_new, y_new = lstm_data_transform(x, y, num_steps=Window_size)
    Cons_X.append(x_new)
    cons_y.append(y_new)

    
    
# Make final arrays
x_array = np.array(Cons_X)
y_array = np.array(cons_y)


X_Train_Tensor = x_array[0]
Y_Train_Tensor = y_array[0]

for k in range(1,len(x_array)):
    X_Train_Tensor=np.concatenate((X_Train_Tensor, x_array[k]))
    Y_Train_Tensor=np.concatenate((Y_Train_Tensor, y_array[k]))



import tensorflow as tf

import keras.backend as K

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM,GRU,Conv1D,TimeDistributed,Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import load_model

shape = (X_Train_Tensor.shape[0],X_Train_Tensor.shape[1],X_Train_Tensor.shape[2],1)

X_Train_Tensor=X_Train_Tensor.reshape(shape)
# Y_Train_Tensor=Y_Train_Tensor.reshape(Y_Train_Tensor.shape[0],1)
print(X_Train_Tensor.shape)
print(Y_Train_Tensor.shape)
#np.save('model_test_train.npy',X_Train_Tensor)
#np.save('model_test_labels.npy',Y_Train_Tensor)

# Hybrid model
# Using mse loss
def create_model():
    model = Sequential()
    model.add(TimeDistributed(Conv1D(filters=4, kernel_size=7, activation='relu', input_shape = shape)))
    model.add(TimeDistributed(Conv1D(filters=8, kernel_size=7, activation='relu')))
    
    model.add(TimeDistributed(Flatten()))
    
    model.add(GRU (units = 5,activation='tanh', return_sequences = True))

    model.add(GRU (units = 3,activation='tanh'))
   
    model.add(Dense(units = 1))
   
    model.compile(loss="mse", optimizer='adam')
    print (model.summary)
    return model

model_gru = create_model()

model_gru.build(shape)

model_gru.summary()
print(X_Train_Tensor.shape)
from keras.callbacks import ModelCheckpoint



def fit_model(model):
    early_stop = keras.callbacks.EarlyStopping(monitor = 'val_loss',patience = 20)

    history = model.fit(X_Train_Tensor, Y_Train_Tensor, epochs = 15,  
                        validation_split = 0.15, batch_size = 32, 
                        shuffle = True)
    return history

history_gru = fit_model(model_gru)
model_gru.save('/home/aesicd_42/Desktop/tejas/Hyundai_project/model_sriker.h5')