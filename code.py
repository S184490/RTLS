import os
import json
import numpy as np
import psycopg2
from collections import OrderedDict
import time
import threading
import paho.mqtt.client as mqtt
import uuid
from itertools import chain
import sys
import random
from collections import deque
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D,Conv3D,MaxPooling3D, MaxPooling2D,Flatten
from keras.layers import Dropout
from keras.layers import LSTM, TimeDistributed
from keras import backend
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from sklearn.cluster import KMeans
from sklearn.svm import SVC
import warnings
import pickle
import pandas
from collections import Counter

sys.setrecursionlimit(1000000000) # let the code run almost forever

occupation = 'patient'
ifReTrainModel = True
ifSaveTrainModel = False
ifExitAfterTrain = False
dimSubSpace = 10 # keep 10 subspace dimension
trainLength    = 90 #s
trainInterval  = 1 #s
trainWindow    = 5 #s
threadInterval = 1 #s
threadWindow   = 5 #s
stayLength = 10 #s
lookBack = 8
thresholdProb = 0.0
resetStep = 10
resetTime = 30 # dispear if not seen in 30s
Nvote = 10

## read in arguments. Example: python locator_zohaib.py patient 1 5
if len(sys.argv)>1:
    occupation = sys.argv[1]
if len(sys.argv)>2:
    threadInterval = int(sys.argv[2])
if len(sys.argv)>3:
    threadWindow = int(sys.argv[3])
if len(sys.argv)>4:
    ifReTrainModel = sys.argv[4][0]=='T' # T or F for retrain model
if len(sys.argv)>5:
    ifSaveTrainModel = sys.argv[5][0]=='T' # T or F for saving model
if len(sys.argv)>6:
    ifExitAfterTrain = sys.argv[6][0]=='T' # T or F for existing program after training
if len(sys.argv)>7:
    Nvote = int(sys.argv[7])
if occupation.lower()[:3]=='pat':
    limTag = [0,9999]
elif occupation.lower()[:3]=='phy':
    limTag = [10000,10999]
elif occupation.lower()[:3]=='equ':
    limTag = [11000,13999]
elif occupation.lower()[:3]=='tt1':
    limTag = [131073,131077]
elif occupation.lower()[:3]=='tt2':
    limTag = [131078,131082]
else:
    print('Wrong occupation, must be patient, physician, or equipment')
    print('Existing...')
    sys.exit()

## read room connection information
roomConnect = {}
with open('data/zones-EC-connection.txt') as f:
    lines = [line.rstrip('\n') for line in f][1:]
for line in lines:
    rooms = map(int, line.split())
    roomConnect[rooms[0]] = rooms[1:]

## allowed rooms dict
dictAllowedLocs = {}
for room in roomConnect.keys():
    allowedLocs = [room,]
    dictAllowedLocs[room] = [[room,],]
    for n in range(resetStep):
        allowedLocs1 = allowedLocs[:]
        for allowedLoc in allowedLocs:
            allowedLocs1.extend(roomConnect[allowedLoc])
        allowedLocs1 = list(set(allowedLocs1))
        dictAllowedLocs[room] += [allowedLocs1,]
        allowedLocs = allowedLocs1

## room distance dict
dictRoomDist = {}
for room in roomConnect.keys():
    dictRoomDist[room] = {}
    for room1 in roomConnect.keys():
        dictRoomDist[room][room1] = resetStep
    for n in range(resetStep):
        for room1 in dictAllowedLocs[room][n]:
            dictRoomDist[room][room1] = min(dictRoomDist[room][room1], n)

## normalize strength
def normalize(streth):
    return min(max(0.,(streth+90.)/60.),1.) # logorithm
    #return min(max(0.,(10**(streth/10.)*10000)**0.5),1.) # recipocal
    #return 10.**(streth/100.) # recipocal

def prepareTrainData1(list_scan,list_zone,scanner,tag,strength,timestamp,zone,window=5,interval=1,advrate=0.1,shuffle=False):
    Nwindow = int(window/advrate)
    Ninterval = int(interval/advrate)
    data = prepareGridData(list_scan,list_zone,scanner,tag,strength,timestamp,zone,shuffle)
    list1_zone = [n for n in data.keys() if n in list_zone]
    collectAll = []
    collectZne = []
    zneTagData = {}
    for zne in list1_zone:
        n = 0
        zneTagData[zne] = {}
        for tg in data[zne].keys():
            dd = data[zne][tg]
            dd[dd==0] = np.nan
            csum = np.nancumsum(dd,axis=0)
            dd[dd!=1] = 1.
            dsum = np.nancumsum(dd,axis=0)
            Nwindow = int(window*len(dd)/trainLength)
            Ninterval = int(interval*len(dd)/trainLength)
            mavg = (csum[Nwindow::Ninterval]-csum[:-Nwindow:Ninterval])/(dsum[Nwindow::Ninterval]-dsum[:-Nwindow:Ninterval]+1e-9)
            collectAll.append(mavg)
            collectZne.append([zne,]*len(mavg))
            zneTagData[zne][tg] = mavg
            n+= len(mavg)
    collectAll = np.concatenate(collectAll)
    collectZne = np.concatenate(collectZne)
    return collectAll,collectZne,zneTagData  # [Nsample,Nscanner], [Nsample]
    

def prepareGridData(list_scan,list_zone,scanner,tag,strength,timestamp,zone,shuffle=False):
    #### encode scanner
    encoder = LabelEncoder()
    encoder.fit(list_scan)
    encoded_scannerB = np.zeros(scanner.shape,dtype=bool)
    for n,scan in enumerate(scanner):
        if scan in list_scan:
            encoded_scannerB[n] = True
        else:
            encoded_scannerB[n] = False
    encoded_scanner = np.ones(scanner.shape,dtype=int)*-1
    encoded_scanner[encoded_scannerB] = encoder.transform(scanner[encoded_scannerB])
    t0 = timestamp[0]
    tg0 = tag[0]
    zne0 = zone[0]
    data = OrderedDict()
    data[zne0] = OrderedDict()
    data[zne0][tg0] = [np.zeros((len(list_scan))),]
    for escan,tg,tstp,zne,strth in zip(encoded_scanner,tag,timestamp,zone,strength):
        if zne!=zne0:
            data[zne0][tg0] = np.array(data[zne0][tg0])
            data[zne] = OrderedDict()
            data[zne][tg] = [np.zeros((len(list_scan))),]
        elif tg!=tg0:
            data[zne0][tg0] = np.array(data[zne0][tg0])
            data[zne0][tg] = [np.zeros((len(list_scan))),]
        elif tstp>t0+0.05e9:
            data[zne0][tg0].append(np.zeros((len(list_scan))))
        data[zne][tg][-1][escan] = normalize(strth)
        t0 = tstp
        tg0 = tg
        zne0 = zne
    data[zne][tg] = np.array(data[zne][tg])
    if shuffle:
        for zne in data.keys():
            for tg in data[zne].keys():
                random.shuffle(data[zne][tg])
    return data # data[zne][tg][sample,scan]

def prepareGridDataTrajectory(list_scan,list_zone,scanner,tag,strength,timestamp,zone):
    #### encode scanner
    encoder = LabelEncoder()
    encoder.fit(list_scan)
    encoded_scannerB = np.zeros(scanner.shape,dtype=bool)
    for n,scan in enumerate(scanner):
        if scan in list_scan:
            encoded_scannerB[n] = True
        else:
            encoded_scannerB[n] = False
    encoded_scanner = np.ones(scanner.shape,dtype=int)*-1
    encoded_scanner[encoded_scannerB] = encoder.transform(scanner[encoded_scannerB])
    t0 = timestamp[0]
    tg0 = tag[0]
    zne0 = zone[0]
    data = OrderedDict()
    data[tg0] = [np.zeros((len(list_scan))),]
    label = OrderedDict()
    label[tg0] = [0,]
    time = OrderedDict()
    time[tg0] = [0,]
    for escan,tg,tstp,zne,strth in zip(encoded_scanner,tag,timestamp,zone,strength):
        if tg!=tg0:
            data[tg0] = np.array(data[tg0])
            data[tg] = [np.zeros((len(list_scan))),]
            label[tg0] = np.array(label[tg0])
            label[tg] = [0,]
            time[tg0] = np.array(time[tg0])
            time[tg] = [0,]
        elif tstp>t0+0.05e9:           #########
            data[tg0].append(np.zeros((len(list_scan))))
            label[tg0].append(0)
            time[tg0].append(0)
        data[tg][-1][escan] = normalize(strth)
        label[tg][-1] = zne
        time[tg][-1] = tstp
        t0 = tstp
        tg0 = tg
        zne0 = zne
    data[tg] = np.array(data[tg])
    label[tg] = np.array(label[tg])
    time[tg] = np.array(time[tg])

    return data,label # data[zne][tg][sample,scan]

def prepareTrainData1Trajectory(list_scan,list_zone,scanner,tag,strength,timestamp,zone,window=5,interval=1,advrate=0.1):
    Nwindow = int(window/advrate)
    Ninterval = int(interval/advrate)
    data,label = prepareGridDataTrajectory(list_scan,list_zone,scanner,tag,strength,timestamp,zone)
    collectAll = []
    collectZne = []
    tagData = {}
    tagLabel = {}
    for tg in data.keys():
        dd = data[tg]
        dd[dd==0] = np.nan
        csum = np.nancumsum(dd,axis=0)
        dd[dd!=1] = 1.
        dsum = np.nancumsum(dd,axis=0)
        mavg = (csum[Nwindow::Ninterval]-csum[:-Nwindow:Ninterval])/(dsum[Nwindow::Ninterval]-dsum[:-Nwindow:Ninterval])
        collectAll.append(mavg)
        collectZne.append(label[tg][Nwindow::Ninterval])
        tagData[tg] = mavg
        tagLabel[tg] = label[tg][Nwindow::Ninterval]
    collectAll = np.concatenate(collectAll)  
    collectZne = np.concatenate(collectZne)
    return collectAll,collectZne,tagData,tagLabel  # [Nsample,Nscanner], [Nsample]

def generateSequenceTrajectory(list_zone,data,label):
    sequenceAll = []
    sequenceZne = []
    for tg in data.keys():
        for n in range(len(data[tg])-lookBack):
            sequenceAll.append(list(data[tg][n:n+lookBack]))
            sequenceZne.append(label[tg][n:n+lookBack])
    return np.array(sequenceAll), np.array(sequenceZne)

def generateSequence(list_zone,data,N1=500,N2=50):
#[sample,timestep,character]
    sequenceAll = []
    sequenceZne = []
    for zne in data.keys():
        for tg in data[zne].keys():
            for n in range(len(data[zne][tg])-lookBack):
                sequenceAll.append(list(data[zne][tg][n:n+lookBack]))
            sequenceZne += [[[zne,]]*lookBack,]*(len(data[zne][tg])-lookBack)
    for zne in data.keys():
        data[zne] = np.concatenate(data[zne].values())
    for zne in data.keys():
        for n in range(N1):
            indx = random.sample(range(len(data[zne])),lookBack)
            sequenceAll.append( list(data[zne][indx]))
        sequenceZne += [[[zne,]]*lookBack,]*N1
        for connectedZne in roomConnect[zne]:
            if connectedZne in list_zone:
                for n in range(N2):
                    n1 = random.randint(1,lookBack-1)
                    n2 = lookBack-n1
                    indx1 = random.sample(range(len(data[zne])),n1)
                    indx2 = random.sample(range(len(data[connectedZne])),n2)
                    sequenceAll.append(list(data[connectedZne][indx2])+list(data[zne][indx1]))
                    sequenceZne += [[[connectedZne,]]*n2 + [[zne,]]*n1,]
    sequenceZne = np.concatenate(sequenceZne,1).T
    return np.array(sequenceAll), np.array(sequenceZne)

def generateSequenceAug(list_zone,data,N=60,M=5,window=5):
#[sample,timestep,character]
    sequenceAll = []
    sequenceZne = []
    for zne in data.keys():
        data[zne] = np.concatenate(data[zne].values())
    for zne in data.keys():
        for connectedZne in roomConnect[zne]:
            if connectedZne in list_zone:
                for m in range(M):
                    #indx1 = random.sample(range(len(data[zne])),N)
                    #indx2 = random.sample(range(len(data[connectedZne])),N+window)
                    i1 = random.randint(0,len(data[zne])-N); indx1 = list(range(i1,i1+N))
                    i2 = random.randint(0,len(data[connectedZne])-N-window); indx2 = list(range(i2,i2+N+window))
                    aaa = list(data[connectedZne][indx2])+list(data[zne][indx1])
                    dd = np.array(aaa)
                    dd[dd==0] = np.nan
                    csum = np.nancumsum(dd,axis=0)
                    dd[dd!=1] = 1.
                    dsum = np.nancumsum(dd,axis=0)
                    mavg = (csum[window::1]-csum[:-window:1])/(dsum[window::1]-dsum[:-window:1])
                    aaa = [aa for aa in mavg]
                    bbb = [[connectedZne,]]*N + [[zne,]]*N
                    for n in range(N*2-lookBack+1):
                        sequenceAll.append(aaa[n:n+lookBack])
                        sequenceZne += [bbb[n:n+lookBack],]
    sequenceZne = np.concatenate(sequenceZne,1).T
    return np.array(sequenceAll), np.array(sequenceZne)

def generateSequenceTest(list_zone,data):
    sequenceAll = []
    sequenceZne = []
    for zne in data.keys():
        for tg in data[zne].keys():
            for n in range(len(data[zne][tg])-lookBack):
                sequenceAll.append(list(data[zne][tg][n:n+lookBack]))
            sequenceZne += [[[zne,]]*lookBack,]*(len(data[zne][tg])-lookBack)
    sequenceZne = np.concatenate(sequenceZne,1).T
    return np.array(sequenceAll), np.array(sequenceZne)

def train_svm(df,trainTag=[4,5,6,7,8],testTag=[3,],post=False,vote=False):
    scanner,tag,strength,timestamp,zone = np.array(df['scanner']),np.array(df['tag']),np.array(df['strength']),np.array(df['timestamp']),np.array(df['zone'],dtype=int)
    ## prepare training data
    Nbatch = 1000
    Nepoch = 70
    indx = [False,]*len(df)
    for tg in trainTag:
        indx = np.logical_or(indx,tag==tg)
    list_scan = np.unique(scanner[indx])
    list_zone = np.unique(zone[indx])
    list_tag = np.unique(tag[indx])
    collectAll,collectZne,_ = prepareTrainData1(list_scan,list_zone,scanner[indx],tag[indx],strength[indx],timestamp[indx],zone[indx])

    ## prepare test data
    indx = [False,]*len(df)
    for tg in testTag:
        indx = np.logical_or(indx,tag==tg)
    testAll,testZne,_ = prepareTrainData1(list_scan,list_zone,scanner[indx],tag[indx],strength[indx],timestamp[indx],zone[indx],shuffle=False)

    ## machine learning training
    warnings.simplefilter("ignore", DeprecationWarning)
    
    X = collectAll
    Y = collectZne
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    X_train,y_train = X,encoded_Y
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(testAll)
    y_test = encoder.transform(testZne)
    ## Callback class for cross-validation
    ## Zone distance dict
    dictZoneDist = {}
    for room in roomConnect.keys():
        dictZoneDist[room] = np.array([ dictRoomDist[room][tmp] for tmp in list_zone ])

    model = SVC(kernel='rbf',probability=True)
    model.fit(X_train[:,:], y_train)
    y_pred = model.predict(X_test)
    yp = encoder.inverse_transform(y_pred).squeeze()
    ## consider zone connection
    if post:
        for n in range(len(testZne)):
            if n==0:
                yp[0] = encoder.inverse_transform(y_pred[:1])[0]
            elif testZne[n]==testZne[n-1]:
                y_pred[n,:] *= 10.**(-dictZoneDist[yp[n-1]]*1.)
                yp[n] = encoder.inverse_transform(y_pred[n:n+1])[0]

    ## consider voting
    if vote:
        ypv = yp.copy()
        for n in range(len(testZne)):
            ypv[n],num = Counter(yp[max(n-Nvote+1,0):n+1]).most_common(1)[0]
            if num<min(n+1,Nvote)*0.8:
                ypv[n] = ypv[n-1]
        yp = ypv
    ## compute accuracy
    accuracy = len(yp[yp==testZne])*1./len(yp)
    nCorrect,nTotal = 0,0
    trueZneOld = testZne[0]
    for trueZne,predZne in zip(testZne,yp):
        if trueZne!=trueZneOld:
            nCorrect,nTotal = 0,0
        if trueZne==predZne:
            nCorrect += 1
        nTotal += 1
        trueZneOld = trueZne
    print 'accuracy:',accuracy
    return list_scan,list_zone,model

def train_mlp(df,trainTag=[4,5,6,7,8],testTag=[3,],post=[False,10],vote=False,loadTraj=False):
    scanner,tag,strength,timestamp,zone = np.array(df['scanner']),np.array(df['tag']),np.array(df['strength']),np.array(df['timestamp']),np.array(df['zone'],dtype=int)
    ## prepare training data
    Nbatch = 1000
    Nepoch = 1000
    indx = [False,]*len(df)
    for tg in trainTag:
        indx = np.logical_or(indx,tag==tg)
    list_scan = np.unique(scanner[indx])
    list_zone = np.unique(zone[indx])
    list_tag = np.unique(tag[indx])
    collectAll,collectZne,_ = prepareTrainData1(list_scan,list_zone,scanner[indx],tag[indx],strength[indx],timestamp[indx],zone[indx])

    ## prepare test data
    indx = [False,]*len(df)
    for tg in testTag:
        indx = np.logical_or(indx,tag==tg)
    testAll,testZne,_ = prepareTrainData1(list_scan,list_zone,scanner[indx],tag[indx],strength[indx],timestamp[indx],zone[indx],shuffle=False)

    ## machine learning training
    warnings.simplefilter("ignore", DeprecationWarning)
    
    X = collectAll
    Y = collectZne
    encoder = OneHotEncoder()
    encoder.fit(Y.reshape(-1,1))
    encoded_Y = encoder.transform(Y.reshape(-1,1))
    X_train,y_train = X,encoded_Y
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    model = Sequential()
    model.add(Dense(units = 200, kernel_initializer = 'uniform',activation = 'relu', input_dim=len(list_scan)))
#    #model.add(Conv1D(2,2,strides=1,activation = 'relu',input_shape=(1,len(list_scan))))
    model.add(Dropout(0.5))
    model.add(Dense(units = 200, kernel_initializer = 'uniform',activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units = 200, kernel_initializer = 'uniform',activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(list_zone),activation='softmax'))
    #model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=[])
    #model.compile(loss='mean_squared_error',optimizer='adam',metrics=[])
    X_test = sc.transform(testAll)
    y_test = encoder.transform(testZne.reshape(-1,1))
    ## Callback class for cross-validation
    ## Zone distance dict
    dictZoneDist = {}
    for room in roomConnect.keys():
        dictZoneDist[room] = np.array([ dictRoomDist[room][tmp] for tmp in list_zone ])
    class LossHistory(keras.callbacks.Callback):
        def __init__(self):
            self.predhis = []
    
        def on_epoch_end(self, batch, logs={}):
            y_pred = model.predict(X_test)
            yp = encoder.inverse_transform(y_pred[:,:]).squeeze()
            ## consider zone connection
            if post[0]:
                for n in range(len(testZne)):
                    if n==0:
                        yp[0] = encoder.inverse_transform(y_pred[:1,:])[0]
                    elif testZne[n]==testZne[n-1]:
                        y_pred[n,:] *= post[1]**(-dictZoneDist[yp[n-1]]*1.)
                        yp[n] = encoder.inverse_transform(y_pred[n:n+1,:])[0]

            ## consider voting
            if vote:
                ypv = yp.copy()
                for n in range(len(testZne)):
                    ypv[n],num = Counter(yp[max(n-Nvote+1,0):n+1]).most_common(1)[0]
                    if num<min(n+1,Nvote)*0.6:
                        ypv[n] = ypv[n-1]
                yp = ypv
            ## compute accuracy
            accuracy = len(yp[yp==testZne])*1./len(yp)
            nCorrect,nTotal = 0,0
            trueZneOld = testZne[0]
            for trueZne,predZne in zip(testZne,yp):
                if trueZne!=trueZneOld:
                    nCorrect,nTotal = 0,0
                if trueZne==predZne:
                    nCorrect += 1
                nTotal += 1
                trueZneOld = trueZne
            #print 'accuracy:',accuracy

    history = LossHistory()
    model.fit(X_train[:,:], y_train.toarray(), batch_size=Nbatch,epochs=Nepoch,callbacks=[history],validation_data=(X_test,y_test))
    return list_scan,list_zone,model

def train_lstm(df,trainTag=[4,5,6,7,8],testTag=[3,],augTrain=False,augTest=False,post=[False,10],loadTraj=False,window=5):
    Maug = 25
    scanner,tag,strength,timestamp,zone = np.array(df['scanner']),np.array(df['tag']),np.array(df['strength']),np.array(df['timestamp']),np.array(df['zone'],dtype=int)
    ## prepare training data
    Nbatch = 1000
    Nepoch = 70
    indx = [False,]*len(df)
    for tg in trainTag:
        indx = np.logical_or(indx,tag==tg)
    list_scan = np.unique(scanner[indx])
    list_zone = np.unique(zone[indx])
    list_tag = np.unique(tag[indx])
    collectAll,collectZne,data = prepareTrainData1(list_scan,list_zone,scanner[indx],tag[indx],strength[indx],timestamp[indx],zone[indx],window=window)
    if augTrain:
        #collectAll,collectZne = generateSequence(list_zone,data,500,200)
        collectAll,collectZne = generateSequenceAug(list_zone,data,M=5,N=Maug,window=window)
    else:
        collectAll,collectZne = generateSequenceTest(list_zone,data)
    list_zone = np.unique(collectZne)
    
    ## prepare testing data
    indx = [False,]*len(df)
    for tg in testTag:
        indx = np.logical_or(indx,tag==tg)
    testAll,testZne,data = prepareTrainData1(list_scan,list_zone,scanner[indx],tag[indx],strength[indx],timestamp[indx],zone[indx],window=window)
    if augTest:
        #testAll,testZne = generateSequence(list_zone,data,100,20)
        testAll,testZne = generateSequenceAug(list_zone,data,M=1,N=Maug,window=window)
    else:
        testAll,testZne = generateSequenceTest(list_zone,data)

    ## load trajectory
    if loadTraj:
        df = pandas.read_csv('traj_1.csv')
        df = df[df['strength']>-90] # neglect <-90 strength
        df = df.sort_values(by=['tag','timestamp']) # sort
        scannerT,tagT,strengthT,timestampT,zoneT = np.array(df['scanner']),np.array(df['tag']),np.array(df['strength']),np.array(df['timestamp']),np.array(df['zone'],dtype=int)
        testAll,testZne,data,label = prepareTrainData1Trajectory(list_scan,list_zone,scannerT,tagT,strengthT,timestampT,zoneT,window=window)
        testAll,testZne = generateSequenceTrajectory(list_zone,data,label)

    ## machine learning training
    warnings.simplefilter("ignore", DeprecationWarning)
    
    X = collectAll
    Y = collectZne
    encoder = OneHotEncoder()
    encoder.fit(Y.reshape(-1,1))
    encoded_Y = encoder.transform(Y.reshape(-1,1)).toarray().reshape(len(Y),lookBack,-1)
    X_train,y_train = X,encoded_Y
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train.reshape(-1,len(list_scan))).reshape(-1,lookBack,len(list_scan))
    model = Sequential()
    model.add(LSTM(units = 200, input_shape=(lookBack,len(list_scan)),return_sequences=True))
    model.add(Dropout(0.5))
    model.add(Dense(units = 200, kernel_initializer = 'uniform',activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units = 200, kernel_initializer = 'uniform',activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(list_zone),activation='softmax'))
    #model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=[])
    #model.compile(loss='mean_squared_error',optimizer='adam',metrics=[])
    X_test = sc.transform(testAll.reshape(-1,len(list_scan))).reshape(-1,lookBack,len(list_scan))
    y_test = encoder.transform(testZne.reshape(-1,1)).toarray().reshape(len(testZne),lookBack,-1)
    y_true = testZne[:,-1]
    dictZoneDist = {}
    for room in roomConnect.keys():
        dictZoneDist[room] = np.array([ dictRoomDist[room][tmp] for tmp in list_zone ])
    ## Callback class for cross-validation
    class LossHistory(keras.callbacks.Callback):
        def __init__(self):
            self.predhis = []
    
        def on_epoch_end(self, batch, logs={}):
            y_pred = model.predict(X_test)[:,-1,:]
            yp = encoder.inverse_transform(y_pred).squeeze()
            ## consider zone connection
            if post[0]:
                for n in range(len(testZne)):
                    if n==0:
                        yp[n] = encoder.inverse_transform(y_pred[n:n+1,:])[0]
                        #yp[n] = y_true[n]
                    else:# testZne[n,-1]==testZne[n-1,-1]:
                        y_pred[n,:] *= post[1]**(-dictZoneDist[yp[n-1]]*1.)
                        yp[n] = encoder.inverse_transform(y_pred[n:n+1,:])[0]
            ## compute accuracy
            accuracy = len(yp[yp==y_true])*1./len(yp)
#            nCorrect,nTotal = 0,0
#            trueZneOld = y_true[0]
#            for trueZne,predZne in zip(y_true,yp):
#                if trueZne!=trueZneOld:
#                    nCorrect,nTotal = 0,0
#                if trueZne==predZne:
#                    nCorrect += 1
#                nTotal += 1
#                trueZneOld = trueZne
            print 'accuracy:',accuracy
            print 'jumps:',sum(np.diff(yp)!=0), sum(np.diff(y_true)!=0)
            if loadTraj:
                acc = np.array([False,]*(len(yp)-20))
                for n in [-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10]:
                    acc = np.logical_or(acc, yp[10:-10]==y_true[10-n:len(yp)-10-n])
                print 'acc:',sum(acc)*1./len(acc)
#                for n in [0,1,2,3,4,5,6,7,8,9,10]:
#                    print 'accuracy%d:'%n, sum(yp[10:]==y_true[10-n:len(yp)-n])*1./len(yp[10:])
#                print yp
#            if augTest:
#                accA = np.ones(len(yp)/(2*Maug-lookBack+1))
#                for m in range(len(yp)/(2*Maug-lookBack+1)):
#                    accA[m] = 0
#                    for n in [0,1,2,3,4,5,6]:
#                        accA[m] = max(accA[m],sum((yp.reshape(-1,2*Maug-lookBack+1)[m,6:]==y_true.reshape(-1,2*Maug-lookBack+1)[m,6-n:2*Maug-lookBack-n+1]).reshape(-1))*1./len(yp.reshape(-1,2*Maug-lookBack+1)[m,6:].reshape(-1)))
#                for n in [0,1,2,3,4,5,6]:
#                    print 'accuracy%d:'%n, sum((yp.reshape(-1,2*Maug-lookBack+1)[:,6:]==y_true.reshape(-1,2*Maug-lookBack+1)[:,6-n:2*Maug-lookBack-n+1]).reshape(-1))*1./len(yp.reshape(-1,2*Maug-lookBack+1)[:,6:].reshape(-1))
#                print accA.mean()

    history = LossHistory()
    model.fit(X_train[:,:], y_train, batch_size=Nbatch,epochs=Nepoch,callbacks=[history],validation_data=(X_test,y_test))
    sys.exit()
    return list_scan,list_zone,model

scannerCoordinatesDict = {
202481599873784: [184, 326  ],
202481592148541: [220, 251  ],
202481596030207: [253, 317  ],
202481589970268: [325, 185  ],
202481587856433: [284, 251  ],
202481589778594: [328, 326  ],
202481602374721: [422, 326  ],
202481597950131: [460, 254  ],
202481600304414: [491, 314  ],
202481598624604: [563, 185  ],
202481591964219: [521, 251  ],
202481585994159: [566, 324  ],
202481591934782: [661, 326  ],
202481602243125: [697, 254  ],
202481598535568: [730, 314  ],
202481595411999: [730, 173  ],
202481596591029: [760, 251  ],
202481590823753: [805, 326  ],
202481597517835: [898, 326  ],
202481592562121: [931, 251  ],
202481597280647: [964, 314  ],
202481593769689: [964, 176  ],
202481596606950: [995, 249  ],
202481595101358: [1040, 326 ],
202481590401959: [1133, 326 ],
202481597952377: [1171, 251 ],
202481586647187: [1201, 312 ],
202481595654280: [1204, 176 ],
202481595962342: [1232, 251 ],
202481590012428: [1276, 326 ],
202481590961552: [1367, 326 ],
202481598748190: [1403, 251 ],
202481598238909: [1436, 314 ],
202481588473028: [1433, 176 ],
202481601543339: [1465, 251 ],
202481600217603: [1511, 326 ],
202481591182572: [1609, 326 ],
202481593219173: [1643, 251 ],
202481601537651: [1678, 312 ],
202481601690131: [1678, 176 ],
202481588014102: [1708, 251 ],
202481595300756: [1753, 326 ],
202481593073202: [209, 1151 ],
202481586770178: [287, 1212 ],
202481595930789: [379, 1149 ],
202481599572071: [511, 1025 ],
202481592951276: [478, 1209 ],
202481595583081: [575, 1149 ],
202481587122663: [769, 1110 ],
202481586687629: [73, 818   ],
202481592814994: [73, 732   ],
202481592285809: [73, 648   ],
202481599414040: [73, 564   ],
202481592757237: [185, 939  ],
202481594469276: [251, 882  ],
202481586283485: [341, 882  ],
202481597435935: [386, 939  ],
202481589823194: [433, 881  ],
202481594345615: [523, 879  ],
202481594292315: [566, 939  ],
202481590274456: [313, 629  ],
202481594821872: [575, 618  ],
202481588235228: [713, 939  ],
202481597627647: [778, 866  ],
202481591346384: [778, 774  ],
202481594361561: [778, 686  ],
202481590805393: [778, 596  ],
202481596210804: [922, 827  ],
202481588308259: [860, 561  ],
202481600436103: [1018, 854 ],
202481601519759: [1043, 620 ],
202481593293050: [1090, 749 ],
202481594518960: [1126, 854 ],
202481598412739: [1144, 620 ],
202481597561918: [1231, 572 ],
202481597145316: [1213, 999 ],
202481588476203: [1213, 882 ],
202481588991215: [1270, 942 ],
202481598240252: [1325, 1061],
202481586930795: [1325, 999 ],
202481600336277: [1417, 1059],
202481600281519: [1417, 999 ],
202481594535730: [1534, 1155],
202481590239181: [1510, 1064],
202481593705307: [1510, 926 ],
202481593329060: [1691, 1115],
202481588007352: [1748, 1034],
202481599403958: [1657, 930 ],
202481591104544: [1831, 917 ],
202481594876472: [1315, 833 ],
202481598031159: [1331, 675 ],
202481594630976: [1390, 675 ],
202481591756967: [1561, 660 ],
202481588627595: [1610, 660 ],
202481601040379: [1684, 774 ],
202481600327270: [1684, 690 ],
202481599071756: [1658, 591 ],
202481599656877: [1753, 593 ],
202481594609210: [85, 494   ],
202481591694768: [1178, 492 ],
202481600215218: [1847, 450 ],
202481602084751: [94, 377   ],
202481590956569: [73, 269   ],
202481601426033: [94, 159   ],
202481600146643: [109, 50   ],
202481598108029: [229, 102  ],
202481593142629: [364, 50   ],
202481587700937: [460, 102  ],
202481596352747: [635, 50   ],
202481600749663: [703, 102  ],
202481589387491: [895, 50   ],
202481598968436: [979, 102  ],
202481591856896: [1084, 50  ],
202481600320493: [1159, 102 ],
202481601536502: [1313, 50  ],
202481600928060: [1421, 102 ],
202481594208274: [1561, 50  ],
202481594931693: [1691, 102 ],
202481585889781: [1871, 300 ],
202481597952975: [1871, 50  ],
202481595588330: [65, 307   ],
202481599275972: [78, 437   ],
202481589174252: [239, 397  ],
202481599323929: [675, 397  ],
202481601180343: [833, 437  ],
202481593534803: [940, 608  ],
202481587091977: [993, 523  ],
202481590930337: [1065, 397 ],
202481597056241: [1164, 774 ],
202481601995306: [1164, 667 ],
202481601738649: [1164, 584 ],
202481591115309: [1164, 481 ],
202481589620558: [1260, 397 ],
202481593583243: [1350, 589 ],
202481591721573: [1350, 473 ],
202481591447923: [1427, 685 ],
202481596578391: [1460, 517 ],
202481588471439: [1518, 397 ],
202481602112786: [1622, 421 ],
202481600284849: [1735, 784 ],
202481586745007: [1735, 641 ],
202481589056738: [1735, 553 ],
202481592457971: [1735, 462 ],
202481594981542: [1860, 432 ],
202481589921925: [2059, 397 ],
202481602030726: [104, 358  ],
202481585910238: [225, 358  ],
202481596909065: [381, 358  ],
202481600826922: [453, 358  ],
202481597889233: [620, 419  ],
202481600158213: [620, 299  ],
202481592905726: [270, 78   ],
202481599235133: [718, 78   ],
202481597396653: [894, 247  ],
202481593034624: [1941, 78  ],
}

scannerIdDict = {
202481599873784 : 1001 ,
202481592148541 : 1002 ,
202481596030207 : 1003 ,
202481589970268 : 1004 ,
202481587856433 : 1005 ,
202481589778594 : 1006 ,
202481602374721 : 1007 ,
202481597950131 : 1008 ,
202481600304414 : 1009 ,
202481598624604 : 1010 ,
202481591964219 : 1011 ,
202481585994159 : 1012 ,
202481591934782 : 1013 ,
202481602243125 : 1014 ,
202481598535568 : 1015 ,
202481595411999 : 1016 ,
202481596591029 : 1017 ,
202481590823753 : 1018 ,
202481597517835 : 1019 ,
202481592562121 : 1020 ,
202481597280647 : 1021 ,
202481593769689 : 1022 ,
202481596606950 : 1023 ,
202481595101358 : 1024 ,
202481590401959 : 1025 ,
202481597952377 : 1026 ,
202481586647187 : 1027 ,
202481595654280 : 1028 ,
202481595962342 : 1029 ,
202481590012428 : 1030 ,
202481590961552 : 1031 ,
202481598748190 : 1032 ,
202481598238909 : 1033 ,
202481588473028 : 1034 ,
202481601543339 : 1035 ,
202481600217603 : 1036 ,
202481591182572 : 1037 ,
202481593219173 : 1038 ,
202481601537651 : 1039 ,
202481601690131 : 1040 ,
202481588014102 : 1041 ,
202481595300756 : 1042 ,
202481593073202 : 1043 ,
202481586770178 : 1044 ,
202481595930789 : 1045 ,
202481599572071 : 1046 ,
202481592951276 : 1047 ,
202481595583081 : 1048 ,
202481587122663 : 1049 ,
202481586687629 : 1050 ,
202481592814994 : 1051 ,
202481592285809 : 1052 ,
202481599414040 : 1053 ,
202481592757237 : 1054 ,
202481594469276 : 1055 ,
202481586283485 : 1056 ,
202481597435935 : 1057 ,
202481589823194 : 1058 ,
202481594345615 : 1059 ,
202481594292315 : 1060 ,
202481590274456 : 1061 ,
202481594821872 : 1062 ,
202481588235228 : 1063 ,
202481597627647 : 1064 ,
202481591346384 : 1065 ,
202481594361561 : 1066 ,
202481590805393 : 1067 ,
202481596210804 : 1068 ,
202481588308259 : 1069 ,
202481600436103 : 1070 ,
202481601519759 : 1071 ,
202481593293050 : 1072 ,
202481594518960 : 1073 ,
202481598412739 : 1074 ,
202481597561918 : 1075 ,
202481597145316 : 1076 ,
202481588476203 : 1077 ,
202481588991215 : 1078 ,
202481598240252 : 1079 ,
202481586930795 : 1080 ,
202481600336277 : 1081 ,
202481600281519 : 1082 ,
202481594535730 : 1083 ,
202481590239181 : 1084 ,
202481593705307 : 1085 ,
202481593329060 : 1086 ,
202481588007352 : 1087 ,
202481599403958 : 1088 ,
202481591104544 : 1089 ,
202481594876472 : 1090 ,
202481598031159 : 1091 ,
202481594630976 : 1092 ,
202481591756967 : 1093 ,
202481588627595 : 1094 ,
202481601040379 : 1095 ,
202481600327270 : 1096 ,
202481599071756 : 1097 ,
202481599656877 : 1098 ,
202481594609210 : 1099 ,
202481591694768 : 1100 ,
202481600215218 : 1101 ,
202481602084751 : 1102 ,
202481590956569 : 1103 ,
202481601426033 : 1104 ,
202481600146643 : 1105 ,
202481598108029 : 1106 ,
202481593142629 : 1107 ,
202481587700937 : 1108 ,
202481596352747 : 1109 ,
202481600749663 : 1110 ,
202481589387491 : 1111 ,
202481598968436 : 1112 ,
202481591856896 : 1113 ,
202481600320493 : 1114 ,
202481601536502 : 1115 ,
202481600928060 : 1116 ,
202481594208274 : 1117 ,
202481594931693 : 1118 ,
202481585889781 : 1119 ,
202481597952975 : 1120 ,
202481595588330 : 2001 ,
202481599275972 : 2002 ,
202481589174252 : 2003 ,
202481599323929 : 2004 ,
202481601180343 : 2005 ,
202481593534803 : 2006 ,
202481587091977 : 2007 ,
202481590930337 : 2008 ,
202481597056241 : 2009 ,
202481601995306 : 2010 ,
202481601738649 : 2011 ,
202481591115309 : 2012 ,
202481589620558 : 2013 ,
202481593583243 : 2014 ,
202481591721573 : 2015 ,
202481591447923 : 2016 ,
202481596578391 : 2017 ,
202481588471439 : 2018 ,
202481602112786 : 2019 ,
202481600284849 : 2020 ,
202481586745007 : 2021 ,
202481589056738 : 2022 ,
202481592457971 : 2023 ,
202481594981542 : 2024 ,
202481589921925 : 2025 ,
202481602030726 : 3001 ,
202481585910238 : 3002 ,
202481596909065 : 3003 ,
202481600826922 : 3004 ,
202481597889233 : 3005 ,
202481600158213 : 3006 ,
202481592905726 : 3007 ,
202481599235133 : 3008 ,
202481597396653 : 3009 ,
202481593034624 : 3010 ,
}

zoneCoordinatesDict = {
'1100' :  [(68,1266),(68,1023),(270,1023),(270,959),(459,959),(459,1110),(576,1110),(576,1078),(723,1078),(723,1266)],
'1110' :  [(1306,1161),(1306,1119),(1776,1119),(1776,1161)],
'1102' :  [(732,1264),(730,1065),(732,959),(956,959),(956,905),(1036,905),(1036,982),(1176,982),(1176,1109),(1292,1109),(1292,1163),(1052,1163),(1052,1264)],
'10198' :  [(63,947),(63,895),(500,895),(500,947)],
'10199' :  [(508,947),(508,895),(790,895),(790,947)],
'10001' :  [(642,853),(642,537),(683,537),(683,853)],
'10005' :  [(244,780),(244,735),(631,735),(631,780)],
'10009' :  [(165,849),(165,552),(236,552),(236,777),(208,777),(208,849)],
'10200' :  [(803,657),(803,535),(947,535),(947,657)],
'10201' :  [(60,521),(60,407),(289,407),(289,521)],
'10202' :  [(299,521),(299,407),(526,407),(526,521)],
'10203' :  [(536,521),(536,407),(763,407),(763,521)],
'10204' :  [(773,521),(773,407),(999,407),(999,521)],
'10205' :  [(1009,521),(1009,407),(1236,407),(1236,521)],
'10206' :  [(1246,521),(1246,407),(1473,407),(1473,521)],
'10207' :  [(1483,521),(1483,407),(1710,407),(1710,521)],
'10208' :  [(1720,521),(1720,407),(1869,407),(1869,521)],
'10398' :  [(60,395),(60,125),(95,125),(95,395)],
'10399' :  [(1824,310),(1824,20),(1871,20),(1871,310)],
'10401' :  [(61,90),(61,20),(310,20),(310,90)],
'10402' :  [(326,90),(326,20),(550,20),(550,90)],
'10403' :  [(562,90),(562,20),(786,20),(786,90)],
'10404' :  [(798,90),(798,20),(1023,20),(1023,90)],
'10405' :  [(1036,90),(1036,20),(1260,20),(1260,90)],
'10406' :  [(1271,90),(1271,20),(1496,20),(1496,90)],
'10407' :  [(1510,90),(1510,20),(1816,20),(1816,90)],
'1608' :  [(171,340),(171,157),(331,157),(331,340)],
'1610' :  [(409,340),(409,157),(569,157),(569,340)],
'1612' :  [(646,340),(646,157),(806,157),(806,340)],
'1614' :  [(882,340),(882,157),(1042,157),(1042,340)],
'1616' :  [(1119,340),(1119,157),(1280,157),(1280,340)],
'1618' :  [(1356,340),(1356,157),(1516,157),(1516,340)],
'1620' :  [(1593,340),(1593,157),(1753,157),(1753,340)],
'1214' :  [(698,619),(698,537),(789,537),(789,619)],
'1216' :  [(698,708),(698,630),(789,630),(789,708)],
'1218' :  [(698,797),(698,718),(789,718),(789,797)],
'1220' :  [(698,883),(698,807),(789,807),(789,883)],
'1224' :  [(486,883),(486,790),(562,790),(562,883)],
'1226' :  [(396,883),(396,790),(475,790),(475,883)],
'1228' :  [(308,883),(308,790),(386,790),(386,883)],
'1230' :  [(219,883),(219,790),(297,790),(297,883)],
'1202' :  [(60,883),(60,807),(155,807),(155,883)],
'1204' :  [(60,795),(60,719),(155,719),(155,795)],
'1206' :  [(60,707),(60,631),(155,631),(155,707)],
'1208' :  [(60,619),(60,535),(155,535),(155,619)],
'1222' :  [(573,883),(573,811),(629,811),(629,883)],
'1210' :  [(251,724),(251,536),(462,536),(462,724)],
'1212' :  [(474,723),(474,536),(539,536),(539,723)],
'1201' :  [(550,612),(550,536),(629,536),(629,612)],
'1291' :  [(551,725),(551,625),(630,625),(630,725)],
'1260' :  [(469,1099),(469,960),(556,960),(556,1099)],
'1300' :  [(803,930),(803,673),(947,673),(947,930)],
'1306' :  [(1007,735),(1007,535),(1176,535),(1176,735)],
'1308' :  [(959,892),(959,817),(1003,817),(1003,749),(1175,749),(1175,973),(1046,973),(1046,892)],
'1402' :  [(1190,1098),(1190,688),(1292,688),(1292,1098)],
'1400' :  [(1299,871),(1299,735),(1615,735),(1615,812),(1705,812),(1705,871)],
'1412' :  [(1321,705),(1321,557),(1449,557),(1449,705)],
'1414' :  [(1498,705),(1501,557),(1613,557),(1613,705)],
'1410' :  [(1190,677),(1190,536),(1292,536),(1292,677)],
'1418' :  [(1553,1108),(1553,907),(1613,907),(1613,991),(1688,991),(1688,1108)],
'1420' :  [(1299,1108),(1299,885),(1543,885),(1543,1108)],
'1124' :  [(1781,995),(1781,958),(1837,958),(1837,995)],
'1122' :  [(1785,943),(1781,894),(1868,894),(1868,943)],
'1159' :  [(1716,1107),(1716,919),(1770,919),(1770,1107)],
'1506' :  [(1719,883),(1719,747),(1870,747),(1870,883)],
'1514' :  [(1719,732),(1719,647),(1870,647),(1870,732)],
'1516' :  [(1719,635),(1719,549),(1870,549),(1870,635)],
'1508' :  [(1666,802),(1666,755),(1706,755),(1706,802)],
'1512' :  [(1645,720),(1645,669),(1707,669),(1707,720)],
'1518' :  [(1645,659),(1645,536),(1707,536),(1707,659)],
'1416' :  [(1629,961),(1629,885),(1705,885),(1705,961)],
'10500' :  [(958,805),(958,533),(992,533),(992,805)],
'20098' :  [(1158,438),(1158,394),(1408,394),(1408,438)],
'20099' :  [(1418,438),(1418,394),(1650,394),(1650,438)],
'20001' :  [(1234,615),(1234,447),(1277,447),(1277,615)],
'20005' :  [(1234,817),(1234,625),(1289,625),(1289,817)],
'20007' :  [(1605,817),(1605,625),(1649,625),(1649,817)],
'20009' :  [(1605,615),(1605,447),(1649,447),(1649,615)],
'20100' :  [(600,438),(600,394),(900,394),(900,438)],
'20598' :  [(47,438),(47,394),(299,394),(299,438)],
'20599' :  [(1978,430),(1978,394),(2218,394),(2218,430)],
'20698' :  [(45,352),(45,192),(81,192),(81,352)],
'20699' :  [(1665,430),(1665,394),(1967,394),(1967,430)],
'2100' :  [(907,586),(907,394),(1116,394),(1116,586)],
'2102' :  [(360,557),(360,394),(548,394),(548,557)],
'2402' :  [(1129,524),(1129,448),(1223,448),(1223,524)],
'2404' :  [(1291,524),(1289,448),(1382,448),(1382,524)],
'2406' :  [(1129,613),(1129,536),(1223,536),(1223,613)],
'2408' :  [(1291,613),(1289,536),(1382,536),(1382,613)],
'2410' :  [(1129,702),(1129,625),(1223,625),(1223,702)],
'2412' :  [(1129,819),(1129,714),(1223,714),(1223,819)],
'2426' :  [(1661,819),(1661,714),(1755,714),(1755,819)],
'2428' :  [(1661,702),(1661,625),(1755,625),(1755,702)],
'2430' :  [(1661,613),(1661,536),(1755,536),(1755,613)],
'2429' :  [(1661,524),(1661,448),(1755,448),(1755,524)],
'2150' :  [(1439,518),(1439,460),(1513,460),(1513,518)],
'2160' :  [(1525,492),(1525,448),(1587,448),(1587,492)],
'2422' :  [(1419,575),(1419,529),(1498,529),(1498,652),(1463,652),(1463,575)],
'2491' :  [(1393,652),(1393,590),(1450,590),(1450,652)],
'2424' :  [(1532,652),(1532,593),(1595,593),(1595,652)],
'2432' :  [(1532,580),(1532,514),(1595,514),(1595,580)],
'2420' :  [(1307,819),(1307,664),(1581,664),(1581,819)],
'2174' :  [(1766,819),(1766,440),(1962,440),(1962,819)],
'2176' :  [(1976,819),(1976,440),(2023,440),(2023,547),(2218,547),(2218,819)],
'2140' :  [(47,827),(47,569),(706,569),(706,827)],
'2200' :  [(87,302),(87,104),(2210,104),(2210,302)],
'2120' :  [(607,547),(607,449),(742,449),(742,547)],
'3100' :  [(485,97),(485,52),(1000,52),(1000,97)],
'3200' :  [(731,467),(731,237),(907,237),(907,167),(1139,167),(1139,467)],
'31301' :  [(40,496),(40,233),(290,233),(290,496)],
'31302' :  [(309,496),(309,233),(500,233),(500,496)],
'31303' :  [(525,496),(525,233),(700,233),(700,496)],
'3220' :  [(1171,490),(1171,52),(2205,52),(2205,490)],
'30098' :  [(42,98),(42,52),(456,52),(456,98)],
}
def convertImage(list_scan,sightVec): # sightVec[time,scanner], return sightImage[time,x,y,z]
    pixelsPerGrid = 30
    encoder = LabelEncoder()
    encoder.fit(list_scan)
    maxScannerCoordinateX = np.array(scannerCoordinatesDict.values())[:,0].max()
    maxScannerCoordinateY = np.array(scannerCoordinatesDict.values())[:,1].max()
    maxScannerZ = 3
    sightImage = np.zeros((len(sightVec),maxScannerCoordinateX/pixelsPerGrid+1,maxScannerCoordinateY/pixelsPerGrid+1,maxScannerZ))
    indxScannerX = np.array([scannerCoordinatesDict[tmp][0] for tmp in list_scan]) / pixelsPerGrid
    indxScannerY = np.array([scannerCoordinatesDict[tmp][1] for tmp in list_scan]) / pixelsPerGrid
    indxScannerZ = np.array([int(str(scannerIdDict[tmp])[0])-1 for tmp in list_scan])
    sightImage[:,indxScannerX,indxScannerY,indxScannerZ] = sightVec
#    import matplotlib.pyplot as plt
#    plt.pcolor(sightImage[0,:,:,0])
#    plt.colorbar()
#    plt.show()
#    sys.exit()
    return sightImage

def train_cann(df,trainTag=[4,5,6,7,8],testTag=[3,],loadTraj=False):
    scanner,tag,strength,timestamp,zone = np.array(df['scanner']),np.array(df['tag']),np.array(df['strength']),np.array(df['timestamp']),np.array(df['zone'],dtype=int)
    Nbatch = 100
    Nepoch = 100
    ## prepare training data
    indx = [False,]*len(df)
    for tg in trainTag:
        indx = np.logical_or(indx,tag==tg)
    list_scan = np.unique(scanner[indx])
    list_zone = np.unique(zone[indx])
    list_tag = np.unique(tag[indx])
    collectAll,collectZne,data = prepareTrainData1(list_scan,list_zone,scanner[indx],tag[indx],strength[indx],timestamp[indx],zone[indx],window=5)
    collectAll = convertImage(list_scan,collectAll)                ####

    ## prepare testing data
    indx = [False,]*len(df)
    for tg in testTag:
        indx = np.logical_or(indx,tag==tg)
    testAll,testZne,data = prepareTrainData1(list_scan,list_zone,scanner[indx],tag[indx],strength[indx],timestamp[indx],zone[indx],window=5)
    testAll = convertImage(list_scan,testAll)

    ## load trajectory
    if loadTraj:
        df = pandas.read_csv('traj_1.csv')
        df = df[df['strength']>-90] # neglect <-90 strength
        df = df.sort_values(by=['tag','timestamp']) # sort
        scannerT,tagT,strengthT,timestampT,zoneT = np.array(df['scanner']),np.array(df['tag']),np.array(df['strength']),np.array(df['timestamp']),np.array(df['zone'],dtype=int)
        testAll,testZne,data,label = prepareTrainData1Trajectory(list_scan,list_zone,scannerT,tagT,strengthT,timestampT,zoneT,window=5)
        testAll,testZne = generateSequenceTrajectory(list_zone,data,label)

    ## machine learning training
    warnings.simplefilter("ignore", DeprecationWarning)

    X = collectAll
    Y = collectZne
    encoder = OneHotEncoder()
    encoder.fit(Y.reshape(-1,1))
    encoded_Y = encoder.transform(Y.reshape(-1,1))
    X_train,y_train = X,encoded_Y
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train.reshape(len(X_train),-1)).reshape(X_train.shape)
    model = Sequential()
    #model.add(Dense(units = 200, kernel_initializer = 'uniform',activation = 'relu', input_dim=len(list_scan)))
    layers = {}
    model.add(Conv3D(32,kernel_size=(3,3,3),strides=(1,1,1),padding='valid',activation = 'relu',input_shape=(69,41,3,1)))
    model.add(MaxPooling3D(pool_size=(2,2,1)))
    model.add(Dropout(0.5))
    model.add(Conv3D(32,kernel_size=(3,3,1),strides=(1,1,1),padding='same',activation = 'relu'))
    model.add(MaxPooling3D(pool_size=(2,2,1)))
    model.add(Dropout(0.5))
    model.add(Conv3D(32,kernel_size=(3,3,1),strides=(1,1,1),padding='same',activation = 'relu'))
    model.add(MaxPooling3D(pool_size=(2,2,1)))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(len(list_zone),activation='softmax'))
    model.compile(loss='mean_squared_error',optimizer='adam',metrics=[])
    X_test = sc.transform(testAll.reshape(len(testAll),-1)).reshape(testAll.shape)
    y_test = encoder.transform(testZne.reshape(-1,1))
    ## Callback class for cross-validation
    ## Zone distance dict
    dictZoneDist = {}
    for room in roomConnect.keys():
        dictZoneDist[room] = np.array([ dictRoomDist[room][tmp] for tmp in list_zone ])
    class LossHistory(keras.callbacks.Callback):
        def __init__(self):
            self.predhis = []

        def on_epoch_end(self, batch, logs={}):
            y_pred = model.predict(X_test[...,None])
            yp = encoder.inverse_transform(y_pred[:,:]).squeeze()
            ## consider zone connection
#            if post[0]:
#                for n in range(len(testZne)):
#                    if n==0:
#                        yp[0] = encoder.inverse_transform(y_pred[:1,:])[0]
#                    elif testZne[n]==testZne[n-1]:
#                        y_pred[n,:] *= post[1]**(-dictZoneDist[yp[n-1]]*1.)
#                        yp[n] = encoder.inverse_transform(y_pred[n:n+1,:])[0]

            ## consider voting
#            if vote:
#                ypv = yp.copy()
#                for n in range(len(testZne)):
#                    ypv[n],num = Counter(yp[max(n-Nvote+1,0):n+1]).most_common(1)[0]
#                    if num<min(n+1,Nvote)*0.6:
#                        ypv[n] = ypv[n-1]
#                yp = ypv
            ## compute accuracy
            accuracy = len(yp[yp==testZne])*1./len(yp)
            nCorrect,nTotal = 0,0
            trueZneOld = testZne[0]
            for trueZne,predZne in zip(testZne,yp):
                if trueZne!=trueZneOld:
                    nCorrect,nTotal = 0,0
                if trueZne==predZne:
                    nCorrect += 1
                nTotal += 1
                trueZneOld = trueZne
            print 'accuracy:',accuracy
    history = LossHistory()
    model.fit(X_train[...,None], y_train.toarray(), batch_size=Nbatch,epochs=Nepoch,callbacks=[history],validation_data=(X_test,y_test))

    X_train = model.predict(X_train[...,None])
    X_test = model.predict(X_test[...,None])
    Xtrain = []
    Ytrain = []
    Xtest = []
    Ytest = []
    for n in range(len(X_train)-lookBack):
        Xtrain.append(X_train[n:n+lookBack].reshape(-1))
        Ytrain = y_train[n+lookBack]
    for n in range(len(X_test)-lookBack):
        Xtest.append(X_test[n:n+lookBack].reshape(-1))
        Ytest = y_test[n+lookBack]
    model = Sequential()
    model.add(Dense(units = 200, kernel_initializer = 'uniform',activation = 'relu', input_dim=len(list_scan)*lookBack))
    model.add(Dropout(0.5))
    model.add(Dense(units = 200, kernel_initializer = 'uniform',activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(list_zone),activation='softmax'))
    model.compile(loss='mean_squared_error',optimizer='adam',metrics=[])
    y_true = testZne
    class LossHistory(keras.callbacks.Callback):
        def __init__(self):
            self.predhis = []

        def on_epoch_end(self, batch, logs={}):
            y_pred = model.predict(X_test)
            yp = encoder.inverse_transform(y_pred[:,:]).squeeze()

            ## compute accuracy
            if loadTraj:
                acc = np.array([False,]*(len(yp)-10))
                for n in [-5,-4,-3,-2,-1,0,1,2,3,4,5]:
                    acc = np.logical_or(acc, yp[5:-5]==y_true[5-n:len(yp)-5-n])
    history = LossHistory()
    model.fit(Xtrain[:,:], ytrain.toarray(), batch_size=Nbatch,epochs=Nepoch,callbacks=[history])
    
    return list_scan,list_zone,model
    
def trilaterate3D(distances):
    p1=np.array(distances[0][:3])
    p2=np.array(distances[1][:3])
    p3=np.array(distances[2][:3])
    p4=np.array(distances[3][:3])
    r1=distances[0][-1]
    r2=distances[1][-1]
    r3=distances[2][-1]
    r4=distances[3][-1]
    e_x=(p2-p1)/np.linalg.norm(p2-p1)
    i=np.dot(e_x,(p3-p1))
    e_y=(p3-p1-(i*e_x))/(np.linalg.norm(p3-p1-(i*e_x)))
    e_z=np.cross(e_x,e_y)
    d=np.linalg.norm(p2-p1)
    j=np.dot(e_y,(p3-p1))
    x=((r1**2)-(r2**2)+(d**2))/(2*d)
    y=(((r1**2)-(r3**2)+(i**2)+(j**2))/(2*j))-((i/j)*(x))
    z1=np.sqrt(r1**2-x**2-y**2)
    z2=np.sqrt(r1**2-x**2-y**2)*(-1)
    ans1=p1+(x*e_x)+(y*e_y)+(z1*e_z)
    ans2=p1+(x*e_x)+(y*e_y)+(z2*e_z)
    dist1=np.linalg.norm(p4-ans1)
    dist2=np.linalg.norm(p4-ans2)
    if np.abs(r4-dist1)<np.abs(r4-dist2):
        return ans1
    else:
        return ans2

def train_trilaterate(df,testTag=[3,],loadTraj=False):
    scanner,tag,strength,timestamp,zone = np.array(df['scanner']),np.array(df['tag']),np.array(df['strength']),np.array(df['timestamp']),np.array(df['zone'],dtype=int)
    ## prepare testing data
    indx = [False,]*len(df)
    for tg in testTag:
        indx = np.logical_or(indx,tag==tg)
    list_scan = np.unique(scanner[indx])
    list_zone = np.unique(zone[indx])
    list_tag = np.unique(tag[indx])
    testAll,testZne,data = prepareTrainData1(list_scan,list_zone,scanner[indx],tag[indx],strength[indx],timestamp[indx],zone[indx],window=5)
    ## scanner coordinate array
    scannerCoordinatesArray = np.zeros((len(list_scan),3))
    for n,scan in enumerate(list_scan):
        floor = int(str(scannerIdDict[scan])[0])
        if floor==0:
            scannerCoordinatesArray[n,:] = scannerCoordinatesDict[scan]+[floor,]
        elif floor==1:
            scannerCoordinatesArray[n,:] = [scannerCoordinatesDict[scan][0]+197, scannerCoordinatesDict[scan][1]+502, floor] # align floor
        elif floor==2:
            scannerCoordinatesArray[n,:] = [scannerCoordinatesDict[scan][0]+181, scannerCoordinatesDict[scan][1]+844, floor]
    scannerCoordinatesArray[:,:2] *= 0.0511 # 1/16 inch to 1 foot
    scannerCoordinatesArray[:,2] *= 4. # 4 meter ceiling height
    ## predict zone
    for zne in zoneCoordinatesDict.keys():
        for n in range(len(zoneCoordinatesDict[zne])):
            zoneCoordinatesDict[zne][n] = (zoneCoordinatesDict[zne][n][0]*0.0511, zoneCoordinatesDict[zne][n][1]*0.0511)
    from shapely.geometry import Point, Polygon
    for sight,zneTrue in zip(testAll,testZne):
        indx = np.argsort(sight)[:-5:-1]
        distances = np.zeros((4,4))
        distances[:,-1] = 10**((-(sight[indx]*60.-90.)-59)/20)
        distances[:,:-1] = scannerCoordinatesArray[indx]
        pointCoordinates = trilaterate3D(distances)
        for zne in zoneCoordinatesDict.keys():
            floor = int(str(scannerIdDict[scan])[0])
            polygon = Polygon(zoneCoordinatesDict[zne])
            point = Point(pointCoordinates[:2])
            if polygon.contains(point) and (floor-1)*4.<pointCoordinates[2]<floor*4.:
                znePred = zne
                break
    return list_scan,list_zone,0

if ifReTrainModel:
    df1 = pandas.read_csv('train.csv')
    df2 = pandas.read_csv('valid.csv')
    df = pandas.concat([df1,df2], axis=0, ignore_index=True)
    df = df[~df['zone'].isnull()] # no NULL
    df = df[df['strength']>-90] # neglect <-90 strength
    df = df.sort_values(by=['zone','tag','timestamp']) # sort

    tags = [4,5,6,7,8]
    for tg in [3],:
    #for tg in [1,2],:
        list_scan,list_zone,model = train_lstm(df,[tmp for tmp in tags if tmp not in tg], tg, post=[False,40], augTrain=False, augTest=False, loadTraj=True, window=1)
        #list_scan,list_zone,model = train_cann(df,[tmp for tmp in tags if tmp not in tg], tg, loadTraj=False)
    sys.exit()

