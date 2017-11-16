import os
import sys
from glob import glob
import cv2  
from numpy import *    
import Image
import numpy as np
from collections import Counter
caffe_root = '/home/caojiajiong/workspace/caffe-windows/'
sys.path.insert(0, os.path.join(caffe_root+'python'))
import caffe
import lmdb     
screenLevels = 255.0
width_i = 40
height_i = 8
def flag_read(filename):
    print filename.split('/')[-1]
    lines = open(filename).readlines()
    lines = lines[2:-3]
    flags = np.array([int(l.split(': ')[-1]) for l in lines])
    for k,c in Counter(flags).items():
        print k, c/(1.0*len(flags))
    quan = 1
    if quan==1:
        print flags[:10]
        for i in range(len(flags)):
            if flags[i]<=1:
                flags[i] = 0
            else:
                flags[i] = np.ceil(flags[i]/7.0)
        print flags[:10]
    return flags
def y_read(filename, dims, numfrm):
    fp = open(filename, 'rb')
    Y = np.zeros((numfrm, 1, height_i, width_i))
    print dims[0], dims[1]
    #Yt = zeros((dims[0], dims[1]), uint8, 'C')
    for i in range(numfrm):
        fp.seek(dims[0]*dims[1]*i, 0)
        Yt = np.array([ord(d) for d in fp.read(8*dims[1])]).reshape((8, dims[1])) 
        Y[i, 0, :, :] = Yt[-1*height_i:, :width_i]
    return Y
def gen_lmdb(filenames1, filenames2, savename):
    data_train = np.zeros((2000000, 1, height_i, width_i), dtype=np.uint8)
    labels_train = np.zeros(2000000, dtype=np.int64)
    data_val = np.zeros((200000, 1, height_i, width_i), dtype=np.uint8)
    labels_val = np.zeros(200000, dtype=np.int64)
    N_train = 0
    N_val = 0
    test = 0 
    if test==1:
        N_train = 35*50000
        N_val = 35*5000
    else:
        for i in range(len(filenames1)):
            labels = flag_read(filenames2[i])
            data = y_read(filenames1[i], [16, 320], len(labels))
            print len(labels)
            cnt_val = int(np.floor(len(labels)*0.1))
            cnt_train = len(labels) - cnt_val
            #print cnt_train
            #print N_train, N_train+cnt_train
            data_train[N_train:N_train+cnt_train] = data[:cnt_train]
            labels_train[N_train:N_train+cnt_train] = labels[:cnt_train]
            data_val[N_val:N_val+cnt_val]= data[cnt_train:]
            labels_val[N_val:N_val+cnt_val] = labels[cnt_train:]
            N_train = N_train+cnt_train
            N_val = N_val+cnt_val

    X_train = data_train[:35*50000].copy()
    y_train = labels_train[:35*50000].copy()
    print X_train.shape, data_train.shape
    for i in range(5):
        idx = labels_train[np.where(labels_train[:N_train]==i)]
        p = np.random.permutation(len(idx))
        #print len(idx)
        idx = [idx[k] for k in p]
        n = len(idx)
        if n==0:
            continue
	#print k, i
        for j in range(350000):
            #print X_train[i*50000+j].shape
            #print data_train[idx[j%n]].shape
            #print idx[j%n]
            X_train[i*350000+j] = data_train[idx[j%n]]
            y_train[i*350000+j] = labels_train[idx[j%n]]
    map_size = 35*50000*height_i*width_i*10
    idx = np.random.permutation(35*50000)
    env_train = lmdb.open(savename+'_train', map_size=map_size)
    print len(idx)
    with env_train.begin(write=True) as txn:
        for i in range(35*50000):
            datum = caffe.proto.caffe_pb2.Datum()
            datum.channels = X_train.shape[1]
            datum.height = X_train.shape[2]
            datum.width = X_train.shape[3]
            datum.data = X_train[idx[i]].tobytes()
            datum.label = int(y_train[idx[i]])
            str_id = '{:08}'.format(i)
            txn.put(str_id.encode('ascii'), datum.SerializeToString())

    X_val = data_val[:N_val]
    y_val = labels_val[:N_val]
    map_size = N_val*height_i*width_i*10
    idx = np.random.permutation(N_val)
    env_val = lmdb.open(savename+'_val', map_size=map_size)
    print N_val, len(idx)
    with env_val.begin(write=True) as txn:
        for i in range(N_val):
            datum = caffe.proto.caffe_pb2.Datum()
            datum.channels = X_val.shape[1]
            datum.height = X_val.shape[2]
            datum.width = X_val.shape[3]
            datum.data = X_val[idx[i]].tobytes()
            datum.label = int(y_val[idx[i]])
            str_id = '{:08}'.format(i)
            txn.put(str_id.encode('ascii'), datum.SerializeToString())




def yuv_import(filename,dims,numfrm,startfrm):    
    fp=open(filename,'rb')    
    blk_size = prod(dims) *3/2    
    fp.seek(blk_size*startfrm,0)    
    Y=[]    
    U=[]    
    V=[]    
    print dims[0]    
    print dims[1]    
    d00=dims[0]//2    
    d01=dims[1]//2    
    print d00    
    print d01    
    Yt=zeros((dims[0],dims[1]),uint8,'C')    
    Ut=zeros((d00,d01),uint8,'C')    
    Vt=zeros((d00,d01),uint8,'C')    
    for i in range(numfrm):    
        for m in range(dims[0]):    
            for n in range(dims[1]):    
                #print m,n    
                Yt[m,n]=ord(fp.read(1))    
        for m in range(d00):    
            for n in range(d01):    
                Ut[m,n]=ord(fp.read(1))    
        for m in range(d00):    
            for n in range(d01):    
                Vt[m,n]=ord(fp.read(1))    
        Y=Y+[Yt]    
        U=U+[Ut]    
        V=V+[Vt]    
    fp.close()    
    return (Y,U,V)    
if __name__ == '__main__':  
    filenames1 = sorted(glob('/data1/caojiajiong/CNN_Encoding/DL_intra_mode/train_data/y/sample/22/*.yuv'))
    filenames2 = sorted(glob('/data1/caojiajiong/CNN_Encoding/DL_intra_mode/train_data/y/flag/22/*.txt'))
    print len(filenames1)	
    #print [(filenames1[i], filenames2[i]) for i in range(len(filenames1))]
    savedir = '/data1/caojiajiong/CNN_Encoding/DL_intra_mode/train_data/y/lmdb/'
    if not os.path.isdir(savedir):
        os.mkdir(savedir)
    savename = '/data1/caojiajiong/CNN_Encoding/DL_intra_mode/train_data/y/lmdb/22_balance_5'
    gen_lmdb(filenames1, filenames2, savename) 
