import os
import cv2
import numpy as np
import base64
import math
from struct import unpack
import pickle
from collections import Counter
import random
import scipy.io as sio
import scipy
def get_feature(img):
    os.system('curl -F "img=@%s" http://biscore.dev.alipay.net:7080/extract >> %s.txt' % (img,img))

def get_features(txt):
    lines = open(txt).readlines()
    lines = [l.split()[0] for l in lines]
    
    for i in xrange(lines):
        print i
        get_feature(lines[i])
def b64_cmp(a, b):
    if len(a) == 3200 and len(b) == 3200:
        feature_a = unpack('600f', base64.b64decode(a))
        feature_b = unpack('600f', base64.b64decode(b))
        dist = np.linalg.norm(np.asarray(feature_a)-np.asarray(feature_b))
        score = (1.0 / (1.0 + math.exp(0.00174828930217 * dist * dist - 3.12728738249))) * 100;
        return score
    else:
        return -100
def ExtractFeature(txt):
    fid = open(txt, 'r')
    data = fid.readlines()
    fid.close()
    dictRet = eval(data[0].replace('null', 'None').replace('true', '1').replace('false', '0'))
    if(dictRet['faces'] != None):
        dictFace = eval(str(dictRet['faces'][0]))
        return dictFace['feature']
    else:
        return ''
def ExtractFaces(txt):
    data = open(txt).readlines()
    dictRet = eval(data[0].replace('"label":null', '"label":None').replace('"faces":null', '"faces":None').replace('"result":true', '"result":1').replace('"result":false', '"result":0'))
    faces = dictRet['faces']
    if faces == None:
        return None
    else:
        maxIdx = 0
        maxScore = 0
        for i in range(len(faces)):
            if float(faces[i]['quality_score']) > maxScore:
                maxIdx = i
                maxScore = float(faces[i]['quality_score'])
        return faces[maxIdx]
def SaveObj(name, obj):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, 2)
def LoadObj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)
def Align(imgFile, srcL, savePath):
    img = cv2.imread(imgFile)
    dst = np.asarray([[30.2946, 51.6963], [65.5318, 51.5014], [48.0252, 71.7366], [33.5493, 92.3655], [62.7299, 92.2041]])
    srcLandmark = np.zeros((5, 2))
    srcLandmark = np.asarray([[srcL[0], srcL[1]], [srcL[2], srcL[3]], [srcL[4], srcL[5]], [srcL[6], srcL[7]], [srcL[8], srcL[9]]])
    Y = dst.flatten()
    a = []
    for i in srcLandmark:
        a.append([i[0], -i[1], 1, 0])
        a.append([i[1], i[0], 0, 1])
    a = np.asarray(a)
    inv = np.dot(np.linalg.inv(np.dot(a.T, a)), a.T)
    X = np.dot(inv, Y)
    trans = np.asarray([[X[0], -X[1], X[2]], [X[1], X[0], X[3]]])
    img = cv2.warpAffine(img, trans, (96, 112))
    cv2.imwrite(savePath, img, [cv2.IMWRITE_JPEG_QUALITY, 100])
    print savePath
def Align144(imgFile, srcL, savePath):
    img = cv2.imread(imgFile)
    dst = np.asarray([[24+30.2946, 51.6963], [24+65.5318, 51.5014], [24+48.0252, 71.7366], [24+33.5493, 92.3655], [24+62.7299, 92.2041]])
    srcLandmark = np.zeros((5, 2))
    srcLandmark = np.asarray([[srcL[0], srcL[1]], [srcL[2], srcL[3]], [srcL[4], srcL[5]], [srcL[6], srcL[7]], [srcL[8], srcL[9]]])
    Y = dst.flatten()
    a = []
    for i in srcLandmark:
        a.append([i[0], -i[1], 1, 0])
        a.append([i[1], i[0], 0, 1])
    a = np.asarray(a)
    inv = np.dot(np.linalg.inv(np.dot(a.T, a)), a.T)
    X = np.dot(inv, Y)
    trans = np.asarray([[X[0], -X[1], X[2]], [X[1], X[0], X[3]]])
    img = cv2.warpAffine(img, trans, (144, 144))
    cv2.imwrite(savePath, img, [cv2.IMWRITE_JPEG_QUALITY, 100])
    print savePath
def AllPairs(xl, yl, T):
    allPairs = []
    for x in xl:
        for y in yl:
            if(x > y):
                allPairs.append((y, x))
            else:
                allPairs.append((x, y))
    allPairs = [k for (k, v) in Counter(allPairs).iteritems()]
    if(T == 'p'):
        return allPairs
    num = 10*min(len(xl), len(yl))
    perm = np.random.permutation(len(allPairs))
    return [allPairs[perm[i]] for i in xrange(num)]

def GenPairs(txtFile, saveName, num):
    lines = open(txtFile).readlines()
    ids = [int(line.split(' ')[-1].strip()) for line in lines]
    idx = np.array(ids).argsort()
    sortedIds = [ids[idx[i]] for i in range(len(idx))]
    posPairs = []
    negPairs = []
    curPos = 0
    postPos = 0
    lastPos = sortedIds.index(max(sortedIds))
    for i in range(lastPos):
        curPos = sortedIds.index(sortedIds[i])
        nextId = sortedIds[i]+1
        while(True):
            try:
                postPos = sortedIds.index(nextId)
                break
            except ValueError:
                nextId += 1
        #postPos = sortedIds.index(sortedIds[i]+1)
        count = 0
        while count < num:
            n = random.randint(curPos, postPos-1)
            if n != i:
                count += 1
                posPairs.append((min(idx[i], idx[n]), max(idx[i], idx[n])))
        count = 0
        while count < num:
            n = random.randint(0, len(ids)-1)
            if n >= curPos and n < postPos:
                continue
            else:
                negPairs.append((min(idx[i], idx[n]), max(idx[i], idx[n])))
                count+=1
        if i%1000 == 999:
            print i+1
    #last id
    for i in range(lastPos, len(sortedIds)):
        count = 0
        while count < num:
            n = random.randint(lastPos, len(sortedIds)-1)
            if n != i:
                count += 1
                posPairs.append((min(idx[i], idx[n]), max(idx[i], idx[n])))
        count = 0
        while count < num:
            n = random.randint(0, len(ids)-1)
            if n >= lastPos:
                continue
            else:
                negPairs.append((min(idx[i], idx[n]), max(idx[i], idx[n])))
                count+=1
        if (i+lastPos)%1000 == 999:
            print i+1
    correctPosPairs = []
    correctNegPairs = []
    for k, c in Counter(posPairs).items():
        correctPosPairs.append(k)
    for k,c in Counter(negPairs).items():
        correctNegPairs.append(k)
    sio.savemat(saveName+'_pos.mat', {'posPairs':correctPosPairs})
    sio.savemat(saveName+'_neg.mat', {'negPairs':correctNegPairs})
def GeneratePairs(txtlist, saveName):
    idList = [l[6:10] for l in txtlist]
    uniqueId = [k for (k,v) in Counter(idList).iteritems()]
    posPairs = []
    negPairs = []
    for ID in uniqueId:
        idx = [i for i in xrange(len(idList)) if idList[i]==ID]
        if type(idx) == int:
            idx = [idx]
        nidx = []
        for i in xrange(len(idList)):
            count = 0
            for j in xrange(len(idx)):
                if i != idx[j]:
                    count += 1
            if count == len(idx):
                nidx.append(i)
        if len(idx) != 1:
            posPairs = posPairs + AllPairs(idx, idx, 'p')
        negPairs = negPairs + AllPairs(idx, nidx, 'n')
    for i in xrange(len(posPairs)):
        a, b = posPairs[i]
        if(a > b):
            posPairs[i] = (b, a)
    for i in xrange(len(negPairs)):
        a, b = negPairs[i]
        if(a > b):
            negPairs[i] = (b, a)
    posPairs = [k for (k, v) in Counter(posPairs).iteritems()]
    negPairs = [k for (k, v) in Counter(negPairs).iteritems()]
    SaveObj(name='E:/DataSet/Face/features_indian-data-300/indian-data-300/' + saveName + '_pos', obj=posPairs)
    SaveObj(name='E:/DataSet/Face/features_indian-data-300/indian-data-300/' + saveName + '_neg', obj=negPairs)
def EvaluatePairs(posPairs, negPairs, features):
    scores = []
    labels = []
    for pair in posPairs:
        if(len(features[pair[0]]) != 3200 or len(features[pair[1]]) != 3200):
            continue
        scores.append(b64_cmp(features[pair[0]], features[pair[1]]))
        labels.append(1)
    for pair in negPairs:
        if(len(features[pair[0]]) != 3200 or len(features[pair[1]]) != 3200):
            continue
        scores.append(b64_cmp(features[pair[0]], features[pair[1]]))
        labels.append(0)
    return scores, labels
def ComputeScoresFacepp(posPairsF, negPairsF, featuresF):
    features = LoadObj(featuresF)
    data = sio.loadmat(posPairsF)
    posPairs = data['posPairs']
    data = sio.loadmat(negPairsF)
    negPairs = data['negPairs']
    scores = [[], []]
    for posPair in posPairs:
        print posPair
        scores[0].append(b64_cmp(features[posPair[0]], features[posPair[1]]))
        scores[1].append(1)
    for negPair in negPairs:
        scores[0].append(b64_cmp(features[negPair[0]], features[negPair[1]]))
        scores[1].append(0)
    return scores
def ComputeScores(posPairsF, negPairsF, featuresF):
    data = sio.loadmat(featuresF)
    features = data['features']
    data = sio.loadmat(posPairsF)
    posPairs = data['posPairs']
    data = sio.loadmat(negPairsF)
    negPairs = data['negPairs']
    for i in range(len(features)):
        features[i] = features[i] / np.linalg.norm(features[i])
    posNum = len(posPairs)
    negNum = len(negPairs)
    scores = [[], []]
    for posPair in posPairs:
        scores[0].append(scipy.dot(np.array(features[posPair[0]]), np.array(features[posPair[1]])))
        scores[1].append(1)
    for negPair in negPairs:
        scores[0].append(scipy.dot(np.array(features[negPair[0]]), np.array(features[negPair[1]])))
        scores[1].append(0)
    return scores
