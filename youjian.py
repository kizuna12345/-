import numpy as np
from functools import reduce

def loadDataSet():
    postinglist=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],      #切分的词条
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec =[0,1,0,1,0,1]
    return postinglist,classVec


def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return  list(vocabSet)

def setOfWords2Vec(vocabList,inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]+=1
        else:
            print("the word:%s is not in my vocabulary" % word)
        return returnVec

def fit(trainMatrix,trainCategory):
    numTrainDocs=len(trainMatrix)
    numWords=len(trainMatrix[0])
    pAbusive=sum(trainCategory)/float(numTrainDocs)
    p0Num=np.ones(numWords)
    p1Num=np.ones(numWords)
    p0Denom=2.0
    p1Denom=2.0
    for i in range(numTrainDocs):
        if trainCategory[i]==1:
            p1Num+=trainMatrix[i]
            p1Denom+=sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect=np.log(p1Num/p1Denom)
    p0Vect=np.log(p0Num/p1Denom)
    return p0Vect,p1Vect,pAbusive

def predict(vec2Classify,p0Vec,p1Vec,pClass1):
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)
    p0 = sum(vec2Classify * p1Vec) + np.log(1.0 - pClass1)
    print('p0:', p0)
    print('p1:', p1)
    if p1 > p0:
        return 1
    else:
        return 0


def checkNB():
    list0posts,listClasses=loadDataSet()
    myVocabList=createVocabList(list0posts)
    trainMat=[]
    for postinDoc in list0posts:
        trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
    p0V,p1V,pAb=fit(np.array(trainMat),np.array(listClasses))
    testEntry=['love','my','dalmation']
    thisDoc=np.array(setOfWords2Vec(myVocabList,testEntry))
    if predict(thisDoc,p0V,p1V,pAb):
        print(testEntry,'侮辱类')
    else:
        print(testEntry,'非侮辱类')



if __name__ == '__main__':
    postingtList,classVec=loadDataSet()
    print('postingList:\n',postingtList)
    myVocabList=createVocabList(postingtList)
    print('myVocabList:\n',myVocabList)
    trainMat=[]
    for postinDoc in postingtList:
        trainMat.append(setOfWords2Vec(myVocabList,postinDoc))

    p0V,p1V,pAb=fit(trainMat,classVec)

    checkNB()

