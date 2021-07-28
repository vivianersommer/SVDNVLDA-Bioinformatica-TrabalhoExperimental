import numpy as np
import csv
import random

def ReadMyCsv(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:
        SaveList.append(row)
    return


def ReadMyCsv2(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:
        for i in range(len(row)):
            row[i] = int(row[i])
        SaveList.append(row)
    return

def StorFile(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    return

def myPosSamp():

    #由RandoList打乱EdgeNum得到PositiveSample
    RandomLisrGroup = []
    ReadMyCsv2(RandomLisrGroup, 'RandomListGroup.csv')
    #print(np.shape(RandomLisrGroup))

    AllEdgeNum = []
    ReadMyCsv(AllEdgeNum, 'AllEdgeNum.csv')
    #print(np.shape(AllEdgeNum))

    PositiveSample = []
    counter = 0
    while counter < len(RandomLisrGroup):
        counter1 = 0
        while counter1 < len(RandomLisrGroup[counter]):
            PositiveSample.append(AllEdgeNum[RandomLisrGroup[counter][counter1]])
            counter1 = counter1 + 1
        counter = counter + 1

    StorFile(PositiveSample, 'PositiveSample.csv')
    return

def myNegSamp():
    #由AssociationMatrix和PositiveSamples得到NegativeSamples

    AssociationMatrix = []
    ReadMyCsv(AssociationMatrix, 'AssociationMatrix.csv')
    PositiveSample = []
    ReadMyCsv(PositiveSample, 'PositiveSample.csv')

    NegativeSample = []
    counterN = 0
    while counterN < len(PositiveSample):
        counter1 = random.randint(0, len(AssociationMatrix)-1)
        counter2 = random.randint(0,len(AssociationMatrix[counter1])-1)

        flag1 = 0
        counter3 = 0
        while counter3 < len(PositiveSample):   #正样本中是否存在
            if counter1 == PositiveSample[counter3][0] and counter2 == PositiveSample[counter3][1]:
                flag1 = 1
                break
            counter3 = counter3 + 1
        if flag1 == 1:
            continue

        flag2 = 0
        counter4 = 0
        while counter4 < len(NegativeSample): #在已选负样本中没有，防止重复
            if counter1 == NegativeSample[counter4][0] and counter2 == NegativeSample[counter4][1]:
                flag2 = 1
                break
            counter4 = counter4 + 1
        if flag2 == 1:
            continue

        if (flag1==0 & flag2==0):
            Pair = []
            Pair.append(counter1)
            Pair.append(counter2)
            NegativeSample.append(Pair)

            #print(counterN)
            counterN = counterN + 1

    print(len(NegativeSample))
    StorFile(NegativeSample, 'NeagtiveSample.csv')

    return




if __name__ == '__main__':
    myPosSamp()
    myNegSamp()
