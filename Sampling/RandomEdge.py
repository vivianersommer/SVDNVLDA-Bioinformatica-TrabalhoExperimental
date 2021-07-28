import numpy as np
import csv
import math
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
    with open(fileName, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    return

def partition(ls, size):
    #按给定size分割列表
    #ls = [1,2,3,4,5,6,7,8,10,11,2,2,4,6,7,7,2,6]
    #print(partition(ls, 5))
    #   [[1, 2, 3, 4, 5], [6, 7, 8, 10, 11], [2, 2, 4, 6, 7], [7, 2, 6]]
    return [ls[i:i+size] for i in range(0, len(ls), size)]

def RandomEdge():
    AllNodeNum = []     #所有节点编号1~节点数量
    ReadMyCsv(AllNodeNum, 'AllNodeNum.csv')

    AllEdgeNum = []     #EdgeNumExchange函数生成的数值化节点对
    ReadMyCsv(AllEdgeNum, 'AllEdgeNum.csv')

    #关联矩阵（下三角矩阵）
    AssociationMatrix = []
    counter = 0
    while counter < len(AllNodeNum):
        Row = []
        counter1 = 0
        while counter1 <= counter:
            Row.append(0)
            counter1 = counter1 + 1
        AssociationMatrix.append(Row)
        counter = counter + 1

    counter = 0
    while counter < len(AllEdgeNum):
        PairA = AllEdgeNum[counter][0]
        PairB = AllEdgeNum[counter][1]
        counter1 = 0
        while counter1 < len(AllNodeNum):
            if int(PairA) == int(AllNodeNum[counter1][0]):
                counterA = counter1
                break
            counter1 = counter1 + 1

        counter2 = 0
        while counter2 < len(AllNodeNum):
            if int(PairB) == int(AllNodeNum[counter2][0]):
                counterB = counter2
                break
            counter2 = counter2 + 1

        if counterA < counterB:
            temp = counterB
            counterB = counterA
            counterA = temp

        AssociationMatrix[counterA][counterB] = 1
        counter = counter + 1

    StorFile(AssociationMatrix, 'AssociationMatrix.csv')

    #生成0~len(AllEdgeNum)之间的随机采样并分为五组 保存随机数列
    RandomList = random.sample(list(range(0,len(AllEdgeNum))), len(AllEdgeNum))
    RandomListGroup = partition(RandomList, math.ceil(len(RandomList) / 10))
    StorFile(RandomListGroup, 'RandomListGroup.csv')

    return





if __name__ == "__main__":
    RandomEdge()