import numpy as  np
import csv

#读取字符串表格
def ReadMyCsv(SaveList, fileName):
    csv_reader = csv.reader(open(fileName,encoding='utf-8'))
    for row in csv_reader:
        #print(row)
        SaveList.append(row)
    return


#保存数据
def StorFile(data, fileName):
    with open(fileName, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    return

def MyEdgeNumExchange():
    AllNode = []
    ReadMyCsv(AllNode, 'AllNode.csv')
    #print(len(AllNode), AllNode[1])

    AllEdge = []
    ReadMyCsv(AllEdge, 'AllEdge.csv')
    #print(len(AllEdge), AllEdge[1], AllEdge[1][1])

    NodeEnum = list(enumerate(AllNode))
    #print(NodeEnum[1][1])

    AllEdgeNum = []
    for [node1,node2] in AllEdge:
        #print(node1)
        #print(node2)
        pair = []
        for index, node in NodeEnum:
            #print(index)
            #print(node)
            if node[0] == node1:
                node1 = index
                #print(index)
                pair.append(index)
                break
        for index, node in NodeEnum:
            if node[0] == node2:
                node2 = index
                pair.append(index)
                break
        AllEdgeNum.append(pair)
    StorFile(AllEdgeNum, 'AllEdgeNum.csv')
    return


if __name__ == '__main__':
    MyEdgeNumExchange()

