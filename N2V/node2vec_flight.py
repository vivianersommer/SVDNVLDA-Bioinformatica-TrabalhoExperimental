#coding=utf-8
import networkx as nx
from node2vec_.node2vec import Node2Vec
import csv

def ReadMyCsv(SaveList, fileName):
    csv_reader = csv.reader(open(fileName, encoding='utf-8'))
    for row in csv_reader:
        SaveList.append(row)
    return

def StorFile(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    return

def ReadMyCsv(SaveList, fileName):
    csv_reader = csv.reader(open(fileName, encoding='utf-8'))
    for row in csv_reader:
        SaveList.append(row)
    return

def StorFile(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    return

G = nx.Graph()
# 加入训练的边
AllEdge = []
ReadMyCsv(AllEdge, r"../Data/AllAsso.csv")
counter1 = 0
while counter1 < len(AllEdge):
    temp = tuple(AllEdge[counter1])
    G.add_edge(*temp)  # 一次添加一条边
    # print('图中边的个数', graph.number_of_edges())
    # print(counter1)
    counter1 = counter1 + 1
node2vec = Node2Vec(G,dimensions=256,walk_length=10,num_walks=80,p=0.25, q=4,workers=4)
model1 =node2vec.fit(window=5,min_count=1, batch_words=4)
model1.wv.save_word2vec_format('embeddings_node2vec_256.vector')
