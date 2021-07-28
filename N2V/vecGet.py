import numpy as np
import pandas as pd

def getVec(fileName, dim):
    file = pd.read_csv(fileName, header=None)
    fileList = np.array(file).tolist()
    #print(np.shape(fileList))  (2007, 1)
    valueArr_str = []
    nameArr = []
    for vec in fileList:
        break_vec = vec[0].split(' ')
        feature = break_vec[-dim:]
        name = ' '.join(break_vec[:-dim])
        valueArr_str.append(feature)
        nameArr.append(name)

    valueArr = np.zeros((len(valueArr_str), dim))
    for i in range(len(valueArr_str)):
        for j in range(dim):
            ele = float(valueArr_str[i][j])
            valueArr[i][j] = ele

    np.savetxt('N2VFeature_%d.csv'%(dim), valueArr)
    np.savetxt('N2VName_%d.txt'%(dim), nameArr, fmt='%s')


if __name__ == '__main__':
    getVec('N2V16.csv', 16)
    getVec('N2V32.csv', 32)
    getVec('N2V64.csv', 64)
    getVec('N2V128.csv', 128)
    getVec('N2V256.csv', 256)