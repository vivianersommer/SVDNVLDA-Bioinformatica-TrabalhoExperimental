import numpy as np
import pandas as pd

def getMat():
    assocFile = pd.read_csv(r'../Data/LncDis.csv', header=None, delimiter='\t')
    lncName = pd.read_csv(r'../Data/LncName.csv', header=None, delimiter='\t')
    disName = pd.read_csv(r'../Data/DisName.csv', header=None, delimiter='\t')

    assocList = np.array(assocFile).tolist()
    lncList = np.array(lncName).tolist()
    disList = np.array(disName).tolist()
    # 将嵌套lncName转换成列表
    # ['21A', '91H', 'AATBC', ...]
    newLncName = []
    for lnc in lncList:
        newLncName.append(lnc[0])

    # 去掉疾病名称前双引号
    # ['Abortion, Spontaneous', 'Acquired Immunodeficiency Syndrome',...]
    newDisName = []
    for dise in disList:
        new_name = dise[0].replace("'", '')
        newDisName.append(new_name)
    # 将关联数据转化为嵌套列表，去掉疾病名称前双引号
    # #[['21A', 'Astrocytoma'], ['21A', 'Neoplasms'],...]
    newAssocList = []
    for assoc in assocList:
        # print(assoc)
        new_assoc = []
        assoc_row = assoc[0].split(',', 1)
        # assoc_row[0]       lncRNA名称
        dis_dequote = assoc_row[1].replace('"', '')  # 疾病名称，去除引号
        new_assoc.append(assoc_row[0])
        new_assoc.append(dis_dequote)
        newAssocList.append(new_assoc)

    assocMat = np.zeros((len(newLncName), len(newDisName)))
    count = 0
    for i in range(len(newLncName)):
        for j in range(len(newDisName)):
            if [newLncName[i], newDisName[j]] in newAssocList:
                assocMat[i][j] = 1
                count += 1
    print(np.shape(assocMat), count)
    np.savetxt('SVDMat.csv', assocMat)
    # np.savetxt('newLNCName.txt', newLncName, fmt='%s')
    # np.savetxt('newDisName.txt', newDisName, fmt='%s')


'''
    checkAssoc = []
    checkLnc = []
    checkDis = []
    for i in range(len(newLncName)):
        for j in range(len(newDisName)):
            getAssoc = []
            if assocMat[i][j] == 1:
                getAssoc.append(newLncName[i])
                getAssoc.append(newDisName[j])
                checkAssoc.append(getAssoc)
                checkLnc.append(newLncName[i])
                checkDis.append(newDisName[j])
    #np.savetxt('CheckedAssociation.csv', checkAssoc, fmt='%s')
    #np.savetxt('CheckedLnc.csv', checkLnc, fmt='%s')
    #np.savetxt('CheckedDis.csv', checkDis, fmt='%s')
'''

if __name__ == '__main__':
    getMat()
