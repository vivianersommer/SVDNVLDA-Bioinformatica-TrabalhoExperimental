import numpy as np
import pandas as pd

DFdata = pd.read_csv('SVDMat.csv', header=None)
ArrDate = np.array(DFdata)
repMart = np.where(ArrDate==0, 0.0000000001, 1)

print(ArrDate.shape)

U,s,V = np.linalg.svd(ArrDate)
print(np.shape(U))
print(np.shape(V))
np.savetxt('s.csv', s)


#选取前173维作为特征向量
LncSVD = np.zeros((861,173))
for i in range(861):
    for j in range(173):
        LncSVD[i,j] = U[i, j]

np.savetxt('LncSVD.csv', LncSVD)

DisSVD = np.zeros((432,173))
for i in range(432):
    for j in range(173):
        DisSVD[i,j] = V[j,i]
np.savetxt('DisSVD.csv', DisSVD)




