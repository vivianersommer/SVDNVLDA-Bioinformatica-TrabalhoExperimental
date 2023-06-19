### This is a brief Python implementation of lncRNA-disease association prediction model SVDNVLDA. 



##### The whole object was implected on Python 3.6, and versions of some import packages are:

| packageName  | version |
| :----------: | ------- |
|   networkx   | 2.5.1   |
|    numpy     | 1.19.5  |
|    pandas    | 0.20.3  |
| scikit-learn | 0.24.2  |
|    scipy     | 1.5.4   |
|   xgboost    | 1.3.3   |

and *node2vec* package was downloaded from [https://github.com/eliorc/node2vec](https://github.com/eliorc/node2vec)



##### There are five directories in this project：

​	***Data* Folder : Contains all data used in our experiment.**

​		eg:  ***LncDis.csv*** :Contains all lncRNA-disease association records; 			

​			***AllAsso.csv***:Contains all association records used in our experiment

​	***SVD* Folder: The implementation of SVD on lncRNA-disease association matrix**

​			***GetMat.py***: transform the association records into association matrix

​			***SVDImp.py***: implete SVD on the matrix and save the linear features of lncRNAs and diseases  

​	***N2V* Folder: The implementation of node2vec on LMDN**

​			***node2vec_flight.py***: implete node2vec on LMDN

​			***vecGet.py***: fit the original node2vec result to classifiers

​	***Sampling* Folder: Create the negative samles and conduct the randomly sample**

​	***Classifier* Folder:**	**implete the ultimate classification exhibit the 10-fold cross validation results**	

### Ideia
Rodar o arquivo classifier.py, mudando os parâmetros do XGBoost, com pelo menos 3 datasets.

Montar os gráficos e fazer uma comparação dos resultados.

