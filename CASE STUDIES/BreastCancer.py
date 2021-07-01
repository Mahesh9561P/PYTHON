import pandas as pd
import sys
import array
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def f_importances(coef, names):
    imp = coef
    imp,names = zip(*sorted(zip(imp,names)))
    plt.barh(range(len(names)), imp, align='center')
    plt.yticks(range(len(names)), names)
    plt.show()

def SVM(x_train,x_test,y_train,y_test):
    model=SVC(kernel='poly')
    model.fit(x_train,y_train)

    print("SCORE ON TRAINDATA is",model.score(x_train,y_train))
    print("SCORE ON TRAINDATA is",model.score(x_test,y_test))

def RandomForest(x_train,x_test,y_train,y_test):
    model=RandomForestClassifier()
    model.fit(x_train,y_train)
    print("SCORE ON TRAINDATA is",model.score(x_train,y_train))
    print("SCORE ON TRAINDATA is",model.score(x_test,y_test))

def Check_Null(dataset):
    print(dataset.isnull().sum())

def Handle_Missing(dataset,miss_col,miss_val):
    return dataset[dataset[miss_col]!=miss_val]

def main():
    dataset=pd.read_csv('/home/mahesh/Downloads/breast-cancer-wisconsin.csv')
    #Check_Null(dataset)

    dataset=Handle_Missing(dataset,dataset.columns[6],'?')

    x_train,x_test,y_train,y_test=train_test_split(dataset.iloc[:,:-1],dataset.iloc[:,-1])
    features_names = [dataset.columns]

    SVM(x_train,x_test,y_train,y_test)
    #x_train,x_test,y_train,y_test=pd.DataFrame(x_train),pd.DataFrame(x_test),pd.DataFrame(y_train),pd.DataFrame(y_test)
    RandomForest(x_train,x_test,y_train,y_test)

    '''
    elif sys.argv[1]==0:
    RandomForest()'''

if __name__=="__main__":
    main()
