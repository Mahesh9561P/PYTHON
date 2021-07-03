import pandas as pd
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

def Wine_Predictor():
    data=pd.read_csv("/home/mahesh/Mahesh/Python/MarvellousPython/CASESTUDY/WinePredictor.csv")
    X=data.iloc[:,1:].values
    Y=data.iloc[:,0].values
    data_train,data_test,target_train,target_test=train_test_split(X,Y,test_size=0.2)
     
    scaler = StandardScaler()
    scaler.fit(data_train)
    data_train = scaler.transform(data_train)
    data_test = scaler.transform(data_test)

    classifier=KNeighborsClassifier(n_neighbors=3)
    classifier.fit(data_train,target_train)
    output=classifier.predict(data_test)
    accuracy=accuracy_score(target_test,output)
    print(accuracy*100)

def main():
    Wine_Predictor()

if __name__=="__main__":
    main()