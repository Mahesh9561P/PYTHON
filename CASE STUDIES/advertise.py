import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

def Advertise():
    dataset=pd.read_csv("Advertising.csv")
    data=dataset.iloc[:,1:4].values
    labels=dataset.iloc[:,4].values

    data_train,data_test,label_train,label_test=train_test_split(data,labels,test_size=0.2)

    Regression=LinearRegression()
    Regression.fit(data_train,label_train)
    out=Regression.predict(data_test)
    print("TRAIN SCORE : ",Regression.score(data_train,label_train)*100)
    print("TEST SCORE : ",Regression.score(data_test,label_test)*100)

def main():
    Advertise()

if __name__ == "__main__":
    main()