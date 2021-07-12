import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def Mean(arr):
    size=len(arr)
    sum=0
    for i in range(size):
        sum=sum+arr[i]  
    return(sum/size)

def HeadBrain(name):
    dataset=pd.read_csv(name)
    print("SIZE : ",dataset.shape)

    X=dataset["Head Size(cm^3)"].values
    Y=dataset["Brain Weight(grams)"].values
    #y=mx+c
    #c=Y-mx

    Mean_X=Mean(X)
    Mean_Y=Mean(Y)
    #m=(sum(x-xb)*y-yb))/sum(x-xb)^2
    numerator=0
    denomenator=0

    for i in range(len(X)):
        numerator=numerator+(X[i]-Mean_X)*(Y[i]-Mean_Y)
        denomenator=denomenator+(X[i]-Mean_X)**2

    m=numerator/denomenator
    print("m",m)
    
    c=Mean_Y-(m*Mean_X)
    print(c)
    
    X_Start=np.min(X)-200
    X_End=np.max(X)+200

    x=np.linspace(X_Start,X_End)
    y=m*x+c
    print("y : " ,y)
    print("x : " ,x)

    plt.plot(x,y,color='c',label='data plot')
    plt.scatter(X,Y,color='r',label='points')

    plt.show()
    

def main():
    HeadBrain("MarvellousHeadBrain.csv")

if __name__=="__main__":
    main()