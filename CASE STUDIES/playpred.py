import pandas as pd
import numpy as np

def Play_Pred(path):
    data=pd.read_csv(path)
    print("Dataset is ready",len(data))


def main():
    print("--------PLAY PREDICTOR----------")
    path=input("ENTER THE CSV FILE NAME: ")
    Play_Pred(path)
    Play_Pred()

if __name__ == "__main__":
    main()