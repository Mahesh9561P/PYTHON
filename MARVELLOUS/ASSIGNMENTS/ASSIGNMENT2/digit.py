def DIGIT(iVal):
    iCnt=0
    while iVal!=0:
        iNo=iVal%10
        iCnt=iCnt+1
        iVal=iVal//10
    return iCnt

def main():
    print(DIGIT(int(input("NUMBER : "))))

if __name__ == "__main__":
    main()  