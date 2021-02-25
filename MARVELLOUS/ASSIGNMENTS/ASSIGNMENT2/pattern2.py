def PATTERN2(iVal):
    iRow=iCol=iVal
    for i in range(0,iRow):
        for j in range(len(range(iCol))):
            if i<=j:
                print("*",end=" ")
        print("\r")
    

def main():
    PATTERN2(int(input("ENTER NUMBER OF ROWS AND COLS")))

if __name__ == "__main__":
    main()