def PATTERN(iVal):
    iRow=iCOl=iVal
    for i in range(iRow):
        for j in range(len(range(iCOl))):
            if i >= j:
                print("{}".format(j+1),end=" ")
        print("\r")

def main():
    PATTERN(int(input("ENTER NUMBER OF ROWS AND COLS")))

if __name__ == "__main__":
    main()  