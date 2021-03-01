iMult=1
def FACT(iVal):
    global iMult
    if iVal!=0 and ((iVal-1)!=0):
        iMult=iMult*iVal
        iVal-=1
        FACT(iVal)
    return iMult


def main():
     print(FACT(int(input("ENTER THE NUMBER : "))))


if __name__ == "__main__":
    main()