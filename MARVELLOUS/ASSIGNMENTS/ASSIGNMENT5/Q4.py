iSum=0
def SUMATION(iVal):
    global iSum
    if iVal!=0:
        iDigit=int(iVal%10)
        iSum=iDigit+iSum
        iVal=int(iVal/10)
        SUMATION(iVal)
    return iSum


def main():
    print(SUMATION(int(input("ENTER THE NUMBER : "))))

if __name__ == "__main__":
    main()