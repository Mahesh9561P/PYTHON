def FACTORIAL(iVal):
    isum=1
    if iVal==0:
        return 1
    while ((iVal-1)!=0):
        isum=iVal*isum
        iVal=iVal-1
    return isum

def main():
    print(FACTORIAL(int(input("ENTER THE NUMBER"))))

if __name__ == "__main__":
    main()