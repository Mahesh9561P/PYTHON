def DIVBY5(iVal):
    if(iVal<0):
        iVal=-iVal
    if iVal%5 == 0:
        return True
    else :
        return False


def main():
    no=int(input("ENTER THE NUMBER "))
    ret=DIVBY5(no)
    if ret == True:
        print("YESS")
    else:
        print("NO")

if __name__ == "__main__":
    main()