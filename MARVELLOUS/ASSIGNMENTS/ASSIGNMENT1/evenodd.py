def CHKEVEN(iVal):
    if(iVal%2==0):
        return True
    else:
        return False


def main():
    no=int(input("ENTER THE NUMBER "))
    ret=CHKEVEN(no)
    if ret == True:
        print("EVEN")
    else:
        print("ODD")

if __name__ == "__main__":
    main()