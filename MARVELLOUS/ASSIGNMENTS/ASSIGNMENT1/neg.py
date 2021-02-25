def CHKNEG(iVal):
    if iVal>0:
        return True
    elif iVal<0:
        return False
    else:
        return None


def main():
    no=int(input("ENTER THE NUMBER "))
    ret=CHKNEG(no)
    if ret == True:
        print("POSITIVE")
    elif ret == False:
        print("NEGATIVE")
    else:
        print("ZERO")

if __name__ == "__main__":
    main()