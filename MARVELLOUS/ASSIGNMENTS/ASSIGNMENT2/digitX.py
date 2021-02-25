def DIGITX(iVal):
    iSum=0
    while iVal!=0:
        iNo=iVal%10
        iSum=iSum+iNo
        iVal=iVal//10
    return iSum

def main():
    print(DIGITX(int(input("NUMBER : "))))

if __name__ == "__main__":
    main()  