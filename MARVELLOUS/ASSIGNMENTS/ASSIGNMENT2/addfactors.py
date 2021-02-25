def ADDFACTORIAL(iNo):
    isum=0
    for i in range(1,int(iNo/2+1)):
        if iNo%i==0:
            isum=isum+i
    return isum

def main():
    print(ADDFACTORIAL(int(input("ENTER THE NUMBER :"))))

if __name__ == "__main__":
    main()