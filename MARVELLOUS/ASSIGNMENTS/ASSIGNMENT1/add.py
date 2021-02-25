def add(iVal1,iVal2):
    return iVal1+iVal2

def main():
    no1=int(input("ENTER THE FIRST NUMBER "))
    no2=int(input("ENTER THE SECOND NUMBER "))
    ret=add(no1,no2)
    print("ADDITION IS ",ret)

if __name__ == "__main__":
    main()