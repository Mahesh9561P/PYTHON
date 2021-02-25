import arithmatic as ar

def main():
    iVal1=int(input("ENTER FIRST NUMBER "))
    iVal2=int(input("ENTER SECOND NUMBER "))
    print(ar.ADD(iVal1,iVal2))
    print(ar.SUB(iVal1,iVal2))
    print(ar.DIV(iVal1,iVal2))
    print(ar.MULT(iVal1,iVal2))

if __name__ == "__main__":
    main()