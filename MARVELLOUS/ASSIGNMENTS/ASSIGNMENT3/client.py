import MarvellousNum as MN
import functools as FT
def main():
    LIST=[]
    LIST2=[]
    for i in range(int(input("ENTER THE NUMBER OF ELEMENTS IN ARRAY"))):
        print("ENTER THE ARRAY ELEMENT ",i+1)
        no=int(input())
        LIST.append(no)
    print(LIST)

    LIST2=MN.PRIME(LIST)
    print(LIST2)
    print(FT.reduce(lambda a,b : a+b,LIST2))

if __name__ == "__main__":
    main()