import functools
CHKEVEN = lambda a : a%2==0

SQUARE = lambda b : b**2

SUM1 = lambda x,y : x+y

def main():
    arr=[]
    num=int(input("Enter the number of elements in list  "))
    for i in range(num):       
        arr.append(int(input("ENTER NUMBER {}  ".format(i+1))))

    newdata=list(filter(CHKEVEN,arr))
    print(newdata)
    mapped=list(map(SQUARE,newdata))
    print(mapped)
    print(functools.reduce(SUM1,mapped))
    

if __name__ == "__main__":
    main()
