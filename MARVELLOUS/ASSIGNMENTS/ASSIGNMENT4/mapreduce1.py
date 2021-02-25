import functools
FILTER=lambda a : a>=70 and a<=90
MAP=lambda b : b+10
d=1
REDUCE= lambda c,d : c*d
def main():
    arr=[]
    num=int(input("Enter the number of elements in list  "))
    for i in range(num):       
        arr.append(int(input("ENTER NUMBER {}  ".format(i+1))))
    
    filtered=list(filter(FILTER,arr))
    print(filtered)

    mapped=list(map(MAP,filtered))
    print(mapped)
    print(functools.reduce(REDUCE,mapped))
    
if __name__ == "__main__":
    main()
