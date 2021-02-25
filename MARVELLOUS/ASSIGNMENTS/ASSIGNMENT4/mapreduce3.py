import functools
GREAT=lambda a,b : a if (a>b) else b

MULT = lambda a : a*2
def PRIME(brr):
    crr=[]
    for i in range(len(brr)):
        num=int(brr[i]/2)
        cnt=0
        for j in range(2,num):
            if brr[i]%j == 0:
                cnt+=1
                break
        if cnt==0:
            crr.append(brr[i])

    return crr

def main():
    numbers = []
    for i in range(int(input("ENTER THE NUMBER : "))):
        numbers.append(int(input("ENTER THE {} NUMBER : ".format(i+1))))

    FILTER=PRIME(numbers)
    print(FILTER)
    MAPPED=list(map(MULT,FILTER))
    print(MAPPED)
    REDUCED=functools.reduce(GREAT,MAPPED)
    print(REDUCED)

if __name__ == "__main__":
    main()