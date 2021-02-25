def LISTMIN(brr):
    i=0
    min=brr[i]
    for i in range(len(brr)):
        if min>brr[i]:
            min=brr[i]
    return min

def main():
    arr=[]
    val=int(input("ENTER THE NUMBER OF ELEMENTS IN ARRAY"))
    for i in range(val):
        print("ENTER THE ARRAY ELEMENT ",i+1)
        arr.append(int(input()))
    print("MINIMUM IS : ",LISTMIN(arr))
    


if __name__== "__main__":
    main()