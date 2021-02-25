def LISTMAX(brr):
    max=0
    for i in range(len(brr)):
        if max<brr[i]:
            max=brr[i]
    return max

def main():
    arr=[]
    val=int(input("ENTER THE NUMBER OF ELEMENTS IN ARRAY"))
    for i in range(val):
        print("ENTER THE ARRAY ELEMENT ",i+1)
        arr.append(int(input()))
    print("MAXIMUM IS : ",LISTMAX(arr))
    


if __name__== "__main__":
    main()