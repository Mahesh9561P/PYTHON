def LISTSUM(brr):
    sum=0
    for i in range(len(brr)):
        sum=sum+brr[i]
    return sum

def main():
    arr=[]
    val=int(input("ENTER THE NUMBER OF ELEMENTS IN ARRAY"))
    for i in range(val):
        print("ENTER THE ARRAY ELEMENT ",i+1)
        arr.append(int(input()))
    print("ADDITION IS : ",LISTSUM(arr))
    


if __name__== "__main__":
    main()