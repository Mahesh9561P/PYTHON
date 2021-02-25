def SEARCH(brr,Num):
    Cnt=0
    for i in range(len(brr)):
        if brr[i]==Num:
            Cnt+=1
    return Cnt

def main():
    arr=[]
    for i in range(int(input("ENTER THE NUMBER OF ELEMENTS IN ARRAY"))):
        print("ENTER THE ARRAY ELEMENT ",i+1)
        arr.append(int(input()))

    print("ANSWER IS : ",SEARCH(arr,int(input("ENTER THE ELEMENT TO BE SEARCHED"))))
    


if __name__== "__main__":
    main()