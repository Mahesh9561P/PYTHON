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