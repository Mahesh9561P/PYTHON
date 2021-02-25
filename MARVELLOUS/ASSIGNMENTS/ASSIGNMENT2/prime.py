def PRIME(iNo):
    
    for i in range(2,int(iNo/2+1)):
        if iNo%i==0:
            print("IT IS NOT A PRIME NUMBER")
            break
    else:
        print("IT IS PRIME NUMBER")        

def main():PRIME(int(input("ENTER THE NUMBER")))

if __name__ == "__main__":
    main()