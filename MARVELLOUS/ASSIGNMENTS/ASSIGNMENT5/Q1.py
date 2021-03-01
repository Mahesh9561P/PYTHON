def PRINT(ival):
    if ival!=0:
        print("*",end="\t")
        ival-=1
        PRINT(ival)
    

def main():
    PRINT(5)
    print()

if __name__ == "__main__":
    main()