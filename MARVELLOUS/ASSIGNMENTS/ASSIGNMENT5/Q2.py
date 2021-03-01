i=1
def PRINT(ival):
    global i
    if i<=ival:
        print(i,end="\t")
        i+=1
        PRINT(ival)

def main():
    PRINT(5)
    print()
   
if __name__ == "__main__":
    main()