class Demo():
    iVal=1
    def __init__(self,no1,no2):
        self.i=no1
        self.j=no2

    def fun(self):
        print("fun")
        print("DEMOOBJ1 i ",self.i)
        print("DEMOOBJ1 j ",self.j)

    def gun(self):
        print("gun")
        print("DEMOOBJ2 i ",self.i)
        print("DEMOOBJ2 j ",self.j)

def main():
    Demoobj1=Demo(int(input("ENTER FIRST FOR DEMOOBJ1 ")),int(input("ENTER SECOND FOR DEMOOBJ1 ")))
    Demoobj2=Demo(int(input("ENTER FIRST FOR DEMOOBJ2 ")),int(input("ENTER SECOND FOR DEMOOBJ2 ")))

    Demoobj1.fun()
    Demoobj2.gun()

if __name__ == "__main__":
    main()