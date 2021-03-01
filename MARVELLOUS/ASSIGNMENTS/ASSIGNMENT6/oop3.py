class Arithmatic:
    def __init__(self):
        self.value1=0
        self.value2=0

    def Accept(self):
        self.value1=int(input("ENTER FIRST NUMBER : "))
        self.value2=int(input("ENTER SECOND NUMBER : "))

    def Addition(self):
        return self.value1+self.value2
    
    def Substraction(self):
        return self.value1-self.value2
    
    def Division(self):
        return self.value1/self.value2
    
    def Multiply(self):
        return self.value1*self.value2

def main():
    
    obj1=Arithmatic()
    obj1.Accept()
    print("ADDITION IS : ",obj1.Addition())
    print("MULTIPLICATION IS : ",obj1.Multiply())
    print("DIVISION IS : ",obj1.Division())
    print("SUBSTRACTION IS : ",obj1.Substraction())

if __name__ == "__main__":
    main()