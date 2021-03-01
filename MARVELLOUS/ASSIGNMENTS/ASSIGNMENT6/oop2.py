class Circle:
    pi=3.14
    
    def __init__(self):
        Radius=0.0
        Area=0.0
        Circumference=0.0
    
    def Accept(self):
        self.Radius=int(input("Enter The Radius "))
    
    def CalculateArea(self):
        self.Area=self.pi*self.Radius*self.Radius

    def CalculateCircumference(self):
        self.Circumference=2*self.pi*self.Radius

    def Display(self):
        print("RADIUS OF CIRCLE IS : ",self.Radius)
        print("AREA OF CIRCLE : ",self.Area)
        print("CIRCUMFERENCE OF CIRCLE : ",self.Circumference)  

def main():
    circleobj=Circle()
    circleobj.Accept()
    circleobj.CalculateArea()
    circleobj.CalculateCircumference()
    circleobj.Display()

    circleobj2=Circle()
    circleobj2.Accept()
    circleobj2.CalculateArea()
    circleobj2.CalculateCircumference()
    circleobj2.Display()
    
if __name__ == "__main__":
    main()