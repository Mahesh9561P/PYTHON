import psutil as ps
from sys import *
import time
import os

def Process(Folder):
    if not os.path.exists(Folder):
        os.mkdir(Folder)
    
    filepath=os.path.join(Folder,"Process%s.log"%time.ctime())
    fd=open(filepath,'w')

    data=[]

    for proc in ps.process_iter():
        value=proc.as_dict(attrs=['pid','name','username'])
        data.append(value)
    for each in data:
        fd.write("%s \n"%each)

def main():
    print("------ Marvellous Infosystems ------")
    print("Script title : "+argv[0])
    Process(argv[1])

if __name__ == "__main__":
    main()
