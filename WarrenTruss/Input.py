from WarrenTruss.StructuralElementsClasses import Node, Element, Support, Load
import numpy as np
# Code by Bhairav Thakkar, https://www.codeproject.com/Articles/1258509/Modal-Analysis-of-Plane-Truss-using-Python


class Input(object):
    """Maintains the input data"""
    def __init__(self, *args, **kwargs):
        return super().__init__(*args, **kwargs)
    
    def Read(self, structure):
        # Ask for the input file name
        self.infilename = f"data/{structure}.in"

        self.f = open(self.infilename)
        self.ReadHeader()
        self.ReadNodes()
        self.ReadElements()
        self.ReadBC()
        self.ReadLoads()
        self.f.close()
        

    def ReadHeader(self):
        # parse each line in the file and extract the trus model

        line = self.f.readline() # first line
        numbers = line.split(",") # numbers is not a list of all numbers available in the line
        self.nnodes = int(numbers[0])
        self.nele = int(numbers[1])
        self.nbcnodes = int(numbers[2])
        self.nloadnodes = int(numbers[3])

    def ReadNodes(self):
        # read the nodes
        self.nodes = []
        for node in range(0, self.nnodes):
            line = self.f.readline() # read next line
            numbers = line.split(",") # numbers is not a list of all numbers available in the line

            x = float(numbers[0])
            y = float(numbers[1])
            n = Node(x,y)
            self.nodes.append(n)

    def ReadElements(self):
        # read the elements
        self.elements = []
        for ele in range(0, self.nele):
            line = self.f.readline()
            numbers = line.split(",")
            n1 = int(numbers[0])-1
            n2 = int(numbers[1])-1
            a = float(numbers[2])
            e = float(numbers[3])
            rho = float(numbers[4])
            x1= self.nodes[n1].x
            y1 = self.nodes[n1].y
            x2 = self.nodes[n2].x
            y2 = self.nodes[n2].y
            
            elem = Element(x1,y1,x2,y2,a,e, n1+1, n2 + 1, rho)
            self.elements.append(elem)    

    def ReadBC(self):
        # read bc
        self.supports = []
        for bc in range(0, self.nbcnodes):
            line = self.f.readline()
            numbers = line.split(",")
            bcnode = int(numbers[0])
            uxr = int(numbers[1])
            uyr = int(numbers[2])
            uxp = float(numbers[3])
            uyp = float(numbers[4])    
            bc = Support(bcnode,uxr,uyr,uxp,uyp)
            self.supports.append(bc)

    def ReadLoads(self):
        # read loads
        self.loads = []
        for p in range(0, self.nloadnodes):
            line = self.f.readline()
            numbers = line.split(",")
            loadednode = int(numbers[0])
            fx = float(numbers[1])
            fy = float(numbers[2])
    
            p = Load(loadednode,fx,fy)
            self.loads.append(p)
