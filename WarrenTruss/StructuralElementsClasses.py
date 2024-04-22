import math
import numpy as np

# Original code by Bhairav Thakkar, https://www.codeproject.com/Articles/1258509/Modal-Analysis-of-Plane-Truss-using-Python

# give node, return dof
def GetDofx(node):
    return (node-1)*2
def GetDofy(node):
    return (node-1)*2+1


class Node(object):
    """Definition of a 2D node"""
   
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Load(object):
    """Definitions for load on a node"""
    def __init__(self, node, fx, fy):
        self.node = node
        self.fx = fx
        self.fy = fy

class Support(object):
    """Maintains the state of support conditions"""
    def __init__(self, node, uxRestraint, uyRestraint, uxPrescribed, uyPrescribed):
        self.node = node
        self.uxRestraint = uxRestraint
        self.uxPrescribed = uxPrescribed
        self.uyRestraint = uyRestraint
        self.uyPrescribed = uyPrescribed

class Element(object):
    """2D Truss element. Contains various definitions required for the element and also computes
    element matrices"""

    def __init__(self, x1, y1, x2, y2, a, e, n1, n2, rho):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.a = a
        self.e = e
        self.n1 = n1
        self.n2 = n2
        self.l = math.sqrt((x2-x1)**2 + (y2-y1)**2)
        self.rho = rho

    def GetKe(self):
        nDofs = 4;
        c = (self.x2-self.x1)/self.l
        s = (self.y2-self.y1)/self.l
        cc = c * c
        ss = s * s
        cs = c * s
        s = (nDofs,nDofs)       
        ke = np.zeros(s)
        ke[0,0] = cc
        ke[0,1] = cs
        ke[1,1] = ss
        ke[1,0] = cs
        for r in range(2,4):
            for c in range(2,4):
                ke[r,c] = ke[r-2,c-2]

        for r in range(2,4):
            for c in range(0,2):
                ke[r,c] = -ke[r-2,c]

        for r in range(0,2):
            for c in range(2,4):
                ke[r,c] = -ke[r,c-2]
        
        return ke


    def GetMe(self, rho):
        nDofs = 4;

        s = (nDofs,nDofs)       
        me = np.zeros(s)
        me[0,0] = 2
        me[1,1] = 2
        me[2,2] = 2
        me[3,3] = 2

        me[0,2] = 1
        me[1,3] = 1

        me[2,0] = 1
        me[3,1] = 1

        me = me * (1/6 * rho * self.a * self.l)

        return me








