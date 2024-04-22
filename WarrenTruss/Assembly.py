from WarrenTruss.StructuralElementsClasses import Element, Load, GetDofx, GetDofy, Support
import numpy as np

class Assembly(object):
    """description of class"""

    def __init__(self, *args, **kwargs):
        return super().__init__(*args, **kwargs)

    # generate the global stiffness and global mass matrices
    def GenerateKgAndMg(self, ndof, elements):
        s = (ndof,ndof)
        self.ndof = ndof
        self.kg = np.zeros(s)
        self.mg = np.zeros(s)

        # compute elemental stiffness and mass matrices
        for ele in elements:
            ke = ele.GetKe() * (ele.a * ele.e / ele.l)
            me = ele.GetMe(ele.rho)
            n1 = ele.n1
            n2 = ele.n2
            dofs = [GetDofx(n1), GetDofy(n1), GetDofx(n2), GetDofy(n2)]
            for i in range(4):
                for j in range(4):
                    self.kg[dofs[i], dofs[j]] += ke[i,j]
                    self.mg[dofs[i], dofs[j]] += me[i,j]


    def GenerateLoadVector(self, loads):
        self.r = [0 for x in range(self.ndof)]
        
        for load in loads:
            n = load.node
            dofx = GetDofx(n)
            self.r[dofx] += load.fx
            dofy = GetDofy(n)
            self.r[dofy] += load.fy
    
    def GetFreeDOFSupportedDOF(self, ndof, supports):
        RestrainedDOFList = []
        for support in supports:
            n = support.node
            dofx = GetDofx(n)
            dofy = GetDofy(n)
            
            if support.uxRestraint == 1:
                RestrainedDOFList.append(dofx)
            if support.uyRestraint == 1:
                RestrainedDOFList.append(dofy)
                
        FreeDOFList = [i for i in range(ndof) if i not in RestrainedDOFList]
            
        return sorted(FreeDOFList), sorted(RestrainedDOFList)
   
        
    def GenerateMatrices(self, ndof, supports):
        # Generate Kff, Kss, Mff, Mss,...
        FreeDOFList, RestrainedDOFList = self.GetFreeDOFSupportedDOF(ndof, supports)
        kff = self.kg[np.ix_(FreeDOFList, FreeDOFList)]
        ksf = self.kg[np.ix_(RestrainedDOFList, FreeDOFList)]
        kfs = self.kg[np.ix_(FreeDOFList, RestrainedDOFList)]
        kss = self.kg[np.ix_(RestrainedDOFList, RestrainedDOFList)]
        
        mff = self.mg[np.ix_(FreeDOFList, FreeDOFList)]
        msf = self.mg[np.ix_(RestrainedDOFList, FreeDOFList)]
        mfs = self.mg[np.ix_(FreeDOFList, RestrainedDOFList)]
        mss = self.mg[np.ix_(RestrainedDOFList, RestrainedDOFList)]
        
        rf = [self.r[i] for i in FreeDOFList]
        rs = [self.r[i] for i in RestrainedDOFList]
        
        return kff, ksf, kfs, kss, mff, msf, mfs, mss, rf, rs
       

