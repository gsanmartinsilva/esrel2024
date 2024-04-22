# Program for analysis of a 2D plane truss
# Original code by Bhairav Thakkar, https://www.codeproject.com/Articles/1258509/Modal-Analysis-of-Plane-Truss-using-Python


from StructuralElementsClasses import Node,  Element, Support, Load, GetDofx, GetDofy
import Input, Assembly, Plot
import math
import numpy as np
import scipy
import sys
import pickle



# Read the input
structure_name = "WarrenTruss"
inp = Input.Input()
inp.Read(structure_name, sys.argv[2])
nnodes= inp.nnodes
nodes = inp.nodes
nele = inp.nele
elements = inp.elements
nbcnodes = inp.nbcnodes
supports = inp.supports
loads = inp.loads
filename = inp.filename

# Assemble the stiffness, mass matrices and load vector
ndof = nnodes * 2
assembly = Assembly.Assembly()
assembly. GenerateKgAndMg(ndof,elements)
# generate load vector
assembly.GenerateLoadVector(loads)
# Retrieve global stiffness matrix, mass matrix and load vector from the assembly object
kg = assembly.kg
mg = assembly.mg
r = assembly.r
FreeDOFList, RestrainedDOFList = assembly.GetFreeDOFSupportedDOF(ndof, supports)
kff, ksf, kfs, kss, mff, msf, mfs, mss, rf, rs = assembly.GenerateMatrices(ndof, supports)

# Solve the static deformation of the truss
print("Performing linear solution... ", end="")
deltaF = np.linalg.solve(kff,rf)
Fs = ksf @ deltaF
print ("done.")
print(f"Displacement on Free DOF: \n{deltaF}")
print(f"Reaction on the supports: \n{Fs}")

# Compute eigenvalues and eigen vectors and sort them
matrix = np.dot(np.linalg.inv(mff), kff)
w2, vr = np.linalg.eig(matrix)
sortedIndex = np.argsort(w2)
w2 = w2[sortedIndex]
vr = vr[:, sortedIndex]

# Mass normalize everything
if sys.argv[1] == 1:
    for i in range(np.shape(kff)[0]):
        vr[:, i] = vr[:, i] / np.sqrt(np.diag(vr.T @ mff @ vr)[i])
    # Check orthogonality conditions
    phiTMphi = vr.T @ mff @ vr
    phiTKphi = vr.T @ kff @ vr
    assert np.mean(abs(phiTMphi - np.eye(phiTMphi.shape[0]))) < 1e-8, "Orthogonality with mass is not working"
    assert np.mean(abs(phiTKphi - np.diag(w2))), "Orthogonality with stiffness is not working"

# Print the natural frequencies
for k, eig_val in enumerate(w2):
    print(f"Natural Frequency of Mode {k}: {np.sqrt(eig_val)}")



data = {"PHI": vr, "K": kff,
        "structure_name": structure_name,
        "mass_normalized": sys.argv[1],
        "FreeDOFList": FreeDOFList,
        "RestrainedDOFList": RestrainedDOFList,
        "noise": sys.argv[2]}
mn = "_MassNormalized" if int(sys.argv[1]) else ""
with open(f"data/{structure_name}{mn}.pickle", 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
# Draw the truss
print(FreeDOFList)
print(RestrainedDOFList)
Plot.plot_undeformed(nodes, elements, FreeDOFList)

