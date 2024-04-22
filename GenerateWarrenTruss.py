# Importing Libraries
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pickle
from WarrenTruss import Input, Assembly, Plot





if __name__=="__main__":

    parser = argparse.ArgumentParser(description="Quantum Computing Optimization - Optimal Sensor Placement")
    #Adding optional parameters
    
    parser.add_argument('-structure',
                        help="name of input file.",
                        type=str,
                        default="WarrenTruss_11")
    
    parser.add_argument('-massnormalize',
                        help="If the matrix should be mass normalized.",
                        action='store_false')
    
    
    #Parsing the argument
    args=parser.parse_args()
    print(args)
 
    # Read the input
    inp = Input.Input()
    inp.Read(args.structure)
    nnodes = inp.nnodes
    nodes = inp.nodes
    nele = inp.nele
    elements = inp.elements
    nbcnodes = inp.nbcnodes
    supports = inp.supports
    loads = inp.loads
    
    # Assemble the stiffness, mass matrices and load vector
    ndof = nnodes * 2
    assembly = Assembly.Assembly()
    assembly. GenerateKgAndMg(ndof,elements)
    
    # Generate load vector
    assembly.GenerateLoadVector(loads)
    
    # Retrieve global stiffness matrix, mass matrix and load vector from the assembly object
    kg = assembly.kg
    mg = assembly.mg
    r = assembly.r
    FreeDOFList, RestrainedDOFList = assembly.GetFreeDOFSupportedDOF(ndof, supports)
    kff, ksf, kfs, kss, mff, msf, mfs, mss, rf, rs = assembly.GenerateMatrices(ndof, supports)

    # Solve the static deformation of the truss
    # print("Performing linear solution... ", end="")
    # deltaF = np.linalg.solve(kff,rf)
    # Fs = ksf @ deltaF
    # print ("done.")
    # print(f"Displacement on Free DOF: \n{deltaF}")
    # print(f"Reaction on the supports: \n{Fs}")
    
    # Compute eigenvalues and eigen vectors and sort them
    matrix = np.dot(np.linalg.inv(mff), kff)
    w2, vr = np.linalg.eig(matrix)
    sortedIndex = np.argsort(w2)
    w2 = w2[sortedIndex]
    vr = vr[:, sortedIndex]
    
    # Mass normalize everything
    if args.massnormalize:
        for i in range(np.shape(kff)[0]):
            vr[:, i] = vr[:, i] / np.sqrt(np.diag(vr.T @ mff @ vr)[i])
        # Check orthogonality conditions
        phiTMphi = vr.T @ mff @ vr
        phiTKphi = vr.T @ kff @ vr
        assert np.mean(abs(phiTMphi - np.eye(phiTMphi.shape[0]))) < 1e-8, "Orthogonality with mass is not working"
        assert np.mean(abs(phiTKphi - np.diag(w2))), "Orthogonality with stiffness is not working"

    # Print the natural frequencies
    for k, eig_val in enumerate(w2):
        print(f"Natural Frequency of Mode {k}: {np.sqrt(eig_val)/(2*np.pi)}")

    data = {"Phi": vr, "K": kff, "M": mff,
            "omega_n": np.sqrt(eig_val),
            "FreeDOFList": FreeDOFList,
            "RestrainedDOFList": RestrainedDOFList}
    mn = "_NotMassNormalized" if not args.massnormalize else ""
    with open(f"data/{args.structure}{mn}.pickle", 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    # Plot.plot_undeformed(nodes, elements, FreeDOFList)
  
        