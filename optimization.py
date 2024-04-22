#!/usr/bin/env python
# coding: utf-8



from pennylane import numpy as np
import argparse
import pandas as pd
import pickle
import pennylane as qml
from pennylane import qaoa
from tqdm import tqdm
from utils_dicke_states import dicke_state
from utils import (GenerateTermsMatrix, GenerateConstrainTerms, solve_qubo, app_ratio_fn, generate_cost_hamiltonian, GenerateGraph, compute_score)
from scipy.optimize import minimize


np.random.seed(412)

parser = argparse.ArgumentParser()
#Adding optional parameters
parser.add_argument('-STRUCTURE',
                    type=str,
                    default = "WarrenTruss_5")

parser.add_argument('-MODES',
                    type=str,
                    default="0,1,4")

parser.add_argument('-MIXER',
                    type=str,
                    default = "trad")

parser.add_argument('-N_S',
                    type=int,
                    default=2)

parser.add_argument('-ALPHA',
                    type=int,
                    default=1)

parser.add_argument('-CIRCUIT_DEPTHS',
                    type=str,
                    default="1,4,16,64")

parser.add_argument('-OPTIMIZER',
                    type=str,
                    default="SGD")

parser.add_argument('-CONST',
                    type=int,
                    default=0)

#Parsing the argument
args=parser.parse_args()
print(args)
STRUCTURE = args.STRUCTURE
MODES = [int(item) for item in args.MODES.split(',')]
print(f"Modes: {MODES}")
MIXER = args.MIXER
ALPHA = args.ALPHA
N_S = args.N_S
CIRCUIT_DEPTHS = [int(item) for item in args.CIRCUIT_DEPTHS.split(',')]
OPTIMIZER = args.OPTIMIZER
CONST = args.CONST


# Read data
with open(f'data/{STRUCTURE}.pickle', 'rb') as handle:
        data = pickle.load(handle)

stiffness_matrix = data["K"]
mass_matrix = data["M"]
modal_matrix = data["Phi"][:, MODES] # Modes to reach 95% MMP
N = len(stiffness_matrix)

# Generate the MSE terms
A = GenerateTermsMatrix(modal_matrix, stiffness_matrix)
normalization_constant = (np.ones((N,1)).T @ A @ np.ones((N,1)))[0][0] # Heuristic: normalize by max energy (all dof with sensors), which is always easy to compute.
print(f"Normalization constant: {normalization_constant}")
A /= normalization_constant

# Generate constraint terms
constrain_terms, constrain_offset = GenerateConstrainTerms(N, N_S) 

# solve the QUBO
df = solve_qubo(A - ALPHA * constrain_terms)
df["f_obj + constraint offset"] = df.f_obj - ALPHA * constrain_offset
print(df.head(50))
print(df.tail(50))

# Generate Hamiltonians, -1 indicate that we are maximizing
H_c, H_c_matrix = generate_cost_hamiltonian(-1 * (A - ALPHA * constrain_terms), N)
H_m = qaoa.x_mixer(range(N))
H_m_xy = qaoa.xy_mixer(GenerateGraph(N))

# Generate QAOA circuit
def qaoa_layer_trad(gamma, beta):
    qaoa.cost_layer(gamma, H_c)
    qaoa.mixer_layer(beta, H_m)
    
def qaoa_layer_xy(gamma, beta):
    qaoa.cost_layer(gamma, H_c)
    qaoa.mixer_layer(beta, H_m_xy)

# Function for trad
@qml.qnode(qml.device("lightning.gpu", wires=len(range(N))), diff_method="adjoint")
def qaoa_fn_trad(params, **kwargs):
    for w in range(N):
        qml.Hadamard(wires=w)
    qml.layer(qaoa_layer_trad, kwargs["P"], params[0], params[1])
    return qml.expval(H_c) 

@qml.qnode(qml.device("lightning.gpu", wires=len(range(N)), shots = 1024))
def qaoa_trad_counts(params, **kwargs):
    for w in range(N):
        qml.Hadamard(wires=w)
    qml.layer(qaoa_layer_trad, kwargs["P"], params[0], params[1])
    return qml.probs()

# function for XY
dicke_circuit = qml.from_qiskit(dicke_state(N, N_S))
@qml.qnode(qml.device("lightning.gpu", wires=len(range(N))), diff_method="adjoint")
def qaoa_fn_xy(params, **kwargs):
    dicke_circuit(wires=list(range(N))[::-1])
    gamma = params[0] * np.asarray([(2*m-1)/(2*kwargs["P"]) for m in range(1,kwargs["P"]+1)])
    beta = params[1] * np.asarray([1 - (2*m-1)/(2*kwargs["P"]) for m in range(1,kwargs["P"]+1)])
    qml.layer(qaoa_layer_xy, kwargs["P"], gamma, beta)
    return qml.expval(H_c)

@qml.qnode(qml.device("lightning.gpu", wires=len(range(N)), shots = 1024))
def qaoa_xy_counts(params, **kwargs):
    dicke_circuit(wires=list(range(N))[::-1])
    gamma = params[0] * np.asarray([(2*m-1)/(2*kwargs["P"]) for m in range(1,kwargs["P"]+1)])
    beta = params[1] * np.asarray([1 - (2*m-1)/(2*kwargs["P"]) for m in range(1,kwargs["P"]+1)])
    qml.layer(qaoa_layer_xy, kwargs["P"], gamma, beta)
    return qml.probs()


def COBYLA_fn(x, P):
    p = len(x) // 2
    gamma = x[:p]
    beta = x[p:]
    return qaoa_fn([gamma, beta], P=P)

def COBYLA_counts(x, P):
    p = len(x) // 2
    gamma = x[:p]
    beta = x[p:]
    return qaoa_counts([gamma, beta], P=P)


def SGD_fn(fn, starting_point, P):
    theta = np.array(starting_point, requires_grad=True) # Initial guess parameters
    angle = [theta] # Store the values of the circuit parameter
    cost = [fn(theta, P=P)] # Store the values of the cost function

    opt = qml.AdamOptimizer(stepsize=0.005) # Our optimizer!
    max_iterations = 500 # Maximum number of calls to the optimizer 
    conv_tol = 1e-4 # Convergence threshold to stop our optimization procedure
    for n in tqdm(range(max_iterations)):
        if n % 50 == 0:
            print(f"Step: {n:4}\tCost function: {cost[-1]:10.8f}\t Approximation Ratio: {cost[-1]:4.3f}")
        theta, prev_cost = opt.step_and_cost(fn, theta, P=P)
        cost.append(fn(theta, P=P))
        angle.append(theta)

        # convergence criteria: if the last 3 iterations do not improve the solution more than 
        if n > 5:
            conv = np.abs(cost[-1] - prev_cost) <= conv_tol and np.abs(cost[-2] - cost[-1]) <= conv_tol and np.abs(cost[-3] - cost[-2]) <= conv_tol 
            if conv:
                break
    
    print(f"\nFinal Cost: {cost[-1]:7.4f}")
    
    return angle, cost


# Define function
if MIXER == "xy":
    qaoa_fn = qaoa_fn_xy
    qaoa_counts = qaoa_xy_counts
elif MIXER == "trad":
    qaoa_fn = qaoa_fn_trad
    qaoa_counts = qaoa_trad_counts
else:
    raise("MIXER IS NOT DEFINED!")

# # trad mixer, WT11
# starting_points_dict = {1: [6.1518, 5.382], # gamma, beta
#                         4: [0.6408, 1.153],
#                         16: [0.1281, 5.6391],
#                         64: [0.12816, 3.3322]}


# # xy mixer, WT11
# starting_points_dict = {1: [5.5110, 0.897], # gamma, beta
#                         4: [0.0, 0.384],
#                         16: [ 0.0, 0.51265],
#                         64: [0.0 , 0.5126]}

##### Optimize
global_results = {}
if OPTIMIZER == "SGD":
    for P in CIRCUIT_DEPTHS:
        r = []
        for n in range(5):
            print(f"Cycle P: {P}\tOPTIMIZER: {OPTIMIZER}\t RUN: {n}")
            gamma = starting_points_dict[P][0] * np.asarray([(2*m-1)/(2*P) for m in range(1,P+1)])
            beta = starting_points_dict[P][1] * np.asarray([1 - (2*m-1)/(2*P) for m in range(1,P+1)])
            starting_point = [gamma, beta]
            print(starting_point)
            angle, cost = SGD_fn(qaoa_fn, starting_point, P)
            counts = qaoa_counts(angle[-1], P=P)
            score = compute_score(counts, df, N_S)
            print(f"Score: {score}")
            print(angle[-1])
            r.append([angle, cost, counts, starting_point, score])
        global_results[f"{OPTIMIZER}_{P}"] = r
        
 
elif OPTIMIZER == "COBYLA":
    for P in CIRCUIT_DEPTHS:
        r = []
        for n in range(5):
            print(f"Cycle P: {P}\tOPTIMIZER: {OPTIMIZER}\t RUN: {n}\t CONST: {CONST}")
            gamma = starting_points_dict[P][0] * np.asarray([(2*m-1)/(2*P) for m in range(1,P+1)])
            beta = starting_points_dict[P][1] * np.asarray([1 - (2*m-1)/(2*P) for m in range(1,P+1)])
            starting_point = np.concatenate((gamma.numpy(), beta.numpy()))
            print(starting_point)
            
            constraints = []
            for i in range(2*P):
                if i < P:
                    # gamma, should increase
                    if i == 0:
                        pass # dont do anything
                    else:
                        constraints.append({'type': 'ineq', 'fun': lambda x: x[i]-x[i-1]})
                else:
                    # beta, should decrease
                    if i == P:
                        pass # dont do anything
                    else: 
                        constraints.append({'type': 'ineq', 'fun': lambda x: x[i-1] - x[i]})
            if CONST == 0:
                constraints = []
                print(f"Eliminating constraints: {constraints}")
                
            cobyla_results = minimize(COBYLA_fn, starting_point, args = (P), constraints=constraints, 
                                      method="COBYLA", options={"maxiter": 500})
            counts = COBYLA_counts(cobyla_results.x, P)
            score = compute_score(counts, df, N_S)
            print(f"Score: {score}")
            print(cobyla_results)
            r.append([cobyla_results.x, cobyla_results.fun, counts, starting_point, score])
        global_results[f"{'SGD'}_{P}"] = r

else:
    raise("OPTIMIZER is not defined!")


print(global_results)

file_name = f"optimization_{STRUCTURE}_{MIXER}_{OPTIMIZER}_{CONST}"            
with open(f'results/{file_name}.pickle', 'xb') as handle:
    pickle.dump({"results": global_results,
                 "circuit_depths": CIRCUIT_DEPTHS,
                 "df_exact_solution": df}, handle, protocol=pickle.HIGHEST_PROTOCOL)       

        
        
        
