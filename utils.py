from pennylane import numpy as np
import pandas as pd
import itertools
import pennylane as qml
import networkx as nx


def app_ratio_fn(f, b, w):
    return (f-w) / (b-w)

def counts_to_df(counts, A, b, w):
    keys_index = [np.where(np.array(list(key), dtype=int)==1)[0] for key,_ in counts.items()]
    keys = [np.array(list(key), dtype=int) for key,_ in counts.items()]
    f_obj = [np.asarray(x).T @ A @ np.asarray(x) for x in keys]
    rep = [i for _, i in counts.items()]
    app_ratio = [app_ratio_fn(f, b, w) for f in f_obj]
    df = pd.DataFrame(data={"solutions": keys, "solutions_index": keys_index,
                            "f_obj": f_obj, "counts": rep,
                           "app_ratio": app_ratio})
    return df.sort_values(by="counts", ascending=False).reset_index(drop=True)
                  
    
def random_symmetric_matrix(N):
    # Create a random N by N matrix with random values
    rand_matrix = np.random.rand(N, N)*2 - 1
    # Make the matrix symmetric by copying the upper triangular part to the lower triangular part
    sym_matrix = np.triu(rand_matrix, k=0) + np.triu(rand_matrix, k=1).T
    assert np.array_equal(sym_matrix, sym_matrix.T)
    return sym_matrix

def solve_qubo(A):
    N = np.shape(A)[0]
    # Solve the QUBO problem testing all combinations
    candidate_solutions = [x for x in itertools.product([0, 1], repeat=N)]
    candidate_solutions_index = [np.where(np.array(x)==1)[0] for x in candidate_solutions]
    # compute objective function
    f_obj = [np.asarray(x).T @ A @ np.asarray(x) for x in candidate_solutions]
    df_exact_solutions = pd.DataFrame(data={"Candidate Solutions": candidate_solutions,
                                            "Candidate Solutions Index": candidate_solutions_index,
                                            "f_obj": f_obj})
    df_exact_solution = df_exact_solutions.sort_values(by="f_obj", ascending=False).reset_index(drop=True)
    return df_exact_solution

def GenerateTermsMatrix(Phi, K):
    N = K.shape[0]
    # Generate terms matrix
    terms = np.zeros((N, N))
    for p in range(N):
        for q in range(N):
            aux = 0
            for i in range(Phi.shape[1]):
                for j in range(Phi.shape[1]):
                    aux += abs(Phi[p,i] * K[p,q] * Phi[q,j])
            terms[p, q] = aux
    return terms

def GenerateConstrainTerms(N, n_s):
    constrain_terms = np.ones((N, N)) - 2 * np.diag([n_s] * N)
    constrain_offset = n_s**2
    return constrain_terms, constrain_offset


def GenerateGraph(N):
    G = nx.Graph()
    G.add_nodes_from(range(N))
    for i in range(N):
        for j in range(i):
            G.add_edge(j, i)
    return G


def obs_identity(N):
    result = qml.Identity(0)
    for i in range(1,N):
        result = result @ qml.Identity(i)
    return result
    
def obs_Z(N, i):
    result = qml.PauliZ(0) if i == 0 else qml.Identity(0)
    for j in range(1,N):
        term = qml.PauliZ(j) if j == i else qml.Identity(j)
        result = result @ term
    return result

def obs_ZZ(N, i, j):
    result = qml.PauliZ(0) if i == 0 or j == 0 else qml.Identity(0)
    for k in range(1,N):
        term = qml.PauliZ(k) if k == i or k == j else qml.Identity(k)
        result = result @ term
    return result

def generate_cost_hamiltonian(A, N):
    obs = []
    coeffs = []
    for i in range(A.shape[0]):
        for j in range(i, A.shape[0]):
            # Diagonal terms are linear terms 
            if i == j:
                obs.extend([obs_identity(N), obs_Z(N, i)])
                coeffs.extend([0.5 * A[i,j], -0.5 * A[i,j]])
            # off-diagonal terms are quadratic terms
            else:
                obs.extend([obs_identity(N), obs_Z(N, i), obs_Z(N, j), obs_ZZ(N, i, j)])
                coeffs.extend([0.25 * 2 * A[i,j], -0.25 * 2 * A[i,j], -0.25 * 2 * A[i,j], 0.25 * 2 * A[i,j]])

    H = qml.Hamiltonian(coeffs, obs)
    if N < 8:
        matrix = 0
        for coeff, op in zip(H.coeffs, H.ops):
            matrix += coeff * op.matrix()
    else:
        matrix = None
    
    return H, matrix


def normalized_performance(x, maxx, minn):
    return (x-minn)/(maxx-minn)

def from_binary_to_int(b):
    return b.dot(2**np.arange(b.size)[::-1])


def compute_score(probs, df_, N_S, verbose=False):
    df = df_.copy()
    df["fulfill"] = df["Candidate Solutions"].apply(lambda x: float(sum(x)==N_S))
    # get max and min for feasible 
    best = df[df.fulfill == 1].f_obj.max().item()
    worst = df[df.fulfill == 1].f_obj.min().item()
    df["int_representation"] = df["Candidate Solutions"].apply(lambda x: from_binary_to_int(np.asarray(x)))
    df["probs"] = df.int_representation.apply(lambda i: probs[i])
    df["normalized_performance"] = df.f_obj.apply(lambda x: normalized_performance(x, best, worst)) * df.fulfill
    if verbose:
        display(df.head(10))
    return sum(df.probs * df.normalized_performance)



def compute_score2(probs, df_):
    df = df_.copy()
    df["probs"] = df.int_representation.apply(lambda i: probs[i])
    return sum(df.probs * df.normalized_performance)