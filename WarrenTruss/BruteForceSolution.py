import itertools
import sys
import pickle
import pandas as pd
from tqdm import tqdm

def obj_function(x, N, PHI, K):
    aux=0    
    for p in range(N):
        for q in range(N):
            for i in range(N):
                for j in range(N):
                    aux += abs(PHI[p,i] * K[p,q] * PHI[q,j]) * x[p] * x[q]
    return aux # we are maximizing

N_S = list(range(1, 11))
mass_normalized = [True, False]


for n_s in N_S:
    for m_n in mass_normalized:
        print(f"Processing mass normalized: {m_n} & n_s: {n_s}")
        mn = "_MassNormalized" if m_n else ""
        with open(f"data/WarrenTruss{mn}.pickle", 'rb') as handle:
            data = pickle.load(handle)

        PHI = data["PHI"]
        K = data["K"]
        FreeDOFList = data["FreeDOFList"]
        N = K.shape[0]


        lst = list(itertools.product([0, 1], repeat=N))
        lst = [i for i in lst if sum(i) == n_s]
        obj_value = []
        for l in tqdm(lst):
            obj_value.append(list(l)+[obj_function(l, N, PHI, K)])
            
        df = pd.DataFrame({"data":obj_value}, index=range(len(lst)))
        df = (pd.DataFrame(df['data'].to_list(), columns=[str(i) for i in FreeDOFList] + ["Objective Value"])
            .sort_values(by="Objective Value"))
        
        df.to_csv(f"results/WarrenTrussBruteSolution_MN_{m_n}_{n_s}.csv")
        
        
        