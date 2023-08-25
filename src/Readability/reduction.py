import pandas as pd 
from src.Readability.readability import * 
from src.scrapers import * 
from numpy.linalg import svd
import umap
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as sk_PCA

def getMatrix(n=66_000, process=False): 
    # n = number of rows we get up to 
    # process=False means that we just retrieve stored matrix 
    
    if process == False: 
        retrieval = pd.read_csv("src/Readability/design_matrix_Final.csv", index_col = 0)
        return retrieval
        
    else: 
    
        df = pd.DataFrame(columns=['RIX', 'ARI', 'CL', 'DB', 'DSH', 'ELF', 'FJP', 'Flesch', 'FK', 'FORCAST', 'FS', 'FOG', 'LW', 'LIX', 'nWS', 'SMOG', 'SI', 'WS'])
        
        for i in range(n): 
            formatted_num = '{:05d}'.format(i)
            try: 
                doc = get_doc(f"data/html{formatted_num}.txt")
                readability = readabilityScores(doc)
                readability.index = [f"html{formatted_num}"]
                df = pd.concat([df, readability])
                
            except: 
                pass 
            
            print(i) 
            
        return df

def PCA(mat:pd.DataFrame): 
    
    mat = pd.DataFrame.to_numpy(mat) 
    mat = StandardScaler().fit_transform(mat)
    
    pca = sk_PCA()
    pca.fit_transform(mat)
    
    return pca

def UMAP(mat:pd.DataFrame, n:int, process=False): 
    
    if process == False: 
        with open(f"src/Readability/umap{n}", "rb") as fp: 
            embed = pickle.load(fp) 
            
        return embed
    
    else: 
        
        mat = pd.DataFrame.to_numpy(mat)
        
        reducer = umap.UMAP(n_components=n)
        
        scaled_mat = StandardScaler().fit_transform(mat)
        
        embedding = reducer.fit_transform(scaled_mat)
        
        with open(f"src/umap{n}", "wb") as fp: 
            pickle.dump(embedding, fp) 
        
        return embedding 
     
    
    