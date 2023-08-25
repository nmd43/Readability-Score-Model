from gensim.utils import simple_preprocess
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from scrapers import get_doc
import pandas as pd 
import os, time 

def doc2vec(process=False):  
    
    if process == False: 
        df = pd.read_csv("src/DeepPipe/embeddings.csv")
        df.drop(columns=df.columns[0], axis=1,  inplace=True)
        return df.to_numpy()
    
    else: 
        documents = []

        for i, file in enumerate(sorted(os.listdir("data"))): 
            file_num = int(file[4:-4])
            text = get_doc(f"data/{file}") 
                
            tokens = simple_preprocess(text) 
            documents.append(TaggedDocument(tokens, [i]))
        
        print(len(documents)) 
                
        # Train doc2vec model 
        model = Doc2Vec(documents, vector_size=300, workers=12) 
        
        print(model.dv.vectors.shape)
        
        df = pd.DataFrame(
            model.dv.vectors, 
            index = sorted(os.listdir("data"))
            
        )
        
        df.to_csv("src/DeepPipe/embeddings.csv")
        
        return model.dv.vectors

# embed = doc2vec() 
# print(embed)

# model.build_vocab(documents)
# word = 'docker'
# print(f"Word '{word}' appeared {model.wv.get_vecattr(word, 'count')} times in the training corpus.")
