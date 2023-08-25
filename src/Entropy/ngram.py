from bs4 import BeautifulSoup 
import random, time
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import ngrams
from pprint import pprint
from numpy.linalg import norm 

class NgramModel(object): 
    
    def __init__(self, N:int, paths:list[str]): 
        
        
        self.N = N 
        self.context = {}
        
        for path in paths: 
            with open(path, 'r') as f:
                sentences = sent_tokenize(f.read().lower()) 
                
                for sentence in sentences: 
                    tokens = word_tokenize(sentence)
                
                    grams = ngrams(tokens, N, 
                        pad_left=True, 
                        pad_right=True, 
                        left_pad_symbol='<s>', 
                        right_pad_symbol='</s>'
                        )
                    
                    for tup in grams: 
                        prev_words = tup[:-1] 
                        next_word = tup[-1]
                        if prev_words in self.context: 
                            self.context[prev_words].append(next_word)
                        else: 
                            self.context[prev_words] = [next_word]
                            
        # finally add the (</s>, </s>) -> <s> | (</s>, <s>) -> <s> ...
        final = ["</s>"] * (self.N - 1) 
        for _ in range(self.N - 1): 
            self.context[tuple(final)] = ["<s>"]
            final.pop(0) 
            final.append("<s>")
            
    def vocabulary(self): 
        return [len(self.context[key]) for key in self.context]
    
    def PQIndex(self, p, q): 
        # PQ Index (inspired by Gini Index) measuring sparsity of vector
        w = self.vocabulary() 
        d = len(w)
        
        return 1 - d ** ((1/p)-(1/q)) * (norm(w, p)/norm(w, q))
        
        
                    
    def next_rand_word(self, prev_words:tuple): 
        return random.choice(self.context[prev_words])
    
    def generate_text(self, n_words = 50): 
        
        result = []
        prev = ["<s>"] * (self.N - 1)
        
        step = 0
        while step < n_words: 
            next = self.next_rand_word(tuple(prev)) 
            
            if next not in ["<s>", "</s>"]: 
                result.append(next) 
                step += 1
            
            prev.pop(0) 
            
            prev.append(next)

        out = " ".join(result)
        print(out)
        
        return out 
        

# frankenstein = "data/html00000.txt"


# m = NgramModel(3, [frankenstein])

# print(m.PQIndex(2, 1))

# print(f'{"="*50}\nGenerated text:')
# out = m.generate_text(200)
# print(f'{"="*50}')



