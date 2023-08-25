from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize import RegexpTokenizer
from syllables import estimate 
from math import sqrt
from pprint import pprint 
from collections import OrderedDict
import pandas as pd 
from string import punctuation

def features(document:str): 
    
    tokenizer = RegexpTokenizer(r'\w+')
    
    nSentences = len(sent_tokenize(document))
    nCharacters = len(document)
    
    
    words = tokenizer.tokenize(document)
    words = [word.lower() for word in words]
    
    nWords = len(words) 
    nUniqueWords = len(set(words)) 
    
    nSyllables = 0 
    nMore6Char = 0 
    nAtLeast6Char = 0
    nMore3Syllables = 0 
    nMore2Syllables = 0 
    nMore1Syllables = 0
    nExact1Syllable = 0
    punctuations = 0
    
    
    for word in words: 
        syllables = estimate(word)
        length = len(word) 
        nSyllables += syllables 
        
        if word in punctuation: 
            punctuations += 1 
            continue 
        
        if length > 6: 
            nAtLeast6Char += 1 
            nMore6Char += 1 
        elif length == 6: 
            nAtLeast6Char += 1 
        
        if syllables > 3: 
            nMore3Syllables += 1 
            nMore2Syllables += 1 
            nMore1Syllables += 1
        elif syllables == 3: 
            nMore2Syllables += 1 
            nMore1Syllables += 1
        elif syllables == 2: 
            nMore1Syllables += 1
        elif syllables == 1: 
            nExact1Syllable += 1
            
    
    out = {
        "Total Words" : nWords, 
        "Unique Words" : nUniqueWords, 
        "Total Sentence" : nSentences, 
        "Total Syllables" : nSyllables, 
        "Total Chars" : nCharacters, 
        "Longish Words" : nAtLeast6Char,                # geq 6 characters
        "Blanks" : nWords - 1,                          # num of spaces 
        "geq2 Syll Words" : nMore1Syllables, 
        "1 Syll Words" : nExact1Syllable,
        "geq3 Syll Words" : nMore2Syllables, 
        "Punctuations" : punctuations,
        "< 3 Syll Words" :  nWords - nMore2Syllables, 
        "Long Words" : nMore6Char                       # strictly > 6 characters 
    }
    
    return out
    

def readabilityScores(document:str): 
    
    Features = features(document)

    ASL = Features["Total Words"] / Features["Total Sentence"]
    AWL = Features["Total Chars"] / Features["Total Words"]
    

    RIX = Features["Long Words"] / Features["Total Sentence"]
    
    ARI = 0.5 * ASL + 4.71 * AWL - 21.34
    
    CL = 5.88 * AWL + 29.6 * (Features["Total Sentence"]/Features['Total Words']) - 15.8
    
    DB = (1.0364 * Features["Total Chars"] / Features["Blanks"]) + (
                0.0194 * Features["Total Chars"] / Features["Total Sentence"]) - 0.6059
    
    DSH = 235.95993 - (7.3021* AWL) - (12.56438 * ASL) - (50.03293 * Features["Unique Words"] / Features["Total Words"] )
    
    ELF = Features["geq2 Syll Words"] / Features ["Total Sentence"]
    
    FJP = -31.517 - (1.015 * ASL) + (1.599 * Features["1 Syll Words"] / Features["Total Words"])
    
    Flesch = 206.835 - (1.015 * ASL) - (84.6 * Features["Total Syllables"] / Features["Total Words"])
    
    FK = 0.39 * ASL + (11.8 * Features["Total Syllables"] / Features["Total Words"]) - 15.59
    
    FORCAST = 20 - ((Features["1 Syll Words"] * 150) / (10 * Features["Total Words"]))
    
    FS = AWL * ASL
    
    FOG = 0.4 * (ASL + 100 * (Features["geq3 Syll Words"] / Features["Total Words"]))
   
    LW =  (100 - (100*Features["< 3 Syll Words"])/(Features["Total Words"]) +
                (3 * (100 * Features["geq3 Syll Words"])/(Features["Total Words"])) ) / ( (100 * Features["Total Sentence"]) / (Features["Total Words"]) ) 
    
    LIX = ( (Features["Total Words"]/ Features["Total Sentence"]) + ( 100 * Features["Long Words"] / Features["Total Words"]) )
    
    nWS = 19.35 * ( Features["geq3 Syll Words"]/ Features["Total Words"] ) + 0.1672 * ASL \
            + 12.97 * (Features["Longish Words"]/ Features["Total Words"]) - 3.27 * ( Features["1 Syll Words"]/ Features["Total Words"]) - 0.875
            
    SMOG = 1.043 * sqrt(Features["geq3 Syll Words"]) * (30 / Features["Total Sentence"]) + 3.1291
    
    SI = 30 * Features["Total Syllables"] / Features["Total Sentence"]
    
    WS = ASL * 10 * (Features["geq2 Syll Words"]/Features["Total Words"])
    

    out = {
        "RIX": RIX,
        "ARI": ARI,
        "CL": CL,
        "DB": DB,
        "DSH": DSH,
        "ELF": ELF,
        "FJP": FJP,
        "Flesch": Flesch,
        "FK": FK,
        "FORCAST": FORCAST,
        "FS": FS,
        "FOG": FOG,
        "LW": LW,
        "LIX": LIX,
        "nWS": nWS,
        "SMOG": SMOG,
        "SI": SI,
        "WS": WS
    }
    
    return pd.DataFrame([out])