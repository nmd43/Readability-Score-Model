# dataplus-2023-techdoc

We use various machine learning methods to predict readability scores of technical documentation of NetApp products. After cloning this repository, create a virtual environment using conda and install the packages specified by `requirements.txt` to make sure all dependency requirements are met. 
```
conda create --name myEnv 
conda activate myEnv 
pip install -r requirements.txt 
```
  

### Readability Scores 

Features: 
1. Total number of words 
2. Total unique words 
3. Total number of sentences 
4. Total number of characters 
5. Total syllables
6. Total number of words with strictly more than 6 characters 
7. Total number of words with at least 6 characters 
8. Total number of blanks (spaces, indents, newlines)

9. Number of words with strictly more than 3 syllables 
10. Number of words with less than or equal to 3 syllables 
11. Number of words with strictly more than 2 syllables 
12. Number of words with exactly 1 syllable. 

 

Some of the 17 readability scores that we considered are: 
1. Flesh-Kincaid
2. Coleman-Liau
3. Wheeler-Smith
4. Anderson's Readability Index
5. SMOG
6. Kuntzsch's Text-Redundanz-Index
7. Farr-Jenkins-Paterson

Then we do PCA, UMAP, or tSNE on them. 

### N-Gram Information Entropy 

Given document $\mathcal{D}$, let $\mathcal{V}$ be the set of all unique words in the document, called the vocabulary, with some discrete probability measure $\mathbb{P}$ defined on it. If we consider the n-gram model, we want to consider the joint measure $\mathbb{P}_n$ defined on $\mathcal{V}^n$, the set of all sequences of $n$ words in $\mathcal{D}$ (plus some start or end tokens). That is, given a sequence of $n-1$ words $w_{k+1}, w_{k+2}, \ldots, w_{k+n-1}$, we can generate the next word in the language model by sampling from the conditional distribution 
$$\mathbb{P}(w_{k+n} \mid w_{k+1}, w_{k+2}, \ldots, w_{k+n-1})$$

The entropy of a discrete random variable $X$ with support $\mathcal{X}$ is 
$$H(X) \coloneqq - \sum_{x \in \mathcal{X}} \mathbb{P}_X (x) \, \log_2 \mathbb{P}_X (x)$$

Intuitively, the entropy measures the average number of bits needed to encode the outcomes of the random variable. If the random variable is highly unpredictable and has many possible outcomes with roughly equal probabilities, the entropy will be high. On the other hand, if the random variable is highly predictable or has a few dominant outcomes, the entropy will be low. 

Therefore, a low entropy would indicate that it is easy for the reader to determine the next word, since there are a few dominant outcomes. For example, the entropy of the phrase 
```
I hope you had a great ___. 
```
may be high since probability distribution of the next word over the entire vocabulary of the English language may be highly concentrated around the words "day", "time", or "night," indicating that it is easy to predict the next word and that readability is high. In contrast, the entropy of the phrase 
```
Surreptitious serendipity amidst chaotic ____. 
```
will be high since it incorporates a mix of intricate vocabulary and contrasting elements to create a sense of complexity and unpredictability. 


We must consider the fact that as $n$ gets large, $\mathbb{P}_n$ gets very sparse, and so we cannot choose $n$ to be too large. Practically, we should have $2 \leq n \leq 5$. 

