from src.Readability.readability import * 
from src.scrapers import * 
from src.Readability.reduction import * 
import time, pickle
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import plotly.express as px

    
np.random.seed(0)

design_matrix = getMatrix() 

# pca = PCA(design_matrix)
# S = pca.singular_values_
# plt.xlabel("Index")
# plt.ylabel("Singular Values")
# plt.xticks(range(1, 19))
# plt.plot(S, c="g")
# plt.show()

# assert False 

mat = pd.DataFrame.to_numpy(design_matrix) 
mat = StandardScaler().fit_transform(mat)
pca = sk_PCA(n_components=2)

pca_scores = pca.fit_transform(mat)

principal_scores_Df2 = pd.DataFrame(data = pca_scores)
#print(principal_scores_Df2.tail())
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

plt.scatter(*zip(*pca_scores), c="b", s=1)
plt.xlim([-5, 30])
plt.ylim([-10, 5])
plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")
plt.show() 

assert False 

total_var_2 = pca.explained_variance_ratio_.sum() * 100
fig = px.scatter(pca_scores, x=0, y=1, title=f'Total Explained Variance: {total_var_2:.2f}%')
fig.show()

# S = pca.singular_values_
# plt.plot(S, c="g")
# plt.xlabel("Index")
# plt.ylabel("Singular Value")
# plt.show()

# plt.plot(range(len(S)), S)
# plt.xlabel("Singular Index")
# plt.ylabel("Singular Values")
# plt.xticks(range(1, len(S)+1))
# plt.title("PCA on Design Matrix of Readability Scores")
# plt.show()


# now = time.time()
# embed1 = UMAP(design_matrix, 1, process=True)
# print(embed1.shape)
# print(f"Time Taken : {time.time() - now}")


# plt.scatter(*zip(*embed2), s=1) 
# plt.show()
# plt.close()

# fig = plt.figure()
# fig.set_size_inches(18.5, 10.5)
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(*zip(*embed3), s=1)
# plt.show()
# plt.close()