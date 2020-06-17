import numpy as np
import mglearn
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF, PCA

S = mglearn.datasets.make_signals()
plt.figure(figsize=(11, 2))
plt.plot(S, '-')
plt.xlabel("Time")
plt.ylabel("Signal")
plt.tight_layout()
plt.show()

A = np.random.RandomState(0).uniform(size=(100, 3))
X = np.dot(S, A.T)
print("Shape of measurements: {}".format(X.shape))


nmf = NMF(n_components=3, random_state=42)
S_ = nmf.fit_transform(X)
print("Recovered signal shape: {}".format(S_.shape))

pca = PCA(n_components=3)
H = pca.fit_transform(X)

models = [X, S, S_, H]
names = ['Observations (first three measurements)',
         'True sources',
         'NMF recovered signals',
         'PCA recovered signals']

fig, axes = plt.subplots(4, figsize=(12, 6), gridspec_kw={'hspace': .5},
                         subplot_kw={'xticks': (), 'yticks': ()})

for model, name, ax in zip(models, names, axes):
    ax.set_title(name)
    ax.plot(model[:, :3], '-')

# plt.savefig('fig.png', bbox_inches='tight')
