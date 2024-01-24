import pandas as pd
from sklearn.datasets import make_classification

from src.boruta import boruta

X, y = make_classification(n_samples=1000, n_features=20,
                           n_informative=10, n_redundant=5,
                           n_clusters_per_class=1, random_state=0)
# X, y = make_classification(n_samples=1000, n_features=2,
#                            n_informative=1, n_redundant=1,
#                            n_clusters_per_class=1, random_state=0)
X: pd.DataFrame = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(X.shape[1])])

print(boruta(X, y))