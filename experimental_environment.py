import numpy as np
import pandas as pd

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.feature_selection import chi2, f_classif

from sklearn.base import clone
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score

clfs = {
  'MLP': MLPClassifier(max_iter=500),
  'kNN': KNeighborsClassifier(),
  'GNB': GaussianNB(),
  'SVC': SVC(),
  'DTC': DecisionTreeClassifier(random_state=42),
  'LR': LogisticRegression(random_state=42)
}

features = [
  'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
  'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num'
]

n_splits = 5
n_repeats = 2

rskf = RepeatedStratifiedKFold(n_splits=n_splits,
                               n_repeats=n_repeats,
                               random_state=42)

scores = np.zeros((
  len(clfs),
  len(features),
  n_splits * n_repeats
))

def main():  
  df = pd.DataFrame(
    np.array(pd.read_csv('dataset/heart-disease-dataset.csv', sep=';')),
    columns=features
  )
  df = df[(df != '?').all(1)]

  X = np.array(df.iloc[:, :-1])
  y = np.array(df.iloc[:, -1].astype(int))
  
  features_rank = np.array(f_classif(X, y)[0]).argsort()[::-1]

  X = X[:, features_rank]

  for fold_id, (train, test) in enumerate(rskf.split(X, y)):
    for clf_id, clf_name in enumerate(clfs):
      for n_features in range(1, len(features)):
        clf = clone(clfs[clf_name])

        X_train = X[train, :n_features]
        X_test = X[test, :n_features]

        clf.fit(X_train, y[train])

        y_pred = clf.predict(X_test)

        scores[
          clf_id,
          n_features,
          fold_id
        ] = accuracy_score(y[test], y_pred)
  
  np.save('results/results', scores)

if __name__ == '__main__':
  main()
