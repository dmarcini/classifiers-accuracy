import numpy as np
import pandas as pd

from scipy.stats import rankdata
from scipy.stats import ranksums

from experimental_environment import clfs
from experimental_environment import features

def accuracy(mean_scores):
  df = pd.DataFrame(
    mean_scores[1:],
    columns=list(clfs.keys()),
    index=[x for x in range(1, len(features))]
  )

  return df

def ranks(mean_scores):
  ranks = []

  for ms in mean_scores:
    ranks.append(rankdata(ms).tolist())

  ranks = np.array(ranks)

  df = pd.DataFrame(
    ranks[1:],
    columns=list(clfs.keys()),
    index=[x for x in range(1, len(features))]
  )

  return df

def steam_tests(ranks):
  ranks = np.array(ranks)

  alfa = .05
  w_statistic = np.zeros((len(clfs), len(clfs)))
  p_value = np.zeros((len(clfs), len(clfs)))

  for i in range(len(clfs)):
    for j in range(len(clfs)):
      w_statistic[i, j], p_value[i, j] = ranksums(ranks.T[i], ranks.T[j])

  advantage = np.zeros((len(clfs), len(clfs)))
  advantage[w_statistic > 0] = 1

  advantage_df = pd.DataFrame(
    advantage,
    columns=list(clfs.keys()),
    index=list(clfs.keys())
  )

  significance = np.zeros((len(clfs), len(clfs)))
  significance[p_value <= alfa] = 1

  significance_df = pd.DataFrame(
    significance,
    columns=list(clfs.keys()),
    index=list(clfs.keys())
  )

  return advantage_df, significance_df

def main():
  mean_scores = np.mean(np.load('results/results.npy'), axis=2).T

  accuracy_df = accuracy(mean_scores)
  ranks_df = ranks(mean_scores)
  advantage_df, significance_df = steam_tests(ranks_df)

  print('\nAccuracy:')
  print(accuracy_df)
  print('\nRanks:')
  print(ranks_df)
  print('\nAdvantage:')
  print(advantage_df)
  print('\nStatistical signifiacance (alpha = 0.5):')
  print(significance_df)

if __name__ == '__main__':
  main()
