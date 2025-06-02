import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

'''
Method 1: Dynamic Time Warping (DTW)
'''

def dtw_distance(x: np.ndarray, y: np.ndarray) -> float:
     # only euclidean cost function for now
     distance, _ = fastdtw(x, y, dist=euclidean)
     return distance

'''
Method 2: Longest Common Subsequence (LCS)
'''

def lcs_distance(x: np.ndarray, y: np.ndarray) -> float:

     x = list(np.round(x, 4))  # round to make values comparable
     y = list(np.round(y, 4))

     m, n = len(x), len(y)
     dp = [[0]*(n+1) for _ in range(m+1)]

     for i in range(m):
          for j in range(n):
               if x[i] == y[j]:
                    dp[i+1][j+1] = dp[i][j] + 1
               else:
                    dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])

     lcs_len = dp[m][n]
     max_len = max(m, n)
     return 1.0 - (lcs_len / max_len)