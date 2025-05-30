'''
This file contains the implementation of a few basic distance metrics to measure
distance between two time series x, y (in our case, 2 different crypto asset prices): 
- Sum of Squared Distances (SSD)
- Eucliean Distance
- Manhattan Distance
- Correlation Distance
'''
import numpy as np

def ssd(x: np.ndarray, y: np.ndarray) -> float:
    return np.sum((x - y)**2)

def euclidean(x: np.ndarray, y: np.ndarray) -> float:
    return np.sqrt(np.sum((x - y)**2))

def manhattan(x: np.ndarray, y: np.ndarray) -> float:
    return np.sum(np.abs(x - y))

def correlation_distance(x: np.ndarray, y: np.ndarray) -> float:
    return 1 - np.corrcoef(x, y)[0, 1]

