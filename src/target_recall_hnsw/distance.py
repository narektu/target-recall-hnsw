import numpy as np

def l2_squared(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    diff = a - b
    return np.dot(diff, diff)

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 1.0
    return 1.0 - float(np.dot(a, b) / denom)
