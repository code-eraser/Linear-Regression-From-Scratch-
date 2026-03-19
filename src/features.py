import numpy as np
import matplotlib.pyplot as plt
def map_features(x1,x2):
     return np.stack([
        x1,
        x2,
        x1*x2,
        x1**2,
        x2**2
], axis = -1)
