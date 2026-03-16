import numpy as np
import matplotlib.pyplot as plt

def plot_decision_boundary(X,y,w,b):

    plt.scatter(X[y==0,0], X[y==0,1], label="Class 0")
    plt.scatter(X[y==1,0],X[y==1,1], label="Class 1")

    # Create Grid
    u = np.linspace(np.min(X[:,0]), np.max(X[:,0]), 100)
    v = np.linspace(np.min(X[:,1]), np.max(X[:,1]), 100)

    U, V = np.meshgrid(u,v)

    # Map features
    from .features import map_features
    X_poly = map_features(U, V)

    # Compute Z
    Z = np.dot(X_poly,w) + b
    
    # Plot contour
    plt.contour(U, V, Z, levels = [0])
    plt.xlabel("x1")
    plt.ylabel("y1")
    plt.legend()
    plt.show()