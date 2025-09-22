import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# --- Configuration (Must match compute_resonance_eigenpairs.py) ---
GRID_SIZE = 32
EIGENMODE_TO_PLOT = 1  # 0=1s, 1=2p(x), 2=2p(y), 3=2p(z), etc.

def load_data():
    """Loads the pre-computed eigenvalues and eigenvectors."""
    eigenvalue_path = os.path.join('data', 'eigenvalues.npy')
    eigenvector_path = os.path.join('data', 'eigenvectors.npy')

    if not os.path.exists(eigenvalue_path) or not os.path.exists(eigenvector_path):
        print("Error: Data files not found.")
        print(f"Please run 'compute_resonance_eigenpairs.py' first to generate the data.")
        return None, None
        
    print("Loading computed data...")
    eigenvalues = np.load(eigenvalue_path)
    eigenvectors = np.load(eigenvector_path)
    print("Data loaded successfully.")
    return eigenvalues, eigenvectors

def plot_orbital(eigenvector, eigenvalue, mode_index, size):
    """
    Visualizes a single eigenmode as a 3D probability cloud, analogous
    to an atomic orbital.
    """
    # The eigenvector is a 1D array; reshape it to the 3D grid of the substrate.
    wavefunction = eigenvector.reshape((size, size, size))
    
    # The probability density is the square of the wavefunction.
    probability_density = wavefunction**2
    
    print(f"Generating 3D plot for eigenmode {mode_index} (Energy: {eigenvalue:.4f})...")

    # Create coordinates for the plot
    x, y, z = np.mgrid[:size, :size, :size]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Use a threshold to only plot the densest parts of the cloud
    threshold = probability_density.max() * 0.1

    # Use a scatter plot where color and size represent probability
    ax.scatter(x[probability_density > threshold], 
               y[probability_density > threshold], 
               z[probability_density > threshold], 
               c=probability_density[probability_density > threshold].flatten(), 
               alpha=0.6)

    ax.set_title(f'Resonant Eigenmode {mode_index} ("Orbital")\nEnergy Level: {eigenvalue:.4f}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.show()

def main():
    """
    Main function to load results and generate the visualization.
    """
    eigenvalues, eigenvectors = load_data()
    
    if eigenvalues is None or eigenvectors is None:
        return

    if EIGENMODE_TO_PLOT >= len(eigenvalues):
        print(f"Error: EIGENMODE_TO_PLOT ({EIGENMODE_TO_PLOT}) is out of bounds.")
        print(f"Only {len(eigenvalues)} modes were computed.")
        return

    # Select the specific eigenmode to visualize
    selected_eigenvalue = eigenvalues[EIGENMODE_TO_PLOT]
    selected_eigenvector = eigenvectors[:, EIGENMODE_TO_PLOT]
    
    plot_orbital(selected_eigenvector, selected_eigenvalue, EIGENMODE_TO_PLOT, GRID_SIZE)

if __name__ == "__main__":
    main()
