import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import eigs
import time
import os

# --- Configuration based on ICU Theory (Section 28.2) ---
# Level 8 substrate provides high fidelity, but is computationally intensive.
# We'll use a smaller Level 6 grid for a runnable example.
GRID_SIZE = 32  # Represents a cubic voxel space (32x32x32)
NUM_EIGENPAIRS = 100  # Number of stable resonant modes to compute

# The Kernel Law governs interactions. A simple Laplacian is a good proxy.
# A real-world implementation would have a more complex kernel.
KERNEL_STIFFNESS = 1.0

def get_1d_index(x, y, z, size):
    """Converts a 3D coordinate to a 1D array index."""
    return x + y * size + z * size * size

def build_laplacian_matrix(size):
    """
    Builds the discrete Laplacian operator for the voxel grid.
    This matrix represents the "Rule of Interaction" or the Kernel Law,
    forming the basis of the eigenproblem Lϕ = λϕ.
    """
    total_voxels = size * size * size
    # Use a sparse matrix for efficiency, as most entries are zero.
    matrix = lil_matrix((total_voxels, total_voxels))

    print(f"Building {size}x{size}x{size} interaction matrix ({total_voxels} voxels)...")

    for z in range(size):
        for y in range(size):
            for x in range(size):
                idx = get_1d_index(x, y, z, size)
                
                # Apply connections to neighbors (6 nearest neighbors for a cubic lattice)
                neighbors = 0
                for dx, dy, dz in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
                    nx, ny, nz = x + dx, y + dy, z + dz
                    
                    # Enforce periodic boundary conditions (a toroidal substrate)
                    if 0 <= nx < size and 0 <= ny < size and 0 <= nz < size:
                        n_idx = get_1d_index(nx, ny, nz, size)
                        matrix[idx, n_idx] = -KERNEL_STIFFNESS
                        neighbors += 1
                
                matrix[idx, idx] = neighbors * KERNEL_STIFFNESS

    print("Interaction matrix built.")
    return matrix.tocsr() # Convert to CSR format for faster computations

def solve_eigenproblem(matrix, num_solutions):
    """
    Solves the eigenproblem to find the stable resonant modes (eigenmodes)
    of the substrate. These modes correspond to the fundamental particles/orbitals.
    """
    print(f"Solving for the {num_solutions} lowest-energy resonant modes...")
    # 'SM' finds the eigenvalues with the smallest magnitude, representing the most stable states.
    eigenvalues, eigenvectors = eigs(matrix, k=num_solutions, which='SM')
    
    # Sort the results by the real part of the eigenvalue
    sorted_indices = np.argsort(eigenvalues.real)
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    
    print("Eigenproblem solved.")
    return sorted_eigenvalues.real, sorted_eigenvectors.real

def main():
    """
    Main execution function to run the full simulation pipeline.
    """
    start_time = time.time()
    
    # Ensure the /data directory exists for the output files
    if not os.path.exists('data'):
        os.makedirs('data')
        print("Created /data directory for output.")

    # 1. Build the substrate's interaction matrix based on the Kernel Law
    laplacian = build_laplacian_matrix(GRID_SIZE)

    # 2. Solve for the resonant eigenmodes
    eigenvalues, eigenvectors = solve_eigenproblem(laplacian, NUM_EIGENPAIRS)
    
    # 3. Save the results for the visualization script
    eigenvalue_path = os.path.join('data', 'eigenvalues.npy')
    eigenvector_path = os.path.join('data', 'eigenvectors.npy')
    
    np.save(eigenvalue_path, eigenvalues)
    np.save(eigenvector_path, eigenvectors)
    
    print(f"\nSuccessfully computed {len(eigenvalues)} resonant modes.")
    print(f"Results saved to '{eigenvalue_path}' and '{eigenvector_path}'")
    
    end_time = time.time()
    print(f"Total simulation time: {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
