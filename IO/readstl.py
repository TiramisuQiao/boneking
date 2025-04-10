import os
import numpy as np
import trimesh
from rich.console import Console

# Initialize the rich console for colored output
console = Console()

def process_stl_to_voxel(stl_path, output_dir, vox_dim=60):
    """
    Loads an STL file, normalizes the mesh into the [0, 3] range,
    voxelizes it into a (vox_dim x vox_dim x vox_dim) 0-1 array, and
    saves the compressed npz file to the specified directory. The output
    file is named as the original file name appended with '_voxel.npz'.

    Parameters:
        stl_path: str
            The path to the STL file.
        output_dir: str
            The target directory where the npz file will be saved.
        vox_dim: int, optional (default 60)
            The desired voxel matrix size (vox_dim x vox_dim x vox_dim).

    Returns:
        voxel_matrix: numpy.ndarray
            The voxelized 0-1 array with shape (vox_dim, vox_dim, vox_dim).
    """
    mesh = trimesh.load(stl_path)
    if mesh.is_empty:
        raise ValueError("Failed to load STL model or it is empty. Please check file: " + stl_path)
    bbox_min, bbox_max = mesh.bounds
    mesh.apply_translation(-bbox_min)
    extent = bbox_max - bbox_min
    scale_factor = 3.0 / extent.max()
    mesh.apply_scale(scale_factor)
    pitch = 3.0 / vox_dim
    voxel_obj = mesh.voxelized(pitch=pitch)
    voxel_matrix = voxel_obj.matrix.astype(np.uint8)
    current_shape = np.array(voxel_matrix.shape)
    desired_shape = np.array([vox_dim, vox_dim, vox_dim])
    
    for d in range(3):
        if current_shape[d] > desired_shape[d]:
            start = (current_shape[d] - desired_shape[d]) // 2
            end = start + desired_shape[d]
            if d == 0:
                voxel_matrix = voxel_matrix[start:end, :, :]
            elif d == 1:
                voxel_matrix = voxel_matrix[:, start:end, :]
            elif d == 2:
                voxel_matrix = voxel_matrix[:, :, start:end]
            current_shape[d] = desired_shape[d]
    
    pad_width = []
    for d in range(3):
        diff = desired_shape[d] - voxel_matrix.shape[d]
        if diff > 0:
            pad_before = diff // 2
            pad_after = diff - pad_before
            pad_width.append((pad_before, pad_after))
        else:
            pad_width.append((0, 0))
    voxel_matrix = np.pad(voxel_matrix, pad_width, mode='constant')

    # Ensure the voxel matrix is exactly vox_dim x vox_dim x vox_dim
    voxel_matrix = voxel_matrix[:vox_dim, :vox_dim, :vox_dim]
    base_name = os.path.splitext(os.path.basename(stl_path))[0]
    out_filename = f"{base_name}_voxel.npz"
    out_path = os.path.join(output_dir, out_filename)
    np.savez_compressed(out_path, voxel=voxel_matrix)
    console.print(f"[INFO] Voxel matrix saved to: {out_path}", style="bold green")
    
    return voxel_matrix

# if __name__ == "__main__":
#     stl_file = '/home/tlmsq/boneking/dataset/1-025.stl'
#     output_directory = '/home/tlmsq/boneking/dataset_voxel/'
    
#     if not os.path.exists(output_directory):
#         os.makedirs(output_directory)
#     voxel_data = process_stl_to_voxel(stl_file, output_directory, vox_dim=60)
#     console.print(f"Voxel matrix shape: {voxel_data.shape}", style="bold cyan")
#     console.print(f"Total number of voxels containing the model (value 1) = {np.sum(voxel_data)}", style="bold cyan")
