import trimesh
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  

stl_mesh = trimesh.load('/home/tlmsq/boneking/dataset/1-025.stl')
bbox_min, bbox_max = stl_mesh.bounds
extent = bbox_max - bbox_min  
scale_factor = 3.0 / max(extent)  
stl_mesh.apply_translation(-bbox_min)  
stl_mesh.apply_scale(scale_factor)    
pitch = 3.0 / 60  
voxelized = stl_mesh.voxelized(pitch=pitch)  
voxel_matrix = voxelized.matrix.astype(np.uint8)
print("Voxel matrix shape:", voxel_matrix.shape)

vox_dim = 60
vox_size = 3.0 / vox_dim   


active_indices = np.argwhere(voxel_matrix == 1)


active_coords = active_indices.astype(np.float32) * vox_size + (vox_size / 2)

x_coords = active_coords[:, 0]
y_coords = active_coords[:, 1]
z_coords = active_coords[:, 2]


fig = plt.figure(figsize=(22, 12))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x_coords, y_coords, z_coords, c='blue', marker='o', s=5, alpha=0.6)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.set_xlim(0, 3)
ax.set_ylim(0, 3)
ax.set_zlim(0, 3)
plt.title("3D Voxel-based ")
plt.show()
