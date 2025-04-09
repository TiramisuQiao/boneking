from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt
from stl import mesh


filename = '/home/tlmsq/boneking/dataset/1-025.stl'
your_mesh = mesh.Mesh.from_file(filename)
figure = plt.figure()
axes = figure.add_subplot(111, projection='3d')
axes.add_collection3d(mplot3d.art3d.Poly3DCollection(your_mesh.vectors, color='lightgrey'))
scale = your_mesh.points.flatten()
axes.auto_scale_xyz(scale, scale, scale)

plt.show()