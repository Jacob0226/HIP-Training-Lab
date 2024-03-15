import open3d as o3d
import numpy as np

print("Load a ply point cloud, print it, and render it")
ply_point_cloud = o3d.data.PLYPointCloud()
pcd = o3d.io.read_point_cloud(ply_point_cloud.path)
o3d.visualization.draw_geometries([pcd],
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])
o3d.io.write_point_cloud("sample.ply", pcd, write_ascii=True)
np_vertex = np.asarray(pcd.points)
np.savetxt('vertex.txt', np_vertex)