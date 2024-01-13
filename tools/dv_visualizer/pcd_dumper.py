
#因为open3d依赖的lib和bev依赖的lib有版本冲突，dumper需要在open3d的env中运行
#conda activate open3d
def _pcd_converter(data_path):
    import open3d as o3d
    import numpy as np
    point_cloud = o3d.io.read_point_cloud(data_path)
    pcd_arr = np.asarray(point_cloud.points)
    np.save("13_arr.npy",pcd_arr)
    # open3d_point_cloud = o3d.geometry.PointCloud()
    # open3d_point_cloud.points = o3d.utility.Vector3dVector(pcd_arr)

    # print(pcd_arr.shape)
    # # 创建窗口并显示俯视图
    # o3d.visualization.draw_geometries(geometry_list=[point_cloud])
    return pcd_arr
def _dump_npy_to_bin(data_path):
    import numpy as np
    pcd_data = np.load(data_path)
    # 保存为二进制文件
    pcd_data.tofile("output_binary.bin")

if __name__ == "__main__":
    data_path="tools/dv_visualizer/gt/13.pcd"
    npy_data_path = "tools/dv_visualizer/gt_npy/0_arr.npy"
    # pcd_np = _pcd_converter(data_path=data_path)
    _dump_npy_to_bin(npy_data_path)
