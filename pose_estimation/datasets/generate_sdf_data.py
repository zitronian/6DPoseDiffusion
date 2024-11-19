import os
import trimesh
import numpy as np
from pose_estimation.utils.utils import sample_sdf_from_mesh
import tqdm

def generate_sdf_data(scale, root_dir, objs=[], n_points=20000):
    """ Function that creates sdf data for a specific object. Stores SDF data as .npy file.

    :param scale: object is scaled by 1/scale
    :param root_dir: path to directory that holds models directory with 3D models of objects
    :param objs: ids of object to create sdf data for
    :param n_points: number of of points with respective sdf -> size of sdf dataset for object
    """
    target_dir = os.path.join(root_dir, 'sdf_data')

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    # load

    for obj_id in tqdm.tqdm(objs):
        ply_file = os.path.join(root_dir, 'models', 'obj_{obj:06d}.ply'.format(obj=obj_id))
        obj_mesh = trimesh.load(ply_file, file_type='ply', force='mesh')

        # scale object mesh with scale
        scale_matrix = np.eye(4)
        scale_matrix[:3, :3] /= scale
        obj_mesh.apply_transform(scale_matrix)

        sdf_query_points, sdf = sample_sdf_from_mesh(obj_mesh, n_points=n_points)

        sdf_dict = {
            'query_points': sdf_query_points,
            'sdf': sdf
        }
        target_file = os.path.join(target_dir, 'obj_{obj:06d}'.format(obj=obj_id))
        np.save(target_file, sdf_dict)


if __name__ == '__main__':

    pass
