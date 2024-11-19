import os
from pose_estimation.datasets.generate_sdf_data import generate_sdf_data
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'pose_estimation'))

# for running headless:
# os.environ['PYOPENGL_PLATFORM'] = 'egl'

def generate_sdf(objs, n_points):
    """
    Generate SDF training data for specific object of lm dataset
    :param objs:
    :param n_points:
    :return:
    """
    scale = 500
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_root = os.path.join(base_path, 'bop_data', 'lm')
    generate_sdf_data(scale, dataset_root, objs=objs, n_points=n_points)


if __name__ == '__main__':

    generate_sdf(objs=[8], n_points=200000)