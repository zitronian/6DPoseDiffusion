from visualizations.pose_visualization import generate_gif, get_scene_results, create_object_dict, load_scene_as_pcl, \
    trimesh_pcl, generate_pose_trajectory_images
import os
import glob
import numpy as np


def create_gif(store_image_path, block, scene_id, obj_id, res_path, root_path, subdir, rotate_gif=True):
    '''
    Results if inference process is used to render images and create GIF.

    :param store_image_path: folder to store images and giv
    :param scene_id: id of relevant scene. same ids as in dataset folder
    :param obj_id: id of relevant object within the scene
    :param res_path: path to the evaluation results. The results contain history of poses that is being used
    to render images for each iteration of inference and subsequently creates giv.
    :param root_path: path to the dataset
    :param subdir:
    :return:
    '''
    ### generate images
    if not os.path.exists(store_image_path):
        os.makedirs(store_image_path)

    dataset_base_path = os.path.join(root_path, subdir)
    res_path = glob.glob(os.path.join(res_path, '**/*.npy'), recursive=True)
    data = np.load(res_path[0], allow_pickle=True).item()
    id = scene_id  # scene id
    scale = 500

    scene_results = get_scene_results(data, id)
    obj_dict = create_object_dict(root_path)
    scene = load_scene_as_pcl(id, dataset_base_path, block)
    obj = obj_dict[str(obj_id)]

    scene_pcl = scene.pcl(obj_id, 2000, crop_image=True, crop_only_around_target=False)
    scene_pcl = trimesh_pcl(scene_pcl, [100, 100, 100])

    generate_pose_trajectory_images(scene_pcl, obj, scene_results, store_image_path, scale, rotate_gif=rotate_gif)

    ### generate GIF
    generate_gif(store_image_path)


if __name__ == '__main__':
    scene_id = 87
    obj_id = 8
    block = '000000'
    name = 'soft-jazz-1-{}-{}fixcamera'.format(scene_id, obj_id)


    results_path = 'pose_estimation/results/100Iterations_soft-jazz-377-1'
    subdir = 'test_pbr_2obj/000000'
    root_path = 'pose_estimation/bop_data_generated/lm'

    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    store_image_path = os.path.join(base_path, 'images', name)
    create_gif(store_image_path, block, scene_id, obj_id, results_path, root_path, subdir, rotate_gif=False)
