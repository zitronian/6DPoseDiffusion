from torch.utils.data import Dataset
import glob
import os
import json
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import trimesh
import numpy as np
import torch
from pose_estimation.utils.image_utils import masked_pcl_from_depth_img, add_noise_to_mask
from pose_estimation.utils.utils import get_points_faces
from typing import List
import re
import pandas as pd
from pose_estimation.utils.utils import SO3_R3
from scipy.stats import gaussian_kde


class Scene:
    
    def __init__(self, block: str, rgb_path: str, depth_path: str, mask_paths: List[str], camera: dict, groundtruth: List[dict],
                 noisy_mask: bool, max_rounds_of_noise: int):
        """
        Class holding all relevant information of a scene. For formats of data please refer to
        https://github.com/thodan/bop_toolkit/blob/master/docs/bop_datasets_format.md (BOP challenge)

        :param block: string referencing the di
        :param rgb_path: path to rgb image of scene
        :param depth_path: path to depth image of scene
        :param mask_paths: list with paths to masks for this scene
        :param camera: camera intrinsics for this scene
        :param groundtruth: ground truth pose information for scene
        :param noisy_mask: if true, noise is added to mask
        :param max_rounds_of_noise: defines amount of noise that is added
        """

        self.block = block
        self.rgb_path = rgb_path
        self.depth_path = depth_path
        self.mask_paths = mask_paths
        self.camera = camera
        self.groundtruth_dict = self.__restructure_groundtruth(groundtruth)
        self.contained_objs = list(self.groundtruth_dict.keys()) # as ids
        self.noisy_mask = noisy_mask
        self.max_rounds_of_noise = max_rounds_of_noise

    def __restructure_groundtruth(self, groundtruth: List[dict]):
        """
        reates dict of all objects in scene with the object id (as string) as key.

        :param groundtruth: dict with ground truth information for scene
        :return: restructured ground truth dict
        """

        gt_dict = {}
        for i, obj_gt in enumerate(groundtruth):
            obj_id = obj_gt['obj_id']
            gt_dict[str(obj_id)] = {}
            gt_dict[str(obj_id)]['id_in_mask'] = i  # masks have form <scene_id>_<id_in_mask>
            for key, value in obj_gt.items():
                if key != 'obj_id':
                    gt_dict[str(obj_id)][key] = value
        return gt_dict
    
    def __get_intrinsics_and_depth(self):
        K = self.camera['cam_K']
        depth_scale = self.camera['depth_scale']
        intrinsics = {
            "cx": K[2],
            "cy": K[5],
            "fx": K[0],
            "fy": K[0],
            "height": 480,
            "width": 640
            }
        return intrinsics, depth_scale
    
    def pcl(self, obj_id, samples=1024, crop_image=True, crop_only_around_target=True):
        """
        Compute point cloud for scene.

        :param obj_id: id of the object we are interested in.
        :param samples: size of returned point cloud
        :param crop_image: legacy, always True
        :param crop_only_around_target: if true, point cloud is cropped based on segmentation mask of image
        :return: point cloud of size #samples x 3 as np.array
        """

        depth_im = np.asarray(Image.open(self.depth_path))
        intrinsics, depth_scale = self.__get_intrinsics_and_depth()

        # get mask_id for specific object in scene
        mask_id = '{id:06d}'.format(id=self.groundtruth_dict[str(obj_id)]['id_in_mask'])
        obj_mask_path = None
        for p in self.mask_paths:
            if re.split('[._]', p)[-2] == mask_id:
                obj_mask_path = p

        if obj_mask_path is None:
            FileNotFoundError()

        obj_mask = np.asarray(Image.open(obj_mask_path))
        if self.noisy_mask:
            obj_mask = add_noise_to_mask(obj_mask, max_rounds=self.max_rounds_of_noise)

        # create and optionally crop point cloud
        if crop_image:
            # create point cloud based on mask of object in image crop point cloud.
            if crop_only_around_target:
                pcl = masked_pcl_from_depth_img(depth_im, intrinsics, depth_scale, obj_mask)
            else:
                raise NotImplementedError
        else:
            pcl = masked_pcl_from_depth_img(depth_im, intrinsics, depth_scale, obj_mask=None)

        # sample points from point cloud.
        if len(pcl.shape) > 1:
            idxs = np.random.permutation(pcl.shape[0])[:samples]
            xyz = pcl[idxs]
        else:
            xyz = pcl

        # if not enough points, repeat random points from point cloud
        if xyz.shape[0] < samples:
            if xyz.shape[0] > 0:
                rand_idxs = np.random.randint(0, xyz.shape[0], samples-xyz.shape[0])
                xyz = np.concatenate([xyz, xyz[rand_idxs]])
            else:
                xyz = np.zeros((samples, 3))

        return xyz
    
    def gt(self, obj_id):
        """
        Get ground truth information for specific object in scene.

        :param obj_id: id of respective object of interest
        :return: returns groundtruth rotation and transformation as 4x4 np.array
        """

        R = np.array(self.groundtruth_dict[str(obj_id)]['cam_R_m2c']).reshape(3,3)
        t = np.array(self.groundtruth_dict[str(obj_id)]['cam_t_m2c']).reshape(3)
        T_homog = np.eye(4)
        T_homog[:3,:3] = R
        T_homog[:3,3] = t
        return T_homog
        

class ObjectModel:
    def __init__(self, path: str, obj_id: int, info: dict):
        """

        :param path: path to obejct model
        :param obj_id: id of object
        :param info: dict with model information like diameter etc.
        """
        self.path = path
        self.obj_id = obj_id
        self.info = info
        self.mesh = trimesh.load(self.path, file_type='ply', force='mesh')

    def pcl(self, samples=1024):
        """
        Compute object point cloud.

        :param samples: size of object point cloud
        :return: point cloud for object as np.array
        """
        # sample points from object 3D model
        pcl = self.mesh.sample(samples)
        pcl = np.array(pcl)
       
        return pcl


class CustomDataset(Dataset):
    
    def __init__(self, root_dir, sub_dir='train_pbr', obj_ids=[], center_scene=True, crop_around_obj=True, crop_around_target=False,
                 scene_n_points=1024, pc_scale=500, depth_postfix='png', rgb_postfix='jpg', mask_dir='mask', fraction=1.0,
                 n_sdf_query_points=10, noisy_mask=False, max_rounds_of_noise=8, fraction_seed=None, idx_txt_file=None):
        """
        Dataset class that follows the directory structure and conventions of the BOP challenge.
        https://github.com/thodan/bop_toolkit/blob/master/docs/bop_datasets_format.md


        :param root_dir: path to data directory
        :param sub_dir: name of sub directory in root_dir that holds
        :param obj_ids: list of obj ids to consider
        :param center_scene: if True, scene is centered
        :param crop_around_obj: legacy, always True
        :param crop_around_target: crop a
        :param scene_n_points: scene point cloud size
        :param pc_scale: factor to scale point cloud units with 1/pc_scale
        :param depth_postfix: postfix of depth image
        :param rgb_postfix: postfix of rgb image
        :param mask_dir: name of directory that holds masks
        :param fraction: in [0,1], allows to load only subset of data in present sub_dir
        :param n_sdf_query_points:
        :param noisy_mask: if True, noise is added to mask
        :param max_rounds_of_noise: number that specifies the extend of noise added
        :param fraction_seed: seed for selection fraction of dataset
        :param idx_txt_file: txt file specifying the indexes that are to be considdered in sub_dir
        """

        self.root_dir = root_dir
        self.obj_ids = [str(x) for x in obj_ids]  # need ids as str
        self.scenes = []
        self.object_dict = {}
        self.crop_around_obj = crop_around_obj
        self.crop_around_target = crop_around_target
        self.center_scene = center_scene
        self.scene_n_points = scene_n_points
        self.pc_scale=pc_scale
        self.depth_postfix = depth_postfix
        self.rgb_postfix = rgb_postfix
        self.mask_dir = mask_dir
        self.sub_dir = sub_dir
        self.object_sdf_dict = {}
        self.n_sdf_query_points = n_sdf_query_points
        
        
        f = open(os.path.join(root_dir, 'models', 'models_info.json'))
        models_info = json.load(f)
        
        ###### fill dictionary of ObjectModels
        for ply_path in glob.glob(os.path.join(root_dir, 'models', '*.ply')):
            obj_id = int(ply_path.split('/')[-1].split('_')[-1].split('.')[0]) # int
            info = models_info[str(obj_id)]
            self.object_dict[str(obj_id)] = ObjectModel(ply_path, obj_id, info)

            # fill sdf dict for object
            '''
            if obj_id in obj_ids or len(obj_ids) == 0:
                sdf_dict = np.load(os.path.join(root_dir, 'sdf_data', 'obj_{obj:06d}.npy'.format(obj=obj_id)),
                                   allow_pickle=True).item()
                self.object_sdf_dict[str(obj_id)] = sdf_dict
            '''
        ##########################################

        ###### fill list of Scenes
        sub_dir = os.path.join(root_dir,sub_dir)
        block_dirs = glob.glob(os.path.join(sub_dir,'*'))
        
        for block_dir in block_dirs:
            idxs=None
            if idx_txt_file is not None:
                with open(os.path.join(block_dir, idx_txt_file)) as file:
                    idxs = [int(line.rstrip()) for line in file]

            
            f = open(os.path.join(block_dir, 'scene_camera.json'))
            scene_camera_dict = json.load(f)
            
            f = open(os.path.join(block_dir, 'scene_gt.json'))
            scene_gt_dict = json.load(f)            
        
            depth_paths = glob.glob(os.path.join(block_dir, 'depth', f'*.{self.depth_postfix}'))

            rgb_paths = glob.glob(os.path.join(block_dir, 'rgb', f'*.{self.rgb_postfix}'))

            # sort paths by id so we can match them
            depth_paths.sort(key=get_number_from_path)
            rgb_paths.sort(key=get_number_from_path)

            for (depth_path, rgb_path) in list(zip(depth_paths, rgb_paths)):
                scene_id = get_number_from_path(depth_path)
                assert get_number_from_path(rgb_path) == scene_id  # need to be the same

                if idxs is not None:
                    if scene_id not in idxs:
                        continue

                # get visible masks paths for this scene (can be multiple paths if multiple objects are present)
                str_id = depth_path.split('/')[-1].split('.')[0]
                scene_mask_paths = glob.glob(os.path.join(block_dir, self.mask_dir, f'{str_id}_*.png'))

                camera = scene_camera_dict[str(scene_id)]
                gt = scene_gt_dict[str(scene_id)]
                
                self.scenes.append(Scene(block_dir.split('/')[-1], rgb_path, depth_path, scene_mask_paths, camera, gt,
                                         noisy_mask=noisy_mask, max_rounds_of_noise=max_rounds_of_noise))
        ##########################################

        data_identifier = self.__get_data_identifier()
        #self.data_identifier = self.__get_data_identifier()

        # only selection fraction of dataset
        if fraction_seed is None:
            idxs = np.arange(0, len(data_identifier))
        else:
            idxs = np.random.RandomState(fraction_seed).permutation(len(data_identifier))[:int(len(data_identifier) * fraction)]

        self.data_identifier = list(np.array(data_identifier)[idxs])

        # get weights for label distribution smoothing
        self.weights = self._prepare_weights()

    def __get_data_identifier(self):
        """

        :return: returns list of data for training/testing. Each scene contains multiple objects that
        should be used for training/testing. Function encodes scene/obj to <sceneID_objID>
        """

        data_identifier = []
        for scene_idx, scene in enumerate(self.scenes):
            rel_ids_in_scene = scene.contained_objs if len(self.obj_ids) == 0 \
                else list(set(scene.contained_objs) & set(self.obj_ids))
            for obj_id in rel_ids_in_scene:
                code = f'{str(scene.block)}_{str(scene_idx)}_{str(obj_id)}'
                data_identifier.append(code)
        return data_identifier
    
    def __decode_data_identifier(self, identifier: str):
        """
        Decodes indetifier (<sceneBlock_sceneID_objID>) into sceneBlock, sceneID and objID

        :param identifier: string
        :return: scene_block, scene_id, obj_id
        """

        parts = identifier.split('_')
        scene_block = int(parts[0])
        scene_id = int(parts[1])
        obj_id = int(parts[2])
        return scene_block, scene_id, obj_id

    def describe_scenes(self):
        '''

        :return: dataframe with ground truth translation and rotation for each scene.
        '''

        H_ws = np.empty((len(self.data_identifier), 6))
        obj_ids = np.empty(len(self.data_identifier), dtype=int)
        for i, identifier in enumerate(self.data_identifier):
            scene_block, scene_id, obj_id = self.__decode_data_identifier(identifier)
            scene = self.scenes[scene_id]
            H = torch.from_numpy(scene.gt(obj_id)[None,:]).float()

            H_ws[i] = SO3_R3(R=H[:, :3, :3], t=H[:, :3, -1]).log_map()[0].numpy()
            obj_ids[i] = obj_id

        H_ws_df = pd.DataFrame(H_ws, columns = ['x', 'y', 'z', 'alpha', 'beta', 'gamma'])
        obj_ids_df = pd.DataFrame(obj_ids, columns=['obj_id'])

        res_df = pd.concat([H_ws_df, obj_ids_df], axis=1)
        res_df['dataset'] = self.root_dir.split('/')[-1] + '-'+self.sub_dir
        return res_df

    def _prepare_weights(self):
        """
        Computes weight for each sample of dataset as scaled inverse KDE. High weight indicates that
        sample is underrepresented in dataset, low that it is overrepresented. Weights can be used for
        Label Distribution Smoothing.

        :return: (np.array) weight for each sample of dataset
        """

        scenes_df = self.describe_scenes()
        H_ws = scenes_df[['x', 'y', 'z', 'alpha', 'beta', 'gamma']].to_numpy()
        kde = gaussian_kde(H_ws.T)
        density = kde(H_ws.T)
        weights = density**-1
        scaling = len(weights)/weights.sum()
        scaled_weights = scaling * weights

        return scaled_weights

    def __len__(self):
        return len(self.data_identifier)
    
    def __getitem__(self, index):
        """

        :param index:
        :return:
        """

        identifier = self.data_identifier[index]
        scene_block, scene_id, obj_id = self.__decode_data_identifier(identifier)
        scene = self.scenes[scene_id]
        obj = self.object_dict[str(obj_id)]
        
        # get scene as pcd
        scene_pcl = scene.pcl(obj_id=obj.obj_id, samples=self.scene_n_points, crop_image=self.crop_around_obj,
                                            crop_only_around_target=self.crop_around_target)

        # get gt rotation and translation for object in scene in homogoneous coordinates
        H = scene.gt(obj_id)
        
        # get object model as pointcloud
        object_mesh = obj.mesh.copy()

        # scale object mesh with scale
        scale_matrix = np.eye(4)
        scale_matrix[:3, :3] /= self.pc_scale
        object_mesh.apply_transform(scale_matrix)

        # scale point clouds and gt translation
        scene_pcl = scene_pcl / self.pc_scale
        H[:3, -1] = H[:3, -1]/self.pc_scale

        # centeres pc by substracting mean. Adjusts gt translation by substracting mean
        mean = 0
        if self.center_scene:
            # center point clouds and gt translation
            if len(scene_pcl.shape)>1:
                mean = scene_pcl.mean(axis=0)
                scene_pcl = scene_pcl-mean
            else:
                mean = np.zeros(3)
            H[:3, -1] = H[:3, -1]-mean


        # get face centroids as points and corresponding face normals
        obj_points, obj_points_normals = get_points_faces(object_mesh, n_points=10000)

        # get sdf data
        '''
        sdf_dict = self.object_sdf_dict[str(obj_id)]
        sdf_idxs = np.random.permutation(sdf_dict['sdf'].shape[0])[:self.n_sdf_query_points]
        sdf_query_points = sdf_dict['query_points'][sdf_idxs,:]
        sdf = sdf_dict['sdf'][sdf_idxs]
        '''

        res = {
            'scene_pcl': torch.from_numpy(scene_pcl).float(),
             # 'scene_labels': torch.from_numpy(scene_labels).float(),
            'obj_points': torch.from_numpy(obj_points).float(),
            'obj_point_normals': torch.from_numpy(obj_points_normals).float(),
            'obj_id': torch.tensor([obj.obj_id]).int(),
            'scene_id': torch.tensor([scene_id]).int(),
            'block_id': torch.tensor([scene_block]).int(),
            'scene_mean': torch.tensor(mean).float(),
            #'sdf_query_points': torch.tensor(sdf_query_points).float(),
            #'sdf': torch.tensor(sdf).float()
        }

        H = torch.from_numpy(H).float()

        weight = self.weights[index].astype(np.float32)
        
        return res, H, weight


def get_number_from_path(path: str):
    """

    :param path: string
    :return: returns ID of file as integer
    """

    return int(path.split('/')[-1].split('.')[0])


if __name__ == '__main__':

    pass