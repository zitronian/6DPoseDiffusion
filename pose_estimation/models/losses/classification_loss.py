import torch.nn as nn
from pose_estimation.utils.utils import SO3_R3

'''
Legacy, not used in current 6D pose estimation method
'''

class ClassificationLoss:

    def __init__(self):
        pass

    def loss_fn(self, model, model_input, groundtruth, weights):

        H_in = SO3_R3(R=groundtruth[:, :3, :3], t=groundtruth[:, :3, -1]).log_map()


        xyz_points = model_input['scene_pcl']
        n_points = xyz_points.shape[1]

        xyz_points_flattened = xyz_points.reshape(-1, xyz_points.shape[-1])  # reshape to [B*n_points, 3]
        labels_flattened = model_input['scene_labels'].reshape(-1, 1)

        model.set_latent(scene_pcl=model_input['scene_pcl'],
                         obj_vertices=model_input['obj_vertices'],
                         obj_faces=model_input['obj_faces'],
                         H=H_in,
                         batch=n_points)

        y_pred = model.classify(xyz_points_flattened)

        loss_fn = nn.BCELoss()
        loss = loss_fn(y_pred, labels_flattened)

        return loss