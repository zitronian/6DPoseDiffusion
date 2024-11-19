import torch.nn as nn
import torch
from pose_estimation.utils.transformation_utils import transformed_obj_front_facing_points
from pose_estimation.utils.utils import SO3_R3, transform_points_batch, SceneObjDescriptor


class PoseModel(nn.Module):

    def __init__(self,
                 scene_encoder,
                 obj_encoder,
                 feature_encoder,
                 score_net,
                 marginal_prob_std,
                 distance_net,
                 sdf_net,
                 device='cpu',
                 center_obj=False,
                 only_front_facing_points=False,
                 scale_score=True,
                 front_facing_noise=False,
                 feed_pose_explicitly=True,
                 attention_net=None,
                 orient_latent=False,
                 geometric_latent=True,
                 scene_obj_descriptor=SceneObjDescriptor.NONE.value):
        """

        :param scene_encoder: encoder module used to encode the scene point cloud
        :param obj_encoder: encoder module used to encode object.
        :param feature_encoder: network to extract features from latent
        :param score_net: Decoder network
        :param marginal_prob_std:
        :param distance_net: Legacy
        :param sdf_net: Legacy
        :param device: cuda, cpu
        :param center_obj: if True, object point cloud is centered
        :param only_front_facing_points: if True, only front facing points of object point cloud are encoded given
        specific transformation
        :param scale_score: if True, score is scaled bei std
        :param front_facing_noise: if True, the threshold to determine front facing of point is sampled from gaussian
        :param feed_pose_explicitly: if True, pose is explicitly fed into the feature encoder
        :param attention_net: if True, attention network is deployed after latent
        :param orient_latent: if True, scene and object latent are transformed with inverse of transformation
        :param geometric_latent: True, if vector neurons are used and latent is three-dimensional.
        :param scene_obj_descriptor: Descriptor used to capture relation between object and scene latent.
        """

        super().__init__()
        self.only_front_facing_points = only_front_facing_points
        self.marginal_prob_std = marginal_prob_std
        self.scale_score = scale_score
        self.front_facing_noise = front_facing_noise
        self.feed_pose_explicitly = feed_pose_explicitly
        self.center_obj = center_obj
        self.scene_obj_descriptor = scene_obj_descriptor
        self.attention_net = attention_net
        self.orient_latent = orient_latent
        self.geometric_latent = geometric_latent
        self.device = device
        self.z = None
        self.scene_latent=None
        self.obj_latent=None
        self.H = None

        # to encode scene point cloud
        self.scene_encoder = scene_encoder

        # to encode object point cloud
        self.obj_encoder = obj_encoder

        # feature encoder
        self.feature_encoder = feature_encoder

        # to compute energy/score
        self.score_net = score_net

        # for classification task
        #self.classification_decoder = classification_decoder

        self.distance_net = distance_net

        # Legacy, not used anymore
        self.sdf_qery_point_emb_net = nn.Sequential(
            nn.Linear(3, 24),
            nn.ReLU(),
        )

        self.sdf_net = sdf_net

    def set_scene_latent(self, scene_pcl, batch=1):
        """
        Feeds scene point cloud through scene encoder network and sets scene latent.

        :param scene_pcl:
        :param batch: scene latent is expanded to fit batch size
        :return:
        """

        scene_latent = self.scene_encoder(scene_pcl)

        # if geometric, ensure that latent is three dimensional after expanding
        if self.geometric_latent:
            self.scene_latent = scene_latent.unsqueeze(1).repeat(1,batch,1,1).reshape(-1, scene_latent.shape[1], 3)
        else:
            self.scene_latent = scene_latent.unsqueeze(1).repeat(1, batch, 1, 1).reshape(-1, scene_latent.shape[1])

    def set_obj_latent(self, obj_points, obj_point_normals, H, obj_pcl=None, batch=1):
        """
        Partially renders object point cloud and feeds it throhg encoder to set object latent.

        :param obj_points:
        :param obj_point_normals:
        :param H: transformation matrices
        :param obj_pcl: if provided, object point cloud is not partially rendered.
        :param batch: if each pose is pertubed multiple times, then object is encoded once and then latent is extended
        :return:
        """

        if obj_pcl is None:
            # apply partial rendering
            obj_pcl = transformed_obj_front_facing_points(obj_points,
                                                          obj_point_normals,
                                                          SO3_R3().exp_map(H).to_matrix().detach(),
                                                          only_front_facing_points=self.only_front_facing_points,
                                                          with_noise=self.front_facing_noise)

        # center object point cloud
        if self.center_obj and self.geometric_latent:
            # remove mean so that latent output of obj_encoder is SO(3) equivariant
            pcl_mean = obj_pcl.mean(dim=1)
            obj_pcl = obj_pcl - pcl_mean.unsqueeze(1).expand_as(obj_pcl)

        # embed object, if obj_encoder is none use same encoder as for scene
        if self.obj_encoder is not None:
            obj_emb = self.obj_encoder(obj_pcl)
        else:
            obj_emb = self.scene_encoder(obj_pcl)

        # if object point cloud was centered, reverse operation in latent space
        if self.center_obj and self.geometric_latent:
            # add mean such that resulting latent is SE(3) equivarient
            obj_emb = (obj_emb + pcl_mean.unsqueeze(1).expand_as(obj_emb))


        self.H = H.unsqueeze(1).repeat(1,batch,1).reshape(-1, 6) #H

        # expand according to batch
        if self.geometric_latent:
            self.obj_latent = obj_emb.unsqueeze(1).repeat(1, batch, 1, 1).reshape(-1, obj_emb.shape[1], 3)
        else:
            self.obj_latent = obj_emb.unsqueeze(1).repeat(1, batch, 1, 1).reshape(-1, obj_emb.shape[1])

    def transform_obj_latent(self, H, relative=True):
        """ Transforms object latent with transformation H (R6) and sets transformation H.

        :param H: B x 1 x 6
        :param relative: if True, H is given in relation to the current latent. As the matrix provided to the feature
        encoder describes the pose of the model in relation to the canonical pose of the obj model when fed into the
        pcl encoder, H needs to be updated accordingly (by composition
        """

        # transform according to H
        self.obj_latent = transform_points_batch(self.obj_latent, SO3_R3().exp_map(H).to_matrix().detach())

        # set H
        # if H is relative to current latent (and not relative to the model input pose)
        # H is added to the current Transformation H.
        if relative:
            self.H += H
        else:
            self.H = H


    def forward(self, time_t=None):
        """

        :param time_t: time step t to condition the model on
        :return: derivative of energy function in respect to H (R6 vector (lie algebra))
        """

        # apply inverse H to scene and object latent
        if self.orient_latent:

            inv_H = SO3_R3().exp_map(self.H).get_inverse()
            self.obj_latent = transform_points_batch(self.obj_latent, inv_H)
            self.scene_latent = transform_points_batch(self.scene_latent, inv_H)

        # use object and scene latent directly to feed into attention network or feature encoder
        if self.scene_obj_descriptor == SceneObjDescriptor.NONE.value:

            if self.attention_net is not None:
                scene_latent_tagged = torch.cat(
                    [self.scene_latent, self.scene_latent.new_ones((*self.scene_latent.size()[:-1], 1))], dim=-1)
                obj_latent_tagged = torch.cat(
                    [self.obj_latent, self.obj_latent.new_zeros((*self.obj_latent.size()[:-1], 1))], dim=-1)
                att_input = torch.cat([scene_latent_tagged, obj_latent_tagged], dim=1)
                attended_scene_obj_latent = self.attention_net(att_input, mask=None)
                scene_obj_descriptor = attended_scene_obj_latent.flatten(start_dim = 1)

            else:
                scene_obj_descriptor = torch.cat([self.scene_latent, self.obj_latent], dim=1)

        # compute pairwise distance between object and scene latent
        elif self.scene_obj_descriptor == SceneObjDescriptor.PAIRWISE_DIST.value:
            scene_obj_descriptor = torch.cdist(self.scene_latent, self.obj_latent)

        # compute distance between object and scene latent
        elif self.scene_obj_descriptor == SceneObjDescriptor.DIST.value:
            scene_obj_descriptor = torch.linalg.norm(self.scene_latent-self.obj_latent, dim=2)

        else:
            raise NotImplementedError

        scene_obj_descriptor = scene_obj_descriptor.flatten(start_dim=1)
        if self.feed_pose_explicitly:
            latent = torch.cat([scene_obj_descriptor, self.H], dim=-1)
        else:
            latent = scene_obj_descriptor

        # create feature
        emb = self.feature_encoder(latent, time_t)

        # get score
        score = self.score_net(emb)

        # scale
        if self.scale_score:
            score = score / self.marginal_prob_std(time_t)[:, None]

        return score


    def compute_sdf(self, obj_pcl, query_points):
        """ Branch that computes SDF. Legacy, not used anymore

        :param obj_pcl: B x n_points x 3, points
        :param query_points: points to query

        :return: sdf for query points
        """

        # embed pcl
        if self.obj_encoder is not None:
            obj_emb = self.obj_encoder(obj_pcl)
        else:
            obj_emb = self.scene_encoder(obj_pcl)

        obj_emb_repeated = obj_emb.unsqueeze(1).repeat(1, query_points.shape[1], 1, 1).reshape(-1, obj_emb.shape[1], 3)
        obj_emb_repeated = obj_emb_repeated.flatten(start_dim=1)

        # encode query points
        query_point_emb = self.sdf_qery_point_emb_net(query_points).reshape(query_points.shape[0]*query_points.shape[1], -1)

        x = torch.cat([obj_emb_repeated, query_point_emb], dim=1)

        # decode to sdf
        sdf = self.sdf_net(x)

        return sdf


if __name__ == '__main__':

    pass
