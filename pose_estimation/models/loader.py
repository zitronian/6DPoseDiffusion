from pose_estimation.models import SimplePointnet, TimeDependentFeatureEncoder, \
    ScoreNet, PoseModel, DistanceNet, SingleChannelVNTResnetWrapper, SingleChannelVNNResnetWrapper, SingleChannelVNTSimpleWrapper, \
    SingleChannelVNTDistangledWrapper, VNN_ResnetPointnet, SDFNet, AttentionEncoderLayer, ResnetPointnet
from pose_estimation.utils.utils import PclEncoder, SceneObjEncoderMode, SceneObjDescriptor
import torch
import numpy as np


def marginal_prob_std(t, sigma=0.5):
    """
    Compute standard deviation of t-pertubed given unpertubed sample x $p_{0t}(x(t) | x(0))$.

    :param t: A vector of time steps
    :param sigma: Sigma in SDE ()default 0.5
    :return: Standard deviation
    """

    return torch.sqrt((sigma ** (2 * t) - 1.) / (2. * np.log(sigma)))


def load_model(pcl_encoder: PclEncoder.POINTNET,
               z_dim=192,
               noise_embed_dim=128,
               pose_dim=6,
               feature_emb_dim=256,
               feature_encoder_layer_dim=[512, 512],
               dropout=[0, 1],
               dropout_prob=0.2,
               k=20,
               distance_awarness_n_points=128,
               only_front_facing_points=True,
               center_obj=False,
               model_path=None,
               scene_object_encoder_mode=SceneObjEncoderMode.DISTINCT_ENCODER,
               scene_obj_descriptor=SceneObjDescriptor.NONE,
               attention_encoding = None,
               front_facing_noise=True,
               feed_pose_explicitly = True,
               orient_latent=False,
               device='cpu'):
    """
    Instantiated model with provided configuration.

    :param pcl_encoder:
    :param z_dim: Latent size for point cloud. Scene latent has size z_dim and object latent has size z_dim (if
    flattened in the case of vector neurons)
    :param noise_embed_dim: size of embedding for time step t
    :param pose_dim: dimension of pose, usually 6 dimensions
    :param feature_emb_dim: size of feature embedding
    :param feature_encoder_layer_dim: number of layers in feature encoder and #neurons of each layer
    :param dropout:
    :param dropout_prob:
    :param k: number of nearest neighbour for point cloud encoder.
    :param distance_awarness_n_points:
    :param only_front_facing_points: if True, only front facing points of transformed object in scene are considered
    :param center_obj: if True, object point cloud is centered
    :param model_path: if provided, model weights are loaded
    :param scene_object_encoder_mode: if True, scene and object are encoded using the same encoder
    :param scene_obj_descriptor: Descriptor used to capture relation between object and scene latent.
    :param attention_encoding: if True, attention net is applied to scene and object latent
    :param front_facing_noise: if True, the threshold to determine front facing of point is sampled from gaussian
    :param feed_pose_explicitly: if True, pose is explicitly fed into the feature encoder
    :param orient_latent: if True, scene and object latent are transformed with inverse of transformation
    :param device: cuda, cpu
    :return: instantiated pose estimation model
    """

    if scene_obj_descriptor == SceneObjDescriptor.NONE.value:
        embedded_pcl_dim = 2 * z_dim
    elif scene_obj_descriptor == SceneObjDescriptor.PAIRWISE_DIST.value:
        embedded_pcl_dim = (z_dim//3) ** 2
    elif scene_obj_descriptor == SceneObjDescriptor.DIST.value:
        embedded_pcl_dim = (z_dim//3)
    else:
        raise NotImplementedError

    if attention_encoding:
        embedded_pcl_dim = (2 * z_dim) + (2 * (z_dim//3)) # account for 4th dimension in pcl embedding

    score_model_out_dim = pose_dim
    feature_encoder_in_dim = embedded_pcl_dim + pose_dim if feed_pose_explicitly else embedded_pcl_dim
    geometric_latent = True

    if pcl_encoder == PclEncoder.POINTNET.value:
        geometric_latent=False
        scene_encoder = SimplePointnet(c_dim=z_dim, dim=3)
        if scene_object_encoder_mode == SceneObjEncoderMode.SHARED_ENCODER.value:
            # triggers model to use same encoder for object and scene
            obj_encoder = None
        elif scene_object_encoder_mode == SceneObjEncoderMode.DISTINCT_ENCODER.value:
            # use distinct encoders for object and scene
            obj_encoder = SimplePointnet(c_dim=z_dim, dim=3)
        else:
            raise NotImplementedError

    elif pcl_encoder == PclEncoder.VNNResnetPointnet.value:
        scene_encoder = VNN_ResnetPointnet(c_dim=int(z_dim / 3), device=device)
        if scene_object_encoder_mode == SceneObjEncoderMode.SHARED_ENCODER.value:
            # triggers model to use same encoder for object and scene
            obj_encoder = None
        elif scene_object_encoder_mode == SceneObjEncoderMode.DISTINCT_ENCODER.value:
            # use distinct encoders for object and scene
            obj_encoder = VNN_ResnetPointnet(c_dim=int(z_dim / 3), device=device)
        else:
            raise NotImplementedError

    elif pcl_encoder == PclEncoder.ResnetPointnet.value:
        geometric_latent=False
        scene_encoder = ResnetPointnet(c_dim=z_dim, device=device)
        if scene_object_encoder_mode == SceneObjEncoderMode.SHARED_ENCODER.value:
            # triggers model to use same encoder for object and scene
            obj_encoder = None
        elif scene_object_encoder_mode == SceneObjEncoderMode.DISTINCT_ENCODER.value:
            # use distinct encoders for object and scene
            obj_encoder = ResnetPointnet(c_dim=z_dim, device=device)
        else:
            raise NotImplementedError


    elif pcl_encoder == PclEncoder.VNTResnetPointnet.value:
        scene_encoder = SingleChannelVNTResnetWrapper(z_dim=z_dim, device=device, k=k)
        if scene_object_encoder_mode == SceneObjEncoderMode.SHARED_ENCODER.value:
            # triggers model to use same encoder for object and scene
            obj_encoder = None
        elif scene_object_encoder_mode == SceneObjEncoderMode.DISTINCT_ENCODER.value:
            # use distinct encoders for object and scene
            obj_encoder = SingleChannelVNTResnetWrapper(z_dim=z_dim, device=device, k=k)
        else:
            raise NotImplementedError

    elif pcl_encoder == PclEncoder.VNTSimplePointnet.value:
        scene_encoder = SingleChannelVNTSimpleWrapper(z_dim=z_dim, device=device, k=k)
        if scene_object_encoder_mode == SceneObjEncoderMode.SHARED_ENCODER.value:
            # triggers model to use same encoder for object and scene
            obj_encoder = None
        elif scene_object_encoder_mode == SceneObjEncoderMode.DISTINCT_ENCODER.value:
            # use distinct encoders for object and scene
            obj_encoder = SingleChannelVNTSimpleWrapper(z_dim=z_dim, device=device, k=k)
        else:
            raise NotImplementedError

    elif pcl_encoder == PclEncoder.VNTDistEncoder.value:
        scene_encoder = SingleChannelVNTDistangledWrapper(z_dim=z_dim, device=device, k=k)
        if scene_object_encoder_mode == SceneObjEncoderMode.SHARED_ENCODER.value:
            # triggers model to use same encoder for object and scene
            obj_encoder = None
        elif scene_object_encoder_mode == SceneObjEncoderMode.DISTINCT_ENCODER.value:
            # use distinct encoders for object and scene
            obj_encoder = SingleChannelVNTDistangledWrapper(z_dim=z_dim, device=device, k=k)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    feature_encoder = TimeDependentFeatureEncoder(time_embed_dim=noise_embed_dim, in_dim=feature_encoder_in_dim,
                                                  dropout=dropout, dropout_prob=dropout_prob,
                                                  layer_dims=feature_encoder_layer_dim, out_dim=feature_emb_dim)

    score_net = ScoreNet(feature_emb_dim, out_dim=score_model_out_dim)

    disance_net = DistanceNet(in_dim=feature_emb_dim, hidden_dim=256, out_dim=distance_awarness_n_points)

    sdf_net = SDFNet(in_dim=z_dim+24)

    if attention_encoding:
        attention_net = AttentionEncoderLayer(d_model=4, n_heads=2, d_ff=16)
    else:
        attention_net = None

    pose_model = PoseModel(scene_encoder, obj_encoder, feature_encoder=feature_encoder, score_net=score_net, device=device,
                           distance_net=disance_net, marginal_prob_std=marginal_prob_std,
                           only_front_facing_points=only_front_facing_points, center_obj=center_obj,
                           feed_pose_explicitly=feed_pose_explicitly, sdf_net=sdf_net, attention_net=attention_net,
                           scene_obj_descriptor=scene_obj_descriptor, front_facing_noise=front_facing_noise, geometric_latent=geometric_latent,
                           orient_latent=orient_latent)

    # load weights if path is provided
    if model_path is not None:
        stored_model = torch.load(
            model_path,
            map_location=device)
        pose_model.load_state_dict(stored_model['model_state_dict'])
        pose_model.to(device)
        pose_model.eval()

    return pose_model