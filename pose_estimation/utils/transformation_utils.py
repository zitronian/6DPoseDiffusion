import torch
from pose_estimation.utils.utils import SO3_R3, transform_points_batch
from theseus import SO3
import theseus as th
import numpy as np
import trimesh
import multiprocessing
from functools import partial


def pertub_H(H_in, time_dependent_std_func, eps=1e-5):
    """
    Perturb transformation matrices.

    :param H_in: np.array of poses (batch_size x 4 x 4 transformation matrix) that are to be perturbed
    :param time_dependent_std_func: function to get std for time step t
    :param eps: minimum time step (for computational stability)
    :return: perturbed transformation in batch_size x 6 x 1 (lie algebra)
    """

    H_original = H_in.reshape(-1, 4, 4)
    H_original = SO3_R3(R=H_original[:, :3, :3], t=H_original[:, :3, -1])
    H_gt_lg = H_original.log_map()  # lie group

    random_t = torch.rand_like(H_gt_lg[..., 0], device=H_gt_lg.device) * (1. - eps) + eps
    z = torch.randn_like(H_gt_lg)
    std = time_dependent_std_func(random_t)
    noise = z * std.reshape(-1, 1)

    noise_rot = SO3.exp_map(noise[:, 3:])  # R3 rotational noise as lie algebra
    noise_t = noise[:, :3]  # translation noise as vector

    # Compose original rotation and noise rotation through multiplication
    pertubed_R = th.compose(H_original.R, noise_rot)  # compose both as matrix multiplication (result lie group)
    pertubed_t = H_original.t + noise_t

    H_hat_lg = torch.cat((pertubed_t, pertubed_R.log_map()), -1)  # R6 lie algebra

    return H_hat_lg, random_t, z, std


def compute_front_facing_points(points, normals, n_samples=1024, with_noise=False):
    """ Compute front facing points of point cloud given normal vectors.

    :param points: (tensor)
    :param normals:  (tensor)
    :param n_samples: number of samples to get
    :param with_noise: if True, threshold to determine if point is front facing is sampled from gaussian
    :return: point cloud containing only front facing points
    """

    # compute which points are front facing. set z-axis viewpoint to -100000 as hack to always have entire object in
    # viewpoint
    startpoint = torch.tensor([0, 0, -100000], device=points.device)
    directional_dist = startpoint - points

    # is cos(angle)
    dots = torch.einsum("bij,bij->bi", directional_dist, normals) / torch.linalg.norm(directional_dist, axis=2)

    if not with_noise:
        front_facing_mask = dots > 1e-5
    else:
        front_facing_mask = dots > np.random.normal(loc=0, size=1, scale=0.1).clip(-0.3, 0.3)[0]

    arr = []
    for pset, front_mask in zip(points, front_facing_mask):
        pts = pset[front_mask][:n_samples]

        # if not enough points front facing, use some twice in pcl
        if pts.shape[0] < n_samples:
            rand_idxs = torch.randint(low=0, high=pts.shape[0], size=(n_samples-pts.shape[0],))
            pts = torch.cat([pts, pts[rand_idxs]])

        arr.append(pts)

    return torch.stack(arr)

def transformed_obj_front_facing_points(points, normals, H, only_front_facing_points=True, n_points=1024, with_noise=False):
    """
    Renders object point cloud into scene with transformation H.

    :param points:
    :param normals:
    :param H:
    :param only_front_facing_points: if True, object is only partially rendern i.e. only front facing points
    :param n_points:
    :param with_noise: if True, threshold to determine if point is front facing is sampled from gaussian
    :return:
    """

    transformed_points = transform_points_batch(points, H.float())
    if only_front_facing_points:
        H_rot = H
        H_rot[:,:3,-1] = 0
        # we only care about front facing and thus can only apply rotation to normals
        transformed_normals = transform_points_batch(normals, H_rot.float())
        obj_points = compute_front_facing_points(transformed_points, transformed_normals, n_samples=n_points,
                                                              with_noise=with_noise)
    else:
        obj_points = transformed_points[:, :n_points, :]

    return obj_points

def get_front_facing_mesh(mesh, rand_rot_noise=None, eps=1e-5, startpoint=np.array([0, 0, 0])):
    ''' Returns subset of mash with only front facing faces (of the
    perspective of startpoint). Fast computation with matrix multiplication.
    See: https://github.com/mikedh/trimesh/issues/883

    :param mesh:
    :param startpoint:
    :return: mesh with faces that are only front facing.
    '''
    mesh = mesh.copy()
    #
    # startpoint = np.array([0,50,0]) # starting point of ray
    directional_dist = startpoint - mesh.triangles_center
    dots = np.einsum("ij,ij->i", directional_dist, mesh.face_normals) / np.linalg.norm(directional_dist, axis=1)
    if rand_rot_noise is not None:
        cos_theta = dots / np.linalg.norm(mesh.face_normals, axis=1)
        # angle is unsigned between [0, 180] degree
        unsigned_angle = np.arccos(cos_theta) * (180 / np.pi)
        # compute normal vector to plane of face normals and directional vector
        cross = np.cross(directional_dist, mesh.face_normals)
        # see if face normal vector and normal vector to plane face in the same direction (dot product > 0) or
        # opposite direction (dot product < 0)
        sign = np.einsum('bs,bs->b', cross, mesh.face_normals)
        # signed angle [-180, 180] degrees
        angle = unsigned_angle
        angle[sign < 0] = -unsigned_angle[sign < 0]

        front_facing = (rand_rot_noise[0] < unsigned_angle) & (unsigned_angle < rand_rot_noise[1])
    else:
        eps = np.random.normal(loc=0, size=1, scale=0.1).clip(-0.3, 0.3)[0]
        front_facing = dots > 1e-5#eps#1e-5  # if dot product is positive, front facing face
    mesh.update_faces(front_facing)

    return mesh


def get_visible_mesh(in_mesh, startpoint=np.array([0, 0, 0]), max_rays=10):
    ''' Sends rays from startpoint to each face of in_mesh and notes idx
    of face that ray its first. Only

    :param in_mesh:
    :param startpoint: perspective to start rays of
    :param n: number of faces to check
    :return: subset of in_mesh with only visible faces from startpoint
    '''
    mesh = in_mesh.copy()
    ray = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
    randidx = np.random.permutation(mesh.faces.shape[0])[:max_rays]

    # print(randidx.dtype)
    # randidx = np.random.randint(low=0, high=mesh.faces.shape[0], size=n)
    directional_dist = (mesh.triangles_center - startpoint)[randidx,:]  # direction vectors form start to triangle centers

    origin = np.full((directional_dist.shape[0], 3), fill_value=startpoint)  # for each ray same origin
    idxs = ray.intersects_first(origin, directional_dist)

    # might be duplicated idxs
    idxs = np.unique(idxs)

    mask = np.zeros(len(mesh.faces), dtype=bool)
    mask[idxs[idxs >= 0]] = 1  # set 1 where ray hit the face
    mesh.update_faces(mask)

    return mesh


def visible_mesh_to_pcl_scene(vertices, faces, H, max_rays, n_points, front_facing=False, mark_points=True,
                              ray_casting=False):
    ''' Transforms object (defined by vertices and faces) into original scene point cloud. Object only
    contains those points that are visible from the origin given the transformation.

    :param scene_pcl: point cloud of scene in which to transform other object
    :param vertices: vertices
    :param faces: faces (faces + vertices create trimesh object)
    :param max_rays: ray computation takes a long time, max_rays determines how many rays to check.
    :param n_points: number of points to sample from visible mesh of object.
    :param H: 4x4 transformation matrix to transform object from origin in scene
    :param mark_points: if true adds another channel to the pcl where 1 indicates that point belongs
    to the object and 0 that it belongs to the original scene.
    :return: point cloud of scene with object transformed into it
    '''
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.apply_transform(H)

    if front_facing:
        mesh = get_front_facing_mesh(mesh, startpoint=np.array([0, 0, -100000]))

    # ray casting takes long time
    if ray_casting:
        visible_mesh = get_visible_mesh(mesh, startpoint=np.array([0, 0, -100000]), max_rays=max_rays)
    else:
        visible_mesh = mesh

    visible_mesh_pcl = np.array(visible_mesh.sample(n_points))

    # if mark_points:
    #    scene_pcl = np.concatenate([scene_pcl, np.zeros((scene_pcl.shape[0], 1))], axis=1)
    #    visible_mesh_pcl = np.concatenate([visible_mesh_pcl, np.zeros((visible_mesh_pcl.shape[0], 1))], axis=1)

    # mesh_in_scene_pcl = np.concatenate([scene_pcl, visible_mesh_pcl], axis=0)

    return visible_mesh_pcl  # mesh_in_scene_pcl


def batch_visible_mesh_to_pcl_scene(vertices, faces, H,
                                    n_points=1024, mark_points=False, front_facing=False, ray_casting=False,
                                    device='cpu'):
    results = []
    for (verts, fac, h) in list(zip(vertices, faces, H)):
        results.append(visible_mesh_to_pcl_scene(verts.cpu(), fac.cpu(), h.cpu(), max_rays=1000, n_points=n_points,
                                                 mark_points=mark_points, front_facing=front_facing,
                                                 ray_casting=ray_casting))

    return torch.from_numpy(np.array(results)).float().to(device)


def batch_visible_mesh_to_pcl_scene_parallel(vertices, faces, H, pool_size=4):
    ''' Performs batch wise mesh to pcl scene in parallel. Only improves runtime if raycasting is on.

    :param scene_pcl:
    :param vertices:
    :param faces:
    :param H:
    :param pool_size: number of cpus to use
    :return: list of point clouds.
    '''

    args = zip(vertices, faces, H)
    pool = multiprocessing.Pool(pool_size)
    results = pool.starmap(partial(visible_mesh_to_pcl_scene, max_rays=1000, n_points=1024), args)

    return torch.from_numpy(np.array(results)).float()
