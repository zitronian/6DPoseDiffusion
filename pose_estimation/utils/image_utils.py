import numpy as np
import random
import scipy


def add_noise_to_mask(mask, max_rounds = 5):
    """
    Adds noise to mask with erosion and dilation.

    :param mask:
    :param max_rounds: number of rounds to add noise. the higher, the more noise.
    :return: noisy mask
    """

    mask_im = mask.astype(bool)

    structures = [scipy.ndimage.generate_binary_structure(2, 1), scipy.ndimage.generate_binary_structure(2, 2)]
    noisy_mask = mask_im.copy()
    n_rounds = int((np.random.rand()*max_rounds))
    for i in range(0, n_rounds):
        if round(random.random()):
            structure = random.choice(structures)
            noisy_mask = scipy.ndimage.binary_erosion(noisy_mask, structure=structure)
        if round(random.random()):
            structure = random.choice(structures)
            noisy_mask = scipy.ndimage.binary_dilation(noisy_mask, structure=structure)

    return noisy_mask

def masked_pcl_from_depth_img(depth_image, intrinsics, depth_scale, obj_mask, mask=None):
    """

    :param depth_image: depth image
    :param intrinsics: camera intrinsics
    :param depth_scale: depth scale (depends on dataset),
    for BOP: https://github.com/thodan/bop_toolkit/blob/master/docs/bop_datasets_format.md
    :param obj_mask: mask of object for image
    :param mask: legacy

    :return: point cloud that only contains points that are in mask of image
    """

    grid = np.mgrid[0: depth_image.shape[0], 0:depth_image.shape[1]]
    u, v = grid[0], grid[1]
    z = depth_image * depth_scale
    x = (v - intrinsics['cx']) * z / intrinsics['fx']
    y = (u - intrinsics['cy']) * z / intrinsics['fy']
    data = np.stack([x, y, z], axis=-1)

    if obj_mask is None:
        # dont crop
        pcl = data.reshape(-1,3)
    else:

        pcl = data[obj_mask > 0]

    return pcl

