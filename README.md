# PyTorch implementation for 6D Pose Estimation using a Noise Conditioned Score Model

This repo provides tools to train and test a Noise Conditioned Score Model for 6D pose estimation.
Training is conducted via a Denoised Score Matching objective and inference is performed using
Langevin dynamics.

For logging we use weights and bias. Please initialize https://docs.wandb.ai/quickstart.

## Environment

Create conda environment

```bash
conda env create -f 6Dpose_estimation_env.yml
```

Active environment

```bash
conda activate 6Dpose_estimation_env
```

This repo relies on the Theseus, Please refer to https://github.com/AI-App/Theseus in case of installation issues.

## Data

We assume the data to be in the format followed in the BOP challenge
(https://github.com/thodan/bop_toolkit/blob/master/docs/bop_datasets_format.md).

E.g. for use of the Linemod dataset, download the respective files according to https://bop.felk.cvut.cz/datasets/
and place them in the following structure:

    ├── ...
    ├── bop_data
    │   ├── lm
    │   │   ├── <subdir name>
    │   │   │   ├── <block_id>                  # e.g. 000001
    │   │   │   │   ├── depth                   # folder with depth images
    │   │   │   │   ├── mask                    # foler with masks
    │   │   │   │   ├── mask_visib              # folder with visible masks
    │   │   │   │   ├── rgb                     # folder with rgb images
    │   │   │   │   ├── scene_gt.json           # folder with gt
    │   │   │   │   ├── scene_gt_info           # folder with further infos
    │   │   │   │   ├── train_idxs.txt          # file with train idxs (optional)
    │   │   │   │   ├── test_idxs.txt           # file with test images (optional)
    │   │   │   ├── <block_id>                  # e.g. 000002
    │   │   │   └── ...
    │   │   └── camera.json                     # camera intrinsics
    │   │   └── models
    │   └── ...
    ├── scripts
    └── pose_estimation

Block ids can be any 6 digit string and the names of images in depth/mask/mask_visib/rgb should be a
number with 6 digits starting at 000000, 000001, ....

## Training the Noise Conditioned Score Model

Configuration files for training are stored in `scripts/config/`. The folder contains one example
configuration for training the model on object 8 (driller).

Training requires two command line arguments

- config_file: `scripts/config/model_config/<config file name>.yaml` defining configurations of training
- wandb_mode: if 'online' the run is logged to the wandb server, if 'offline' it is only logged locally.

```bash
python scripts train_pose_estimation.py --config_file train_config_example --wandb_mode offline
```

## Inference of 6D Poses

To sample 6D poses given an RGB-D image and the 3D Model of the object, we use Langevin dynamics for
iterative inference process. It starts by randomly sampling a pose and then gradually walking
along the gradient (output of the trained model) towards an accurate solution.

Inference requires two command line arguments

- config_file: `scripts/config/inference_config/<config file name>.yaml` defining configurations of inference
- wandb_mode: if 'online' the run is logged to the wandb server, if 'offline' it is only logged locally.

```bash
python scripts train_pose_estimation.py --config_file inference_config_example --wandb_mode offline
```

Inference will sample multiple particles as pose hypothesis. All pose hypothesis will be stored as a dictionary
in a .npy file in `results/<wandb_run_name>`. Load the results dict using

```python
results_dict = np.load(res_path, allow_pickle=True).item()
```

The results dict will contain various information, such as the sampled poses, the history of poses
throughout the inference process, the history of scores as well as the scene and object latent
in the final iteration.

To sample a final pose prediction from the sampled particles, you can use the Selection Strategies in
`pose_estimation/samplers/pose_sampler.py`. Utility functions in `visualizations/utils.py` provide
useful functions to do so. You can use

```python
from visualizations.utils import results_as_df
poses_df = results_as_df(results_dict)
```

to apply random selection, selection by score, selection by latent, selection by ground truth to
the sampled particles and get a dataframe containing the final pose prediction based on each of
the four methods for each sample of the inference dataset.
