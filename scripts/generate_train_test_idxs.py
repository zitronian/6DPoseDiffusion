import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'pose_estimation'))
import glob
import numpy as np
import tqdm

def generate_train_test(objs, sub_dir, train_txt_name='train_idxs.txt', test_txt_name='test_idxs.txt', train_split=0.8, seed=42):
    """
    Generate train test idxs for specifc object of lm dataset

    :param objs:
    :param sub_dir:
    :param train_txt_name:
    :param test_txt_name:
    :param train_split:
    :param seed:
    :return:
    """

    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(base_path, 'bop_data', 'lm', sub_dir)

    for obj in tqdm.tqdm(objs):
        obj_dir = os.path.join(dataset_dir, '{id:06d}'.format(id=obj))
        size = len(glob.glob(os.path.join(obj_dir, 'rgb', '*.*')))

        idxs = np.random.RandomState(seed).permutation(size)
        train_idxs = np.sort(idxs[:int(train_split*size)])
        test_idxs = np.sort(idxs[int(train_split*size):])

        for (split, file_name) in [(train_idxs, train_txt_name), (test_idxs, test_txt_name)]:
            file = os.path.join(obj_dir, file_name)
            assert not os.path.exists(file) # make sure path does not exist yet

            with open(file, 'w') as f:
                for idx in split:
                    f.write('{id:04d}\n'.format(id=idx))



if __name__ == '__main__':

    generate_train_test(objs=[8], sub_dir='train_real')