import numpy as np

def calc_recall(tp_count, targets_count):
    """Calculates recall.

    :param tp_count: Number of true positives.
    :param targets_count: Number of targets.
    :return: The recall rate.
    """
    if targets_count == 0:
        return 0.0
    else:
        return tp_count / float(targets_count)


def calc_scores(results_dict, models_info, n_top=5, correct_th=0.1, normalize_by_diameter=True, error_type='add'):
    ''' Computes scores of best n estimated poses. Pose is true if error_type is less than correct_th*obj_diameter.
    Recall is computed separatly for each object and jointly over all objects.

    :param results_dict: dict with results
    :param models_info: dict from models_info.json
    :param normalize_by_diameter: if True error is normalized with object diameter.
    :param n_top: sampler samples multiple poses. Consider only best n_top poses.
    :return: Dict with results
    '''

    objects = np.unique(results_dict['obj_id']).astype(int)
    diameter_dict = dict.fromkeys(objects)
    for obj in objects:
        diameter_dict[obj] = models_info[str(obj)]['diameter']


    overall_tps = 0
    overall_tars = 0
    counts_dict = {obj: {'success_counts': 0, 'num_counts': 0} for obj in objects}
    for obj, distances in zip(results_dict['obj_id'], results_dict[error_type]):

        # get n_top poses with lowest error
        if np.isscalar(distances):
            n_top_poses = np.array([distances])
        else:
            n_top_poses = np.sort(distances)[:n_top]

        # normalize error by object diameter if normalize_by_diameter is True
        n_top_poses = (n_top_poses / diameter_dict[obj]) if normalize_by_diameter else n_top_poses
        for distance in n_top_poses:
            if distance < correct_th:
                overall_tps += 1
                counts_dict[obj]['success_counts'] += 1
        counts_dict[obj]['num_counts'] += n_top_poses.shape[0]
        overall_tars += n_top_poses.shape[0]

    obj_recalls = {obj: None for obj in objects}
    for obj, obj_counts in counts_dict.items():
        obj_recalls[obj] = calc_recall(obj_counts['success_counts'], obj_counts['num_counts'])

    mean_obj_recall = float(np.mean(list(obj_recalls.values())).squeeze())
    avg_recall = calc_recall(overall_tps, overall_tars)

    scores_dict = {
        'avg_recall': float(avg_recall),  # recall averages over all object
        'obj_recalls': obj_recalls,  # recall for each object
        'mean_obj_recall': float(mean_obj_recall),  # mean of obj_recalls
        'targets': int(overall_tars),  # number of targets
        'overall_tps': int(overall_tps),  # true positives over all objects
        'correct-th': correct_th,  # used threshold
    }

    return scores_dict


