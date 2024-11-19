import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
plt.rc('text', usetex=True)
def get_ranks(arr):
    ''' returns the ranks of each value in arr. the lower the rank, the better

    :param arr:
    :return:
    '''
    order = np.argsort(arr, axis=-1)
    ranks = np.argsort(order)
    return ranks


def filter_arr_with_order(metric_arr, order_arr, n_best=1):
    '''

    :param metric_arr: arr with metric
    :param order_arr: arr holding indices
    :param n_best:
    :return:
    '''
    # filter with rank
    filtered_metrics = np.zeros((metric_arr.shape[0], n_best))
    for i, sample_metric in enumerate(metric_arr):
        filtered_metrics[i] = sample_metric[order_arr[i,:n_best]]
    return filtered_metrics

def select_by_lastn_scores(data, metric='add', last_n=1, n_best=1):
    
    max_len = data['score_hist'].shape[1] 

    metrics = data[metric]
    final_scores_norm = np.linalg.norm(data['score_hist'][:,max_len-last_n:,:,:], ord=2, axis=-1).sum(axis=1)
    order = np.argsort(final_scores_norm)
    filter_metrics = filter_arr_with_order(metrics, order, n_best=n_best)

    return filter_metrics

def select_first(data, metric='add', n_best=1):
    return data[metric][:,0]


def select_by_score_hist(data, metric='add', n_best=1):
    '''

    :retur: for each 10 particles the metric (e.g. add) of the n particles with the lowest sum of all score norms over
    all iterations (e.g. 300 iterations).
    '''
    metrics = data[metric]
    final_scores_norm = np.linalg.norm(data['score_hist'], ord=2, axis=-1).sum(axis=1)
    order = np.argsort(final_scores_norm)
    filter_metrics = filter_arr_with_order(metrics, order, n_best=n_best)

    return filter_metrics

def select_by_gt(data, metric='add', n_best=1):
    metrics = data[metric]
    order = np.argsort(metrics)
    filter_metrics = filter_arr_with_order(metrics, order, n_best=n_best)

    return filter_metrics

def select_by_latent(data, metric='add', n_best=1):
    metrics = data[metric]

    criterion = np.linalg.norm(data['scene_latent'] - data['obj_latent'], ord=2, axis=3).mean(axis=2)
    order = np.argsort(criterion)
    filter_metrics = filter_arr_with_order(metrics, order, n_best=n_best)


    return filter_metrics

def median_selection(data, metric='add', n_best=1):
    '''
    :return: compute median particle (according to metric e.g. add) for each sample
    '''

    return np.median(data[metric], axis=1)

def random_selection(data, metric='add', n_best=1):

    rand_selected = np.empty((data[metric].shape[0], n_best))
    for i, particles in enumerate(data[metric]):
        idxs = np.random.randint(0, data[metric].shape[1], size=(n_best))
        rand_selected[i] = particles[idxs]
    return rand_selected
def particle_selection_statistics(arr):
    '''

    :param arr: 1D array with add metric of selected particle per each sample
    :return:
    '''

    return {
        'mean': np.mean(arr),
        'median': np.median(arr),
    }


if __name__ == '__main__':

    path = 'results/generous-plant-57/results.npy'
    data = np.load(path, allow_pickle=True).item()

    rand_selection = random_selection(data, n_best=1)
    median_selection = median_selection(data, n_best=1)
    by_score = select_by_score_hist(data, n_best=1)
    by_gt = select_by_gt(data, n_best=1)
    by_latent = select_by_latent(data, n_best=1)

    print('By random: ', particle_selection_statistics(rand_selection.flatten()))
    print('By gt: ', particle_selection_statistics(by_gt))
    print('By median: ', particle_selection_statistics(median_selection.flatten()))
    print('By score: ', particle_selection_statistics(by_score.flatten()))
    print('By latent: ', particle_selection_statistics(by_latent))

    df = {'by_score': by_score.flatten(),
          'by_random': rand_selection.flatten(),
          'by_gt': by_gt.flatten(),
          'by_latent': by_latent.flatten(),
          'by_median': median_selection.flatten(),
          'sample': sorted(list(range(by_score.shape[0])))}
    df = pd.melt(pd.DataFrame.from_dict(df), id_vars=['sample'], value_vars=['by_median', 'by_score', 'by_latent', 'by_gt', 'by_random'],
                 var_name='selection', value_name='add')

    #df_noselection = pd.DataFrame({'selection': ['no_selection']*no_selection.shape[0]*no_selection.shape[1],
    #                               'add': no_selection.flatten(),
    #                               'sample': [0]*no_selection.flatten().shape[0]})
    #df = pd.concat([df_noselection,df], axis=0)
    ax = sns.boxplot(df, x='selection', y='add')
    ax.set_ylabel('add [mm]')
    plt.title('Particle Selection')
    plt.show()
    print('FINISHED')