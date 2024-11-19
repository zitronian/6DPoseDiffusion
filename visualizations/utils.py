import numpy as np
import os
import json
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pose_estimation.evaluation.sample_selection import select_by_score_hist, select_by_gt, select_by_latent, random_selection, select_by_lastn_scores, select_first
from pose_estimation.evaluation.evaluate import eval_scores
from prettytable import PrettyTable
import pprint
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
plt.rc('text', usetex=True)
plt.rcParams['image.cmap']='Accent'
sns.set_palette('Accent')

'''
Some util functions for visualization
'''


def get_models_info(dataset_path='pose_estimation/bop_data/lm'):
    f = open(os.path.join(dataset_path, 'models',
                          'models_info.json'))
    models_info = json.load(f)
    return models_info

def get_res_dict(path):
    """

    :param path: path to results .pny
    :return: results dict
    """
    res_path = path
    results_dict = np.load(res_path, allow_pickle=True).item()
    results_dict['obj_id'] = results_dict['obj_id'].astype(int)

    return results_dict

def acc_from_score_dict(score_dict, obj, thr=0.1):
    """
    get accuracy and auc for specifix object or averages over all objects from score dictionary.

    :param score_dict:
    :param obj: respctive object. if None, take average over all objects
    :param thr: threshold for which to get accuracy. i.e. 0.1 -> add < 10% obj diameter
    :return:
    """

    acc = score_dict['obj_avg_recalls'][obj] if obj is not None else score_dict['avg_recalls']
    auc = score_dict['obj_auc'][obj] if obj is not None else score_dict['auc']

    thr_idx = np.argwhere(score_dict['correct_ths']==thr).item()
    return acc[thr_idx], auc

def results_as_df(res_dict, models_info=get_models_info()):
    """
    Aggregates final pose from sampled particles with selection strategies: by score, by latent, random, by gt

    :param res_dict: results dict
    :param models_info: info for 3D object models, e.g. contains diamter
    :return: dataframe with metric for each sample of the dataset and each selection strategy.
    """

    acc_dict = res_dict.copy()
    df_auc = pd.DataFrame(columns=['metric', 'obj_id', '01ACC', 'AUC'])
    df = pd.DataFrame(dtype=float)

    for metric in ['add', 're', 'te']:

        
        ### by last n scores
        by_score = select_by_lastn_scores(data=res_dict, metric=metric, last_n=1, n_best=1)
        m = '{}_byscore'.format(metric)
        df[m] = by_score.squeeze()

        if metric == 'add':
            acc_dict[m] = by_score
            score_eval = eval_scores(acc_dict, models_info, correct_ths=np.arange(0, 0.22, 0.005), normalize_by_diameter=True, error_type=m, n_top=1)
            for obj in score_eval['obj_auc']:
                acc, auc = acc_from_score_dict(score_eval, obj, thr=0.1)
                df_auc = pd.concat([df_auc, pd.Series({'metric':m, 'obj_id':obj, '01ACC': acc, 'AUC': auc}).to_frame().T], ignore_index=True)

        ### by gt
        bygt = select_by_gt(data=res_dict, metric=metric, n_best=1)
        m = '{}_bygt'.format(metric)
        df[m] = bygt.squeeze()

        if metric == 'add':
            acc_dict[m] = bygt
            score_eval = eval_scores(acc_dict, models_info, correct_ths=np.arange(0, 0.22, 0.005), normalize_by_diameter=True, error_type=m, n_top=1)
            for obj in score_eval['obj_auc']:
                acc, auc = acc_from_score_dict(score_eval, obj, thr=0.1)
                df_auc = pd.concat([df_auc, pd.Series({'metric':m, 'obj_id':obj, '01ACC': acc, 'AUC': auc}).to_frame().T], ignore_index=True)

        ### by latent
        bylatent = select_by_latent(data=res_dict, metric=metric, n_best=1)
        m = '{}_bylatent'.format(metric)
        df[m] = bylatent.squeeze()

        if metric == 'add':
            acc_dict[m] = bylatent
            score_eval = eval_scores(acc_dict, models_info, correct_ths=np.arange(0, 0.22, 0.005), normalize_by_diameter=True, error_type=m, n_top=1)
            for obj in score_eval['obj_auc']:
                acc, auc = acc_from_score_dict(score_eval, obj, thr=0.1)
                df_auc = pd.concat([df_auc, pd.Series({'metric':m, 'obj_id':obj, '01ACC': acc, 'AUC': auc}).to_frame().T], ignore_index=True)
        '''
        ### by random
        byrandom = random_selection(data=res_dict, metric=metric, n_best=1)
        m = '{}_byrandom'.format(metric)
        df[m] = byrandom.squeeze()

        if metric == 'add':
            acc_dict[m] = byrandom
            score_eval = eval_scores(acc_dict, models_info, correct_ths=np.arange(0, 0.22, 0.005), normalize_by_diameter=True, error_type=m, n_top=1)
            for obj in score_eval['obj_auc']:
                acc, auc = acc_from_score_dict(score_eval, obj, thr=0.1)
                df_auc = pd.concat([df_auc, pd.Series({'metric':m, 'obj_id':obj, '01ACC': acc, 'AUC': auc}).to_frame().T], ignore_index=True)
        
        '''
        
        ### by random
        byfirst = select_first(data=res_dict, metric=metric, n_best=1)
        m = '{}_byfirst'.format(metric)
        df[m] = byfirst.squeeze()

        if metric == 'add':
            acc_dict[m] = byfirst
            score_eval = eval_scores(acc_dict, models_info, correct_ths=np.arange(0, 0.22, 0.005), normalize_by_diameter=True, error_type=m, n_top=1)
            for obj in score_eval['obj_auc']:
                acc, auc = acc_from_score_dict(score_eval, obj, thr=0.1)
                df_auc = pd.concat([df_auc, pd.Series({'metric':m, 'obj_id':obj, '01ACC': acc, 'AUC': auc}).to_frame().T], ignore_index=True)
        
        
    df['obj_id'] = res_dict['obj_id'].tolist()

    return df, df_auc

def radar_factory(num_vars, frame='circle'):
    """Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle' | 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    def draw_poly_patch(self):
        # rotate theta such that the first axis is at the top
        verts = unit_poly_verts(theta + np.pi / 2)
        return plt.Polygon(verts, closed=True, edgecolor='k')

    def draw_circle_patch(self):
        # unit circle centered on (0.5, 0.5)
        return plt.Circle((0.5, 0.5), 0.5)

    patch_dict = {'polygon': draw_poly_patch, 'circle': draw_circle_patch}
    if frame not in patch_dict:
        raise ValueError('unknown value for `frame`: %s' % frame)

    class RadarAxes(PolarAxes):

        name = 'radar'
        # use 1 line segment to connect specified points
        RESOLUTION = 1
        # define draw_frame method
        draw_patch = patch_dict[frame]

        def __init__(self, *args, **kwargs):
            super(RadarAxes, self).__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, **kwargs):
            """Override fill so that line is closed by default"""
            closed = kwargs.pop('closed', True)
            return super(RadarAxes, self).fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super(RadarAxes, self).plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            return self.draw_patch()

        def _gen_axes_spines(self):
            if frame == 'circle':
                return PolarAxes._gen_axes_spines(self)
            # The following is a hack to get the spines (i.e. the axes frame)
            # to draw correctly for a polygon frame.

            # spine_type must be 'left', 'right', 'top', 'bottom', or `circle`.
            spine_type = 'circle'
            verts = unit_poly_verts(theta + np.pi / 2)
            # close off polygon by repeating first vertex
            verts.append(verts[0])
            path = Path(verts)

            spine = Spine(self, spine_type, path)
            spine.set_color('k')
            spine.set_transform(self.transAxes)
            return {'polar': spine}

    register_projection(RadarAxes)
    return theta


def unit_poly_verts(theta):
    """Return vertices of polygon for subplot axes.

    This polygon is circumscribed by a unit circle centered at (0.5, 0.5)
    """
    x0, y0, r = [0.5] * 3
    verts = [(r*np.cos(t) + x0, r*np.sin(t) + y0) for t in theta]
    return verts