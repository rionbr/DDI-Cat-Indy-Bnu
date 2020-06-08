# coding=utf-8
# Author: Rion B Correia
# Date: June 08, 2020
#
# Description: utility functions
#
#
import os
import math
import scipy
import numpy as np
import pandas as pd


def calc_confidence_interval(x):
    # mean, count, std = x.mean(), x.count(), x.std()
    mean, count, std = np.mean(x), len(x), np.std(x)
    sigma = std / math.sqrt(count)
    df = count - 1
    (ci_min, ci_max) = scipy.stats.t.interval(alpha=0.95, df=df, loc=mean, scale=sigma)
    return pd.Series({
        'mean': mean,
        'count': count,
        'std': std,
        'ci95-max': ci_max,
        'ci95-min': ci_min})


def map_age_to_age_group(x):
    bins = list(range(0, 101, 5))
    labels = ['%02d-%02d' % (x, x1 - 1) for x, x1 in zip(bins[: -1], bins[1:])] + ['>99']
    #
    return pd.cut(x, bins=bins + [np.inf], include_lowest=True, right=False, labels=labels)


def ensurePathExists(path):
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
