from __future__ import absolute_import
from __future__ import division

import theano
from theano import tensor as T
import numpy as np


def clip_by_norm(t, threshold):
    raise NotImplementedError

def clip_by_global_norm(t_list, threshold):
    norm = T.sqrt(sum([T.sum(T.sqr(t)) for t in t_list]))
    if threshold > 0:
        t_list = [T.switch(norm >= threshold, t * threshold / norm, t) for t in t_list]
    # TODO: warn when threshold <= 0
    return t_list

def clip_by_value(t_list, min_threshold, max_threshold):
    return [T.clip(t, min_value, max_value) for t in t_list]
