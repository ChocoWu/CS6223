import json
import os
import errno

import pandas as pd
import torch
import random
import numpy as np
from tqdm import tqdm


class Logger(object):

    def __init__(self, output_name):
        dirname = os.path.dirname(output_name)
        if not os.path.exists(dirname):
            os.mkdir(dirname)

        self.log_file = open(output_name, 'w')
        self.infos = {}

    def append(self, key, val):
        vals = self.infos.setdefault(key, [])
        vals.append(val)

    def log(self, extra_msg=''):
        msgs = [extra_msg]
        for key, vals in self.infos.iteritems():
            msgs.append('%s %.6f' % (key, np.mean(vals)))
        msg = '\n'.join(msgs)
        self.log_file.write(msg + '\n')
        self.log_file.flush()
        self.infos = {}
        return msg

    def write(self, msg):
        self.log_file.write(str(msg) + '\n')
        self.log_file.flush()
        # print(msg)


def load_json(path):
    with open(path, 'r') as f:
        x = json.load(f)
    return x


def save_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f)


def load_csv(path):
    return pd.read_csv(path, header=0)


def save_csv(data, path, col_name):
    data.to_csv(path, header=col_name, index=False)


def create_dir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # CPU random seed
    torch.cuda.manual_seed(seed)  # GPU random seed
    torch.backends.cudnn.benchmark = True


