import os
import numpy as np
import torch
import pickle as pk

# this should be available
import utils.preprocess as prep

# check preprocess_SMAP, both train and test need to have same amount of entity

# MSL dataset
class SIM_Dataset():
    def __init__(self, dataset_pth, entities=None):
        self.dims = 4  
        # Too many entities, don't want to create positional encoding or one-hot, fake it to be 1
        # 4611 is derived from printing id_length when making pk file
        self.num_entity = 1
        with open(dataset_pth, 'rb') as file:
            self.dat = pk.load(file)

    def preprocess(self, params):
        # parameters
        dl = params.dl
        stride = params.stride
        tst_stride = dl if params.tst_stride == 'no_rep' else params.tst_stride

        x_trn_all, x_tst_all, lab_tst_all = [], [], []
        for entity_id in range(4611):
            dat_ent = {}
            for key in self.dat.keys():
                dat_ent[key] = self.dat[key][entity_id]
            dat = prep.preprocess(dat_ent, params, self.dims, self.num_entity, entity_id, quiet=True)
            x_trn_all.append(dat['x_trn'])
            x_tst_all.append(dat['x_tst'])
            lab_tst_all.append(dat['lab_tst'])
        return prep.window_stride(x_trn_all, x_tst_all, lab_tst_all, 4611, dl, stride, tst_stride)

