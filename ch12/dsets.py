from collections import namedtuple
import glob, os
import functools
import csv
import SimpleITK as sitk
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import copy
import random
import math

from util.util import XyzTuple, xyz2irc
from util.logconfig import logging
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
from util.disk import getCache
raw_cache = getCache('luna_cache/','subset0_1_augmented')


CandidateInfoTuple = namedtuple(
    'CandidateInfoTuple',
    'isNodule_bool, diameter_mm, series_uid, center_xyz'
)

@functools.lru_cache(1)
def getCandidateInfoList(dataPath, requireOnDisk=True):
    mhd_list = glob.glob(os.path.join(dataPath, 'subset*/*.mhd'))
    presentOnDisk_set = {os.path.split(p)[-1][:-4] for p in mhd_list}
    
    diameter_dict = {}
    with open(os.path.join(dataPath, 'annotations.csv'), 'r') as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]
            annotationCenter_xyz = tuple([float(x) for x in row[1:4]])
            annotationDiameter_mm = float(row[4])
            diameter_dict.setdefault(series_uid, []).append(
                (annotationCenter_xyz, annotationDiameter_mm)
            )

    candidateInfo_list = []
    with open(os.path.join(dataPath, 'candidates.csv')) as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]
            if series_uid not in presentOnDisk_set and requireOnDisk:
                continue
            isNodule = bool(int(row[4]))
            candidateCenter_xyz = tuple([float(x) for x in row[1:4]])
            candidateDiameter_mm = 0.0
            for annotation_tup in diameter_dict.get(series_uid, []):
                annotationCenter_xyz, annotationDiameter_mm = annotation_tup
                for i in range(3):
                    delta_mm = abs(candidateCenter_xyz[i] - annotationCenter_xyz[i])
                    if delta_mm > annotationDiameter_mm / 4:
                        break
                else:
                    candidateDiameter_mm = annotationDiameter_mm
                    break
            
            candidateInfo_list.append(CandidateInfoTuple(
                isNodule, candidateDiameter_mm, series_uid, candidateCenter_xyz
                ))
            
    candidateInfo_list.sort(reverse=True)
    return candidateInfo_list

class Ct:
    def __init__(self, series_uid, data_path):
        mhd_path_t = os.path.join(data_path, 'subset*', f'{series_uid}.mhd')
        mhd_path_list = glob.glob(mhd_path_t)
        assert len(mhd_path_list) == 1, repr([mhd_path_t, mhd_path_list, len(mhd_path_list)])
        mhd_path = mhd_path_list[0]
        ct_mhd = sitk.ReadImage(mhd_path)
        ct_a = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)
        ct_a.clip(-1000, 1000, ct_a)
        self.series_uid = series_uid
        self.hu_a = ct_a
        self.origin_xyz = XyzTuple(*ct_mhd.GetOrigin())
        self.vxSize_xyz = XyzTuple(*ct_mhd.GetSpacing())
        self.direction_a = np.array(ct_mhd.GetDirection()).reshape(3,3)
        
    def getRawCandidate(self, center_xyz, width_irc):
        center_irc = xyz2irc(
            center_xyz,
            self.origin_xyz,
            self.vxSize_xyz,
            self.direction_a
        )

        slice_list = []
        for axis, center_val in enumerate(center_irc):
            start_ndx = int(round(center_val - width_irc[axis] / 2))
            end_ndx = int(start_ndx + width_irc[axis])
            assert center_val >=0 and center_val < self.hu_a.shape[axis], repr([self.series_uid, center_xyz, self.origin_xyz, self.vxSize_xyz, center_irc, axis])

            if start_ndx < 0:
                start_ndx = 0
                end_ndx = int(width_irc[axis])

            if end_ndx > self.hu_a.shape[axis]:
                end_ndx = self.hu_a.shape[axis]
                start_ndx = int(self.hu_a.shape[axis] - width_irc[axis])

            slice_list.append(slice(start_ndx, end_ndx))
        
        ct_chunk = self.hu_a[tuple(slice_list)]
        return ct_chunk, center_irc

@functools.lru_cache(1, typed=True)
def getCt(series_uid, data_path):
    return Ct(series_uid=series_uid, data_path=data_path)

@raw_cache.memoize(typed=True)
def getCtRawCandidate(series_uid, center_xyz, width_irc, data_path):
    ct = getCt(series_uid=series_uid, data_path=data_path)
    ct_chunk, center_irc = ct.getRawCandidate(center_xyz, width_irc)
    return ct_chunk, center_irc

def getCtAugmentedCandidate(
        augmentation_dict,
        series_uid, center_xyz, width_irc,
        data_path,
        use_cache=True):
    # TODO: for debug purpose, to be removed
    assert use_cache, repr([augmentation_dict,
        series_uid, center_xyz, width_irc,
        data_path,
        use_cache])
    if use_cache:
        ct_chunk, center_irc = getCtRawCandidate(series_uid, center_xyz, width_irc, data_path)
    else:
        ct = getCt(series_uid, data_path=data_path)
        ct_chunk, center_irc = ct.getRawCandidate(center_xyz, width_irc)
    ct_t = torch.tensor(ct_chunk).unsqueeze(0).unsqueeze(0).to(torch.float32)
    transform_t = torch.eye(4)
    for i in range(3):
        if 'flip' in augmentation_dict:
            if random.random() > 0.5:
                transform_t[i, i] *= -1
        if 'offset' in augmentation_dict:
            offset_float = augmentation_dict['offset']
            random_float = (random.random() * 2 -1)
            transform_t[i,3] = offset_float * random_float
        if 'scale' in augmentation_dict:
            scale_float = augmentation_dict['scale']
            random_float = (random.random() * 2 - 1)
            transform_t[i,i] *= 1.0 + scale_float * random_float
    if 'rotate' in augmentation_dict:
        angle_rad = random.random() * math.pi * 2
        s = math.sin(angle_rad)
        c = math.cos(angle_rad)
        rotation_t = torch.tensor([
            [c, -s, 0, 0],
            [s,  c, 0, 0],
            [0,  0, 1, 0],
            [0,  0, 0, 1]
        ])

        transform_t @= rotation_t

    affine_t = F.affine_grid(
        transform_t[:3].unsqueeze(0).to(torch.float32),
        ct_t.size(),
        align_corners=False,
    )

    augmentated_chunk = F.grid_sample(
        ct_t,
        affine_t,
        padding_mode='border',
        align_corners=False,
    ).to('cpu')

    if 'noise' in augmentation_dict:
        noise_t = torch.randn_like(augmentated_chunk)
        noise_t *= augmentation_dict['noise']
        augmentated_chunk += noise_t

    return augmentated_chunk[0], center_irc

class LunaDataset(Dataset):
    def __init__(self, 
                 val_stride=0, isValSet_bool=None, series_uid=None, sortby_str='random',
                 ratio_int=0, augmentation_dict=None, candidateInfo_list=None,
                 data_path='.'):
        self.data_path = data_path
        log.info("CT data loaded from {}".format(self.data_path))

        self.ratio_int = ratio_int
        self.augmentation_dict = augmentation_dict

        if candidateInfo_list:
            self.candidateInfo_list = copy.copy(candidateInfo_list)
            self.use_cache = False
        else:
            self.candidateInfo_list = copy.copy(getCandidateInfoList(self.data_path))
            self.use_cache = True

        log.info('use cache: {}'.format(self.use_cache))
        if series_uid:
            self.candidateInfo_list = [
                x for x in self.candidateInfo_list if x.series_uid == series_uid
            ]
        
        if isValSet_bool:
            assert val_stride > 0, val_stride
            self.candidateInfo_list = self.candidateInfo_list[::val_stride]
            assert self.candidateInfo_list
        elif val_stride > 0:
            del self.candidateInfo_list[::val_stride]
            assert self.candidateInfo_list

        if sortby_str == 'random':
            random.shuffle(self.candidateInfo_list)
        elif sortby_str == 'series_uid':
            self.candidateInfo_list.sort(key=lambda x:(x.series_uid, x.center_xyz))
        elif sortby_str == 'label_and_size':
            pass
        else:
            raise Exception('Unknown sort: ' + repr(sortby_str))

        self.negative_list = [nt for nt in self.candidateInfo_list if not nt.isNodule_bool]
        self.positive_list = [nt for nt in self.candidateInfo_list if nt.isNodule_bool]

        log.info("{!r}: {} {} samples, {} neg, {} pos, {} ratio".format(
            self,
            len(self.candidateInfo_list),
            'validation' if isValSet_bool else 'training',
            len(self.negative_list), len(self.positive_list),
            '{}:1'.format(self.ratio_int) if self.ratio_int else 'unbalanced'
        ))
    
    def shuffleSamples(self):
        if self.ratio_int:
            random.shuffle(self.negative_list)
            random.shuffle(self.positive_list)
    
    def __len__(self):
        if self.ratio_int:
            # number of training samples in subset0~1 is 98879, close to 10k.
            # use only 40% of the negative samples
            return 40000  
        return len(self.candidateInfo_list)
    
    def __getitem__(self, ndx):
        # ration_int = 0 means unbalanced dataset
        if self.ratio_int:  
            pos_ndx = ndx // (self.ratio_int + 1)
            if ndx % (self.ratio_int + 1):
                # negative sample
                neg_ndx = ndx - 1 - pos_ndx
                neg_ndx %= len(self.negative_list)
                candidateInfo_tup = self.negative_list[neg_ndx]
            else:
                pos_ndx %= len(self.positive_list)
                candidateInfo_tup = self.positive_list[pos_ndx]
        else:
            candidateInfo_tup = self.candidateInfo_list[ndx]
        
        
        width_irc = (32, 48, 48)
        if self.augmentation_dict:
            candidate_t, center_irc = getCtAugmentedCandidate(
                self.augmentation_dict,
                candidateInfo_tup.series_uid,
                candidateInfo_tup.center_xyz,
                width_irc,
                use_cache=self.use_cache,
                data_path=self.data_path,
            )
        elif self.use_cache:
            candidate_a, center_irc = getCtRawCandidate(
                candidateInfo_tup.series_uid,
                candidateInfo_tup.center_xyz,
                width_irc,
                data_path=self.data_path,
            )
            candidate_t = torch.from_numpy(candidate_a).to(torch.float32)
            candidate_t = candidate_t.unsqueeze(0)
        else:
            ct = getCt(candidateInfo_tup.series_uid, data_path=self.data_path)
            candidate_a, center_irc = ct.getRawCandidate(
                candidateInfo_tup.center_xyz,
                width_irc
            )
            candidate_t = torch.from_numpy(candidate_a).to(torch.float32)
            candidate_t = candidate_t.unsqueeze(0)

        pos_t = torch.tensor([
            not candidateInfo_tup.isNodule_bool,
            candidateInfo_tup.isNodule_bool
        ], dtype=torch.long)

        return (
            candidate_t,
            pos_t,
            candidateInfo_tup.series_uid,
            torch.tensor(center_irc)
        )