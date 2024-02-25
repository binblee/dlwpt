from collections import namedtuple
import glob, os
import functools
import csv
import SimpleITK as sitk
import numpy as np
import torch
from torch.utils.data import Dataset
import copy
import random

from util.util import XyzTuple, xyz2irc
from util.logconfig import logging
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
from util.disk import getCache
raw_cache = getCache('G:/luna_cache/','ch10_raw')


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

class LunaDataset(Dataset):
    def __init__(self, 
                 val_stride=0, isValSet_bool=None, series_uid=None, sortby_str='random',
                 data_path='.'):
        self.data_path = data_path
        log.info("CT data loaded from {}".format(self.data_path))
        self.candidateInfo_list = copy.copy(getCandidateInfoList(self.data_path))
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

        log.info("{!r}: {} {} samples".format(
            self,
            len(self.candidateInfo_list),
            'validation' if isValSet_bool else 'training'
        ))
    
    def __len__(self):
        return len(self.candidateInfo_list)
    
    def __getitem__(self, ndx):
        candidateInfo_tup = self.candidateInfo_list[ndx]
        width_irc = (32, 48, 48)
        candidate_a, center_irc = getCtRawCandidate(
            candidateInfo_tup.series_uid,
            candidateInfo_tup.center_xyz,
            width_irc,
            data_path=self.data_path
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