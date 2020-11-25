# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 13:57:16 2020

@author: kshitij
"""

# Functions to load model/category and fMRI RDMs

import numpy as np
import scipy.io as sio
import os
import glob
from scipy.stats import spearmanr

def get_uppertriangular(rdm):
    """Returns uppertriangular part of RDM.

    Parameters
    ----------
    rdm : np.array
        Input RDM conditions X conditions.

    Returns
    -------
    np.array
        Vectorized version of upper triangular part of Input RDM.

    """
    num_conditions = rdm.shape[0]
    return rdm[np.triu_indices(num_conditions,1)]


def get_searchlight_rdms_persub(fMRI_RDMs_dir,sl_rad = 1,max_blk_edge=2):
    """Returns searchlight RDM per subject.

    Parameters
    ----------
    fMRI_RDMs_dir : string
        path to searchlight directory.
    sl_rad : int
        Radius of searchlight.
    max_blk_edge : int
        Edge length of searchlight.

    Returns
    -------
    type
        Description of returned object.

    """

    # Loading RDM file corresponding to given searchlight parameters
    sl_dir = os.path.join(fMRI_RDMs_dir,'sl_rad=' + str(sl_rad) + '__max_blk_edge=' + str(max_blk_edge))
    all_sub_slrdm_file = os.path.join(sl_dir,'all_sub_slrdms.npy')
    all_sub_slrdm = np.load(all_sub_slrdm_file)
    print("The shape of searchlight RDMs is ", all_sub_slrdm.shape)

    # Converting RDM to upper triangular vectorized form for all searchlights
    all_sub_sl_rdms_lt = []
    for i in (range(all_sub_slrdm.shape[0])):
        sl_rdms_lt = []
        for j in (range(all_sub_slrdm.shape[1])):
            dep_var = get_uppertriangular(all_sub_slrdm[i,j,:,:])
            sl_rdms_lt.append(dep_var)
        sl_rdms_lt = np.array(sl_rdms_lt)
        all_sub_sl_rdms_lt.append(sl_rdms_lt)
    all_sub_sl_rdms_lt = np.array(all_sub_sl_rdms_lt)
    return all_sub_sl_rdms_lt

def get_fMRI_RDMs_per_subject_lt(fMRI_RDMs_dir,rois):
    """Returns fMRI ROI RDMs.

    Parameters
    ----------
    fMRI_RDMs_dir : string
        directory where ROI rdms are stored.
    rois : list
        list of ROIs from atlas.

    Returns
    -------
    dict
        Dictionary containing ROI RDMs with ROI as key and RDM as value.

    """
    fmri_rdms_mat = {}
    for roi in rois:
        rdm_file_name = os.path.join(fMRI_RDMs_dir, roi + ".mat")
        all_subjects_rdm = sio.loadmat(rdm_file_name)['subject_rdm']
        num_subjects = all_subjects_rdm.shape[0]
        all_subjects_rdm_lt = []
        for i in range(num_subjects):
            all_subjects_rdm_lt.append(get_uppertriangular(all_subjects_rdm[i,:,:]))
        fmri_rdms_mat[roi] = np.array(all_subjects_rdm_lt)

    return fmri_rdms_mat

def get_taskonomy_RDMs_all_blocks_lt(taskonomy_RDM_dir,layers,task_list):
    """Return taskonomy DNN RDMs.

    Parameters
    ----------
    taskonomy_RDM_dir : string
        path to directory containing DNN RDMs.
    layers : list of strings
        List of layers to get corresponding layer RDMs.
    task_list : list of strings
        List of tasks to get corresponding DNN RDMs.

    Returns
    -------
    dict
        Dictionary containing DNN RDMs corresponding to each task.

    """
    # Initialize an empty list for each task
    taskonomy_rdms= {}
    for task in task_list:
        taskonomy_rdms[task] = []

    # Loading layers RDMs and appending to each task list
    for layer in layers:
        rdm_path_list = glob.glob(taskonomy_RDM_dir+"/*" +layer + ".mat")
        rdm_path_list.sort()
        for task in task_list:
            for rdm_path in rdm_path_list:
                if not rdm_path.find(task)==-1:
                    temp_rdm = sio.loadmat(rdm_path)['rdm']
                    taskonomy_rdms[task].append(get_uppertriangular(temp_rdm))
                    print(task,layer,rdm_path,rdm_path.find(task))

    return taskonomy_rdms
