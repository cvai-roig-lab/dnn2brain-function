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
    num_conditions = rdm.shape[0]
    return rdm[np.triu_indices(num_conditions,1)]

def get_searchlight_rdms(fMRI_RDMs_dir,sl_rad = 1,max_blk_edge=2):
    sl_dir = os.path.join(fMRI_RDMs_dir,'sl_rad=' + str(sl_rad) + '__max_blk_edge=' + str(max_blk_edge))
    
    subj_averaged_slrdm_file = os.path.join(sl_dir,'subj_averaged_slrdm.npy')
    subj_averaged_slrdm = np.load(subj_averaged_slrdm_file)
    print(subj_averaged_slrdm.shape)
    sl_rdms_lt = []
    for i in (range(subj_averaged_slrdm.shape[0])):
        dep_var = get_uppertriangular(subj_averaged_slrdm[i,:,:])
        sl_rdms_lt.append(dep_var)
    sl_rdms_lt = np.array(sl_rdms_lt)
    return sl_rdms_lt

def get_searchlight_rdms_persub(fMRI_RDMs_dir,sl_rad = 1,max_blk_edge=2):
    sl_dir = os.path.join(fMRI_RDMs_dir,'sl_rad=' + str(sl_rad) + '__max_blk_edge=' + str(max_blk_edge))
    
    all_sub_slrdm_file = os.path.join(sl_dir,'all_sub_slrdms.npy')
    all_sub_slrdm = np.load(all_sub_slrdm_file)
    print(all_sub_slrdm.shape)
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


def get_best_layer_RDM(fmri_rdms, sc_rdms, sp_rdms):
    rois = fmri_rdms.keys()
    layers_sc = sc_rdms.keys()
    layers_sp = sp_rdms.keys()

    max_data_sc = {}
    max_data_sp = {}
    for roi in rois:
        max_corr_sc = 0
        max_corr_sp = 0
        max_data_sc[roi] = {}
        max_data_sp[roi] = {}
        fmri_rdm = get_uppertriangular(fmri_rdms[roi])
        for layer_sc in layers_sc:
            sc_rdm = get_uppertriangular(sc_rdms[layer_sc])
            corr_sc,_ = spearmanr(fmri_rdm, sc_rdm)
            if corr_sc > max_corr_sc:
                max_corr_sc = corr_sc
                max_data_sc[roi]['layer'] = layer_sc
                max_data_sc[roi]['rdm'] = sc_rdms[layer_sc]

        for layer_sp in layers_sp:
            sp_rdm = get_uppertriangular(sp_rdms[layer_sp])
            corr_sp,_ = spearmanr(fmri_rdm, sp_rdm)
            if corr_sp > max_corr_sp:
                max_corr_sp = corr_sp
                max_data_sp[roi]['layer'] = layer_sp
                max_data_sp[roi]['rdm'] = sp_rdms[layer_sp]
        print(roi,max_data_sc[roi]['layer'],max_corr_sc)
        print(roi,max_data_sp[roi]['layer'],max_corr_sp)
    return max_data_sc, max_data_sp

def get_model_RDMs(root_dir,model_name):
    sc_RDM_dir = os.path.join(root_dir,'sc_' + model_name + '_places')
    sp_RDM_dir = os.path.join(root_dir,model_name + '_placespretrained')
    layers_sc = next(os.walk(os.path.join(sc_RDM_dir,'pearson')))[1]
    sc_rdms_mat = {}
    for layer in layers_sc:
        rdm_file_name = os.path.join(sc_RDM_dir,'pearson',layer,'rdm.mat')
        sc_rdms_mat[layer] = sio.loadmat(rdm_file_name)['rdm']

    layers_sp = next(os.walk(os.path.join(sp_RDM_dir,'pearson')))[1]
    sp_rdms_mat = {}
    for layer in layers_sp:
        rdm_file_name = os.path.join(sp_RDM_dir,'pearson',layer,'rdm.mat')
        sp_rdms_mat[layer] = sio.loadmat(rdm_file_name)['rdm']

    return sc_rdms_mat,sp_rdms_mat

def get_fMRI_RDMs(fMRI_RDMs_dir,rois):
    fmri_rdms_mat = {}
    for roi in rois:
        rdm_file_name = os.path.join(fMRI_RDMs_dir, roi + ".mat")
        fmri_rdms_mat[roi] = sio.loadmat(rdm_file_name)['rdm']

    return fmri_rdms_mat

def get_fMRI_RDMs_lt(fMRI_RDMs_dir,rois):
    fmri_rdms_mat = {}
    for roi in rois:
        rdm_file_name = os.path.join(fMRI_RDMs_dir, roi + ".mat")
        fmri_rdms_mat[roi] = get_uppertriangular(sio.loadmat(rdm_file_name)['rdm'])

    return fmri_rdms_mat

def get_fMRI_RDMs_per_subject_lt(fMRI_RDMs_dir,rois):
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

def get_indices(relevant_categories,csv_data):
    indices= []
    for category in relevant_categories:
        indices.append(np.where(csv_data['Name']==category)[0][0])

    return indices

def get_layout_RDMs(model_rdm_dir,csv_data,relevant_components):
    relevant_indices = get_indices(relevant_components,csv_data)
    layout_rdms=[]
    for i in relevant_indices:
        rdm_file_name = os.path.join(model_rdm_dir, str(i).zfill(3) + "_rdm.mat")
        temp_rdm = sio.loadmat(rdm_file_name)['rdm']
        layout_rdms.append(temp_rdm)
    return layout_rdms

def get_taskonomy_RDMs(taskonomy_RDM_dir,layer,task_list):
    rdm_path_list = glob.glob(taskonomy_RDM_dir+"/*" +layer + ".mat")
    rdm_path_list.sort()
    rdm_path_from_task_list = []
    for task in task_list:
        for rdm_path in rdm_path_list:
            if not rdm_path.find(task)==-1:
                rdm_path_from_task_list.append(rdm_path)
                print(task,rdm_path,rdm_path.find(task))
    #print(rdm_path_from_task_list)
    print(len(rdm_path_from_task_list))
    taskonomy_rdms = []
    for rdm_path in  rdm_path_from_task_list:
        temp_rdm = sio.loadmat(rdm_path)['rdm']
        taskonomy_rdms.append(temp_rdm)
    return taskonomy_rdms


def get_taskonomy_RDMs_lt(taskonomy_RDM_dir,layer,task_list):
    rdm_path_list = glob.glob(taskonomy_RDM_dir+"/*" +layer + ".mat")
    rdm_path_list.sort()
    rdm_path_from_task_list = []
    for task in task_list:
        for rdm_path in rdm_path_list:
            if not rdm_path.find(task)==-1:
                rdm_path_from_task_list.append(rdm_path)
                print(task,rdm_path,rdm_path.find(task))
    #print(rdm_path_from_task_list)
    print(len(rdm_path_from_task_list))
    taskonomy_rdms = []
    for rdm_path in  rdm_path_from_task_list:
        temp_rdm = sio.loadmat(rdm_path)['rdm']
        temp_rdm = [get_uppertriangular(temp_rdm)]
        taskonomy_rdms.append(temp_rdm)
    return taskonomy_rdms

def get_taskonomy_RDMs_all_blocks_lt(taskonomy_RDM_dir,layers,task_list):
    taskonomy_rdms= {}
    for task in task_list:
        taskonomy_rdms[task] = []
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

def get_taskonomy_task_averagedRDMs(taskonomy_RDM_dir):

    taskonomy_rdms = []
    temp_rdm = sio.loadmat(taskonomy_RDM_dir+"/RDM_averaged_2D.mat")['rdm']
    taskonomy_rdms.append(temp_rdm)
    temp_rdm = sio.loadmat(taskonomy_RDM_dir+"/RDM_averaged_3D.mat")['rdm']
    taskonomy_rdms.append(temp_rdm)
    temp_rdm = sio.loadmat(taskonomy_RDM_dir+"/RDM_averaged_semantic.mat")['rdm']
    taskonomy_rdms.append(temp_rdm)
    return taskonomy_rdms
