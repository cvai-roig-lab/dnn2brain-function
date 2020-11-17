import numpy as np
import nibabel as nib
import pickle
import glob
import os
import sys
import argparse
import pickle
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import fdrcorrection

from nilearn import plotting
from library.comparison_class import rsa,multiple_regression
from library.rdm_loader import get_taskonomy_RDMs_all_blocks_lt,get_searchlight_rdms_persub
from library.searchlight_utils import *

def create_color_map():
    cmap = plt.cm.get_cmap('tab20b')

    crange_2D = np.linspace(0.3, 1.0, num=7)
    crange_3D = np.linspace(0.3, 1.0, num=8)
    crange_sem = np.linspace(0.5, 1.0, num=3)

    cmap2D = []
    cmap3D = []
    cmapsem = []

    for color in crange_2D:
        cmap2D.append((0, 0, color, 1.0))
    for color in crange_3D:
        cmap3D.append((0, color, 0, 1.0))
    for color in crange_sem:
        cmapsem.append((color, 0, color, 1.0))

    cmaplist =[(.8, .8, .8, 1.0)]+cmap2D+cmap3D+cmapsem + [(.8, .80, 0, 1.0)]
    # create the new map
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'Custom cmap', cmaplist, cmap.N)
    return cmap


def save_searchlight_nii(brain_mask, data, save_path,\
                         sl_rad = 1, max_blk_edge = 2):
    img = nib.load(brain_mask)
    functional_map =  blocks2sl(brain_mask,data,sl_rad,max_blk_edge)
    task_specificity_map_nii = nib.Nifti1Image(functional_map, img.affine, img.header)
    nib.save(task_specificity_map_nii, save_path)

def display_3D_plot_individual_DNNs(task_list,p_values,lnc_sl,nc_threshold,\
                                    individual_variances,rsa_result_dir,brain_mask):
    # unique variance mask
    uvar_mask = np.argmax(individual_variances, axis=0).astype(np.float)

    # task list with correct names
    task_list = ['autoencoding','object class', 'scene class', 'colorization','curvature',\
             'denoising', '2D edges', '3D edges', 'inpainting','2D keypoints', '3D keypoints', \
                'reshading', 'z-depth', 'distance', 'normals', \
                '2.5d segment', '2D segment', 'semantic segm','random']

    # task list reordered according to 2D, 3D, and semantic tasks
    task_list_reordered = ['autoencoding','colorization','denoising','2D edges','inpainting','2D keypoints','2D segment', \
                           '3D edges',  '3D keypoints','2.5d segment', 'reshading', 'z-depth', 'distance', 'normals','curvature',\
                           'object class', 'scene class', 'semantic segm','random' ]
    # setting the range of unique variance mask from -8 to 9
    uvar_mask_map = uvar_mask-8
    for s in range(uvar_mask.shape[0]):
        max_id = int(uvar_mask[s])

        if lnc_sl[s]<nc_threshold or p_values[max_id,s]>0.05:
            uvar_mask_map[s] = -9
        else:
            uvar_mask_map[s] = task_list_reordered.index(task_list[max_id])-8

    nii_save_path =  os.path.join(rsa_result_dir,'rsa_task_specificity_map.nii')
    save_searchlight_nii(brain_mask, uvar_mask_map, nii_save_path)
    cmap = create_color_map()
    view = plotting.view_img_on_surf(nii_save_path, threshold=None, surf_mesh='fsaverage',\
                                vmax=10,vmin=-9,symmetric_cmap=False,cmap=cmap,\
                                title = 'Functional map: individual DNNs',colorbar=False)
    view.open_in_browser()

def apply_fdr_correction_searchlight(p_values, lnc_sl, nc_threshold):
    num_tasks = p_values.shape[0]
    for t in range(num_tasks):
        masked_pvalues = p_values.T[:,t][lnc_sl>nc_threshold]
        print(masked_pvalues.shape)
        _ , masked_pvalues_corrected = fdrcorrection(masked_pvalues)
        p_values.T[:,t][lnc_sl>nc_threshold] = masked_pvalues_corrected
    return p_values

def main():
    parser = argparse.ArgumentParser(description='ROIs from sabine kastners atlas vs. tasknomy grouped RDMs')
    parser.add_argument('--fMRI_RDMs_dir', help='fMRI_RDMs_dir',\
                        default = "./data", type=str)
    parser.add_argument('--DNN_RDM_dir', help='DNN_RDM_dir', \
                        default = "./data/RDM_taskonomy_bonner50", type=str)
    parser.add_argument('--results_dir', help='results_dir', \
                        default = "./results/individual_DNNS_searchlight/", type=str)
    parser.add_argument('--mask', help='brain mask path', \
                        default = './data/mask.nii', type=str)
    parser.add_argument('-np','--num_perm', help=' number of permutations to select for bootstrap',\
                        default = 10000, type=int)
    parser.add_argument('-stats','--stats', help=' t-test or permuting labels',\
                        default = 'permutation_labels', type=str)
    parser.add_argument('-bs_ratio','--bootstrap_ratio', help='ratio of conditions for bootstrap',\
                        default = 0.9, type=float)
    args = vars(parser.parse_args())

    # list of all tasks
    task_list_nogeometry = ['autoencoder','class_1000', 'class_places', 'colorization',\
                            'curvature', 'denoise', 'edge2d', 'edge3d', \
                            'inpainting_whole','keypoint2d', 'keypoint3d', \
                            'reshade', 'rgb2depth', 'rgb2mist', 'rgb2sfnorm', \
                            'segment25d', 'segment2d', 'segmentsemantic','random']

    # command line arguments
    layers = ['block4','encoder_output']
    fMRI_RDMs_dir = args['fMRI_RDMs_dir']
    taskonomy_RDM_dir = args['DNN_RDM_dir']
    stats_type = args['stats']
    num_perm = args['num_perm']
    bootstrap_ratio = args['bootstrap_ratio']

    # searchlight parameters
    sl_rad = 1
    max_blk_edge = 2
    nc_threshold = 0.033
    brain_mask = args['mask']

    # result directory
    rsa_result_dir = os.path.join(args['results_dir'],'multiple_regression','-'.join(layers))
    if not os.path.exists(rsa_result_dir):
        os.makedirs(rsa_result_dir)
    result_file_name = os.path.join(rsa_result_dir,'sl_'+ stats_type +'.pkl')

    # if searchlight results are not already calculated
    if not os.path.exists(result_file_name):
        # searchlight rdm
        searchlight_rdms_persub = get_searchlight_rdms_persub(fMRI_RDMs_dir,sl_rad = sl_rad,max_blk_edge=max_blk_edge)

        # individual taskonomy DNN's RDMs
        taskonomy_rdms = get_taskonomy_RDMs_all_blocks_lt(taskonomy_RDM_dir,layers,task_list_nogeometry)
        individual_rdms = []
        for key,value in taskonomy_rdms.items():
            print(key,np.array(value).shape)
            individual_rdms.append(np.array(value))

        # Performing multiple Regression RSA
        rsa_object = multiple_regression(individual_rdms,searchlight_rdms_persub,'searchlight',stats_type)
        result = rsa_object.get_explained_variance(num_permutations=num_perm, bootstrap_ratio = bootstrap_ratio)

        result_list = [result]
        with open(result_file_name, 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump(result_list, f)

    with open(result_file_name, 'rb') as f:  # Python 3: open(..., 'rb')
        result = pickle.load(f)
        individual_variances,p_values,p_values_diff,total_variance = result[0]

    # load noise ceiling
    sl_dir = fMRI_RDMs_dir
    nc_dir = os.path.join(sl_dir,'sl_rad=' + str(sl_rad) + '__max_blk_edge=' + str(max_blk_edge))
    nc_result_file = os.path.join(nc_dir,'noise_ceiling.pkl')
    with open(nc_result_file, 'rb') as f:  # Python 3: open(..., 'rb')
        nc_result = pickle.load(f)
        lnc_sl, unc_sl = nc_result

    # apply fdr correction
    corrected_p_values = apply_fdr_correction_searchlight(p_values, lnc_sl, nc_threshold)

    # display 3D interactive plot in browser
    display_3D_plot_individual_DNNs(task_list_nogeometry,corrected_p_values,lnc_sl,nc_threshold,\
                                    individual_variances,rsa_result_dir,brain_mask)

if __name__ == "__main__":
    main()
