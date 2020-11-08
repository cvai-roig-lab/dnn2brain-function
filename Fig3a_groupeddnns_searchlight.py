import os
import argparse
import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from library.comparison_class import variance_partitioning
from library.rdm_loader import get_taskonomy_RDMs_all_blocks_lt, get_searchlight_rdms_persub
from library.searchlight_utils import *
from statsmodels.stats.multitest import fdrcorrection
from nilearn import plotting

def get_grouped_rdms(task_list_nogeometry,taskonomy_RDM_dir,layers):
    tasks_2D =['autoencoder', 'colorization','denoise', 'edge2d',  \
                'inpainting_whole','keypoint2d', 'segment2d']
    tasks_3D = ['curvature', 'edge3d','reshade', 'rgb2depth', \
                'rgb2mist', 'rgb2sfnorm','segment25d','keypoint3d']
    tasks_semantic = ['class_1000', 'class_places','segmentsemantic']
    taskonomy_rdms = get_taskonomy_RDMs_all_blocks_lt(taskonomy_RDM_dir,layers,task_list_nogeometry)
    individual_rdms = []
    rdms_2d = np.zeros((len(layers),1225))
    rdms_3d = np.zeros((len(layers),1225))
    rdms_sem = np.zeros((len(layers),1225))
    individual_indices = []
    iterator = 0
    for key,value in taskonomy_rdms.items():
        print(key,len(value))
        individual_rdms.append(value)
        individual_indices.append(range(iterator,iterator+len(value)))
        iterator = iterator + len(value)
        if key in tasks_2D:
            rdms_2d = rdms_2d + np.array(value)
        elif key in tasks_3D:
            rdms_3d = rdms_3d + np.array(value)
        elif key in tasks_semantic:
            rdms_sem = rdms_sem + np.array(value)
    print(individual_indices)
    grouped_rdms = [list(rdms_2d/float(len(tasks_2D))),\
                    list(rdms_3d/float(len(tasks_3D))),\
                    list(rdms_sem/float(len(tasks_semantic)))]
    return grouped_rdms

def apply_fdr_correction_searchlight(p_values, lnc_sl, nc_threshold):
    num_tasks = p_values.shape[0]
    for t in range(num_tasks):
        masked_pvalues = p_values.T[:,t][lnc_sl>nc_threshold]
        print(masked_pvalues.shape)
        _ , masked_pvalues_corrected = fdrcorrection(masked_pvalues)
        p_values.T[:,t][lnc_sl>nc_threshold] = masked_pvalues_corrected
    return p_values

def apply_fdr_correction_searchlight_diff(p_values_diff, individual_variances, \
                                          lnc_sl, nc_threshold):
    individual_variances_unique = individual_variances[:3,:]
    uvar_mask = np.argmax(individual_variances_unique, axis=0).astype(np.float)
    p_values_diff_masked = np.zeros(individual_variances_unique.shape[1])
    for s in range(uvar_mask.shape[0]):
        max_id = int(uvar_mask[s])
        p_values_diff_masked[s] = p_values_diff[max_id,:,s].max()
    masked_pvalues = p_values_diff_masked[lnc_sl>nc_threshold]
    print(masked_pvalues.shape)
    _ , masked_pvalues_corrected = fdrcorrection(masked_pvalues)
    p_values_diff_masked[lnc_sl>nc_threshold] = masked_pvalues_corrected
    return  p_values_diff_masked

def save_searchlight_nii(brain_mask, data, save_path,\
                         sl_rad = 1, max_blk_edge = 2):
    img = nib.load(brain_mask)
    functional_map =  blocks2sl(brain_mask,data,sl_rad,max_blk_edge)
    task_specificity_map_nii = nib.Nifti1Image(functional_map, img.affine, img.header)
    nib.save(task_specificity_map_nii, save_path)

def display_3D_plot_grouped_DNNs(individual_variances,p_values_diff_masked,\
                                p_values_individual_variances,result_dir,\
                                brain_mask,lnc_sl, nc_threshold):
    individual_variances_unique = individual_variances[:3,:]
    uvar_mask = np.argmax(individual_variances_unique, axis=0).astype(np.float)
    for s in range(uvar_mask.shape[0]):
        max_id = int(uvar_mask[s])
        if  lnc_sl[s]<nc_threshold or p_values_diff_masked[s]>0.05\
            or p_values_individual_variances[max_id,s]>0.05:
            uvar_mask[s] = -0.75
    uvar_mask[uvar_mask==0] = -0.25
    uvar_mask[uvar_mask==1] = 0.25
    uvar_mask[uvar_mask==2] = 0.75

    nii_save_path =  os.path.join(result_dir,'grouped_task_specificity_map.nii')
    save_searchlight_nii(brain_mask, uvar_mask, nii_save_path)
    colors = [(0.8,0.8,0.8),(0.0,0.0,1.0),(0.0,1.0,0.0), (1.0,0.0,1.0)]
    mycmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
    plt.close('all')
    view = plotting.view_img_on_surf(nii_save_path, threshold=None, surf_mesh='fsaverage',\
                                    vmax=0.75, vmin=-0.75, symmetric_cmap=False,cmap=mycmap,\
                                    title = 'Functional map: grouped DNNs ')
    view.open_in_browser()

def main():
    parser = argparse.ArgumentParser(description='searchlight analysis with tasknomy grouped RDMs')
    parser.add_argument('--fMRI_RDMs_dir', help='fMRI_RDMs_dir',\
                        default = "D:/Projects/DNN_func2_brain_func/quick_notebooks/subject_searchlight_rdms_pearson", type=str)
    parser.add_argument('--DNN_RDM_dir', help='DNN_RDM_dir', \
                        default = "D:/Projects/DNN_func2_brain_func/RDM_taskonomy_bonner50", type=str)
    parser.add_argument('--results_dir', help='results_dir', \
                        default = "./results/grouped_DNNS_searchlight/", type=str)
    parser.add_argument('-np','--num_perm', help=' number of permutations to select for bootstrap',\
                        default = 10, type=int)
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
    brain_mask = 'D:/Projects/DNN_func2_brain_func/fMRI_data/mask.nii'

    # result directory
    rsa_result_dir = os.path.join(args['results_dir'],'vpart','-'.join(layers))
    if not os.path.exists(rsa_result_dir):
        os.makedirs(rsa_result_dir)
    result_file_name = os.path.join(rsa_result_dir,'sl_'+ stats_type +'.pkl')

    if not os.path.exists(result_file_name):
        # taskonomy grouped RDMs
        grouped_rdms = get_grouped_rdms(task_list_nogeometry,taskonomy_RDM_dir,layers)
        searchlight_rdms_persub = get_searchlight_rdms_persub(fMRI_RDMs_dir,sl_rad = sl_rad,max_blk_edge=max_blk_edge)
        vpart = variance_partitioning(grouped_rdms,searchlight_rdms_persub,'searchlight',stats_type)
        result = vpart.get_unique_variance(num_permutations=num_perm)
        result_list = [result]
        with open(result_file_name, 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump(result_list, f)
    else:
        with open(result_file_name, 'rb') as f:  # Python 3: open(..., 'rb')
            result = pickle.load(f)
            result = result[0]

    individual_variances, p_values_individual_variances,p_values_diff,total_variance = result

    sl_dir = 'D:/Projects/DNN_func2_brain_func/quick_notebooks/subject_searchlight_rdms_pearson'
    nc_dir = os.path.join(sl_dir,'sl_rad=' + str(sl_rad) + '__max_blk_edge=' + str(max_blk_edge))
    nc_result_file = os.path.join(nc_dir,'noise_ceiling.pkl')
    with open(nc_result_file, 'rb') as f:  # Python 3: open(..., 'rb')
        nc_result = pickle.load(f)
        lnc_sl, unc_sl = nc_result

    corrected_p_values = apply_fdr_correction_searchlight(p_values_individual_variances, lnc_sl, nc_threshold)
    print("shape of p values :", corrected_p_values.shape)
    corrected_p_values_diff = apply_fdr_correction_searchlight_diff(p_values_diff,individual_variances,\
                                                                    lnc_sl, nc_threshold)
    display_3D_plot_grouped_DNNs(individual_variances,corrected_p_values_diff,\
                                corrected_p_values,rsa_result_dir,\
                                brain_mask,lnc_sl,nc_threshold)
        
if __name__ == "__main__":
    main()

