# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 13:07:06 2020

@author: kshitij
"""
import numpy as np
import time
import torch
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection
from library.rdm_loader import get_uppertriangular
from tqdm import tqdm
from contextlib import contextmanager
from numpy.random import permutation

@contextmanager
def timeit_context(name):
    startTime = time.time()
    yield
    elapsedTime = time.time() - startTime
    print('[{}] finished in {} ms'.format(name, int(elapsedTime * 1000)))

def perform_ttest(model_rdms,fmri_rdm,analysis_type,function):
    """performs ttest to compare model with fMRI data.

    Parameters
    ----------
    model_rdms : list
        list of model rdms.
    fmri_rdm :  np.array
        fMRI RDM ROI/searchlight.
    analysis_type : string
        ROI or searchlight..
    function : function
        the comparison function that compares fMRI RDM with model RDMs.

    Returns
    -------
    tuple
        variances, pvalues, and standard deviation

    """
    # Getting number of subjects and models
    num_subjects = fmri_rdm.shape[0]
    num_models = len(model_rdms)

    individual_variances_all_subjects = []
    total_variances_all_subjects = []

    # Calculating individual variances explained for each subject
    for subject in range(num_subjects):
        individual_variances, total_variance = function(model_rdms,fmri_rdm[subject,:])
        individual_variances_all_subjects.append(individual_variances)
        total_variances_all_subjects.append(total_variance)

    # Converting list to numpy arrrays
    individual_variances_all_subjects = np.array(individual_variances_all_subjects)
    individual_variances_mean = np.mean(individual_variances_all_subjects,axis = 0)
    total_variances_all_subjects = np.array(total_variances_all_subjects)
    #total_variances_mean = np.mean(total_variances_all_subjects,axis = 0)
    print("shape of individual_variances_all_subjects", individual_variances_all_subjects.shape)
    #print("shape of total_variances_all_subjects", total_variances_all_subjects.shape)
    #print("total_variances_max ", total_variances_mean.max())
    if analysis_type == 'roi':
        # initalizing p-values, std, and pvalues of difference between model variances
        p_values = np.zeros(num_models)
        error_bars = np.zeros(num_models)
        p_values_diff = np.zeros((num_models,num_models))

        # for all models calculate pvalues and std
        for i in range(num_models):
            _,two_tailed_pvalues = stats.ttest_1samp(individual_variances_all_subjects[:,i,0],0.0)
            print("pvalues are ", two_tailed_pvalues)
            #print("individual variances ", individual_variances_all_subjects[:,i,0])
            p_values[i] = two_tailed_pvalues
            error_bars[i] = np.std(individual_variances_all_subjects[:,i,0])
            # for all comparison between models calculate p value of difference
            for j in range(num_models):
                _,p_values_diff[i,j] = stats.ttest_ind(individual_variances_all_subjects[:,i,0]\
                                                ,individual_variances_all_subjects[:,j,0])

        #FDR correction of p-values and pvalues of difference
        rejected, p_values = fdrcorrection(p_values)
        p_values_diff_1D = p_values_diff[np.triu_indices(num_models,1)]
        _ , corrected_p_values_diff_1D = fdrcorrection(p_values_diff_1D)
        p_values_diff[np.triu_indices(num_models,1)] = corrected_p_values_diff_1D
        p_values_diff[np.tril_indices(num_models,-1)] = p_values_diff.T[np.tril_indices(num_models,-1)]

        return individual_variances_mean,error_bars,p_values,p_values_diff,total_variances_all_subjects

    elif analysis_type == 'searchlight':
        num_sl_spheres = fmri_rdm.shape[1]

        # initalizing p-values, and pvalues of difference between model variances
        p_values = np.zeros((num_models,num_sl_spheres))
        p_values_diff = np.zeros((num_models,num_models,num_sl_spheres))
        print("individual_variances_mean are ",individual_variances_mean.min(),\
              individual_variances_mean.max(),\
              individual_variances_mean.mean())
        # for all models calculate pvalues and pvalues of difference
        for i in range(num_models):
            _,two_tailed_pvalues = stats.ttest_1samp(individual_variances_all_subjects[:,i,:],0.0,axis=0)
            print("pvalues are ", two_tailed_pvalues.shape,\
                  two_tailed_pvalues.max(),two_tailed_pvalues.min(),\
                      np.count_nonzero(~np.isnan(two_tailed_pvalues)))
            p_values[i,:] = two_tailed_pvalues
            for j in range(num_models):
                _,p_values_diff[i,j,:] = stats.ttest_ind(individual_variances_all_subjects[:,i,:]\
                                                ,individual_variances_all_subjects[:,j,:],axis=0)
                print("pvalues diff are ", p_values_diff[i,j,:].shape,\
                   p_values_diff[i,j,:].max(),p_values_diff[i,j,:].min(),\
                       np.count_nonzero(~np.isnan(p_values_diff[i,j,:])) )
        #FDR correction of p-values and pvalues of difference
        p_values_diff[np.isnan(p_values_diff)] = 0.1
        p_values[np.isnan(p_values)] = 0.1
        for s in range(num_sl_spheres):
            rejected, p_values[:,s] = fdrcorrection(p_values[:,s])
            p_values_diff_1D = p_values_diff[:,:,s][np.triu_indices(num_models,1)]
            _ , corrected_p_values_diff_1D = fdrcorrection(p_values_diff_1D)
            p_values_diff[:,:,s][np.triu_indices(num_models,1)] = corrected_p_values_diff_1D
            p_values_diff[:,:,s][np.tril_indices(num_models,-1)] = p_values_diff[:,:,s].T[np.tril_indices(num_models,-1)]

        return individual_variances_mean,p_values,p_values_diff,total_variances_all_subjects


def get_permuted_fmri_rdm(fmri_rdm,indices):
    num_stimuli = 50
    permuted_rdm = []
    if isinstance(fmri_rdm, np.ndarray):
        fmri_rdm = fmri_rdm[0,:]
    else:
        fmri_rdm = fmri_rdm[0,:].cpu().numpy()
    fmri_rdm_full = np.zeros((num_stimuli,num_stimuli))
    fmri_rdm_full[np.triu_indices(num_stimuli,1)] = fmri_rdm
    fmri_rdm_full[np.tril_indices(num_stimuli,-1)] = fmri_rdm_full.T[np.tril_indices(num_stimuli,-1)]
    fmri_rdm_full = fmri_rdm_full[indices[:, None],indices]
    fmri_rdm_full_lt = get_uppertriangular(fmri_rdm_full)
    if isinstance(fmri_rdm, np.ndarray):
        return fmri_rdm_full_lt
    else:
        fmri_rdm_full_lt = torch.from_numpy(fmri_rdm_full_lt).float().cuda()
        return fmri_rdm_full_lt


def get_permuted_layer_rdm(layer_rdm,indices):
    """Given indices returns RDM corresponding to selected indices.

    Parameters
    ----------
    layer_rdm : np.arrray
        Input RDM vectorized form
    indices : np.array
        selected conditions

    Returns
    -------
    np.array
        RDM with selected conditions.

    """
    num_stimuli = 50
    permuted_rdm = []
    if isinstance(layer_rdm, np.ndarray):
        layer_rdm = layer_rdm
    else:
        layer_rdm = layer_rdm.cpu().numpy()
    rdm_full = np.zeros((num_stimuli,num_stimuli))
    rdm_full[np.triu_indices(num_stimuli,1)] = layer_rdm
    rdm_full[np.tril_indices(num_stimuli,-1)] = rdm_full.T[np.tril_indices(num_stimuli,-1)]
    rdm_full = rdm_full[indices[:, None],indices]
    rdm_full_lt = get_uppertriangular(rdm_full)
    if isinstance(layer_rdm, np.ndarray):
        return rdm_full_lt
    else:
        rdm_full_lt = torch.from_numpy(rdm_full_lt).float().cuda()
        return rdm_full_lt

def get_errobars_bootstrap_stimuli(model_rdms,fmri_rdm,function,\
                                   num_permutations = 10000,\
                                   bootstrap_ratio = 0.9):
    """Errorbars using bootstrap.

    Parameters
    ----------
    model_rdms : list
        list of model rdms.
    fmri_rdm :  np.array
        fMRI RDM ROI/searchlight.
    function : function
        the comparison function that compares fMRI RDM with model RDMs.

    Returns
    -------
    np.array
        Standard deviation of the bootstrap distribution.

    """

    total_cndns = 50
    # number of conditions to bootstrap e.g. 0.9x50 = 45
    bootstrap_cndns = int(bootstrap_ratio*total_cndns)

    # creating list of selected indices for bootstrap
    indices = []
    for i in range(num_permutations):
        indices.append(np.random.choice(range(total_cndns), int(bootstrap_cndns), replace=False))

    num_dnns = len(model_rdms)
    individual_variances_all_permutations = []
    # looping over all bootstrap iterations
    for i in tqdm(range(num_permutations)):
        #print("fmri_rdm rdm shape is ", fmri_rdm.shape)
        # Creating permuted fMRI RDM
        temp_fmri_rdm = get_permuted_fmri_rdm(fmri_rdm,indices[i])
        temp_model_rdms = []
        # Creating permuted RDMs for each layer of each model
        for model_rdm in model_rdms:
            #print("model rdm length is ", len(model_rdm))
            layer_rdms = []
            for layer_rdm in model_rdm:
                #print("layer rdm shape is ", layer_rdm.shape)
                temp_layer_rdm = get_permuted_layer_rdm(layer_rdm,indices[i])

                layer_rdms.append(temp_layer_rdm)
            temp_model_rdms.append(layer_rdms)
        temp_model_rdms = np.array(temp_model_rdms)
        individual_variances_permuted,_ = function(temp_model_rdms,temp_fmri_rdm)

        #print(individual_variances.shape,correlations.shape,correlations[:,i].shape)
        individual_variances_all_permutations.append(individual_variances_permuted)
    individual_variances_all_permutations = np.array(individual_variances_all_permutations)
    #print(individual_variances_all_permutations.shape)
    error_bars = np.std(individual_variances_all_permutations, axis=0, ddof=1)
    #print(error_bars.shape)
    return error_bars


def get_pvalues_bootstrap(model_rdms,fmri_rdm,function,\
                          individual_variances,\
                          num_permutations = 10000,\
                          FDR_threshold = 0.05):
    """returns bootstrap pvalues of model and fMRI RDM comparison.

    Parameters
    ----------
    model_rdms : list
        list of model rdms.
    fmri_rdm :  np.array
        fMRI RDM ROI/searchlight.
    function : function
        the comparison function that compares fMRI RDM with model RDMs.
    individual_variances :  np.array
        variance explained by models individually.

    Returns
    -------
    tuple
        pvalues, and pvalues of difference
    """
    num_models = len(model_rdms)
    num_sl_spheres = fmri_rdm.shape[0]

    # True values
    individual_variances_true = individual_variances
    #print(individual_variances_true.shape)

    # initializing difference in variances
    individual_variances_diff_true = np.zeros((num_models,num_models,num_sl_spheres))
    individual_variances_diff_fake = np.zeros((num_models,num_models,num_sl_spheres))

    # Initializing count where randomly permuted RDMs are better predicted than fMRI RDM
    count_all_task =np.zeros((num_models,num_sl_spheres))
    count_diff = np.zeros((num_models,num_models,num_sl_spheres))

    # Initializing pvalues, and pvalues of difference
    p_values_individual_variances = np.zeros((num_models,num_sl_spheres))
    p_values_diff = np.zeros((num_models,num_models,num_sl_spheres))

    # Calcuating true difference in explained variance by different models
    for dnn1,dnn_rdm1 in enumerate(model_rdms):
        for dnn2,dnn_rdm2 in enumerate(model_rdms):
            individual_variances_diff_true[dnn1,dnn2,:] = individual_variances_true[dnn1,:]-individual_variances_true[dnn2,:]

    # Creating a list of permuted indices
    indices_perm = []
    for i in range(num_permutations):
        rdm_dimension = int(fmri_rdm.shape[1])
        lt_condn_permuted = permutation(rdm_dimension)
        indices_perm.append(lt_condn_permuted)
    # For loop to compare true variance values with variance values on randomly permuted RDM
    for i in tqdm(range(num_permutations)):
        # Creating a randomly permuted fMRI RDM
        dep_var = fmri_rdm[:,indices_perm[i]]
        #with timeit_context('RSA function'):
        # variance of permuted RDM explained by models
        individual_variances_fake,_ = function(model_rdms,dep_var)
        # if variance_permuted > variance_true increase count by 1
        count_all_task[individual_variances_fake>=individual_variances_true]+=1
        #with timeit_context('for loop for difference in significance'):

        # Compare whether difference in variance of true RDM explained  is greater
        # than difference in variance of permuted RDM by two models
        for dnn1,dnn_rdm1 in enumerate(model_rdms):
                for dnn2,dnn_rdm2 in enumerate(model_rdms):
                    individual_variances_diff_fake[dnn1,dnn2,:] = individual_variances_fake[dnn1,:]-individual_variances_fake[dnn2,:]
                    count_diff[dnn1,dnn2,:][(individual_variances_diff_true[dnn1,dnn2,:]>0) & (individual_variances_diff_fake[dnn1,dnn2,:]>=individual_variances_diff_true[dnn1,dnn2,:])] +=1
                    count_diff[dnn1,dnn2,:][(individual_variances_diff_true[dnn1,dnn2,:]<0) & (individual_variances_diff_fake[dnn1,dnn2,:]<=individual_variances_diff_true[dnn1,dnn2,:])] +=1

    # Calculating p-values as percentiles: for how many times permuted RDM better
    # predicted than true RDM
    p_values_individual_variances = (count_all_task+1.0)/(num_permutations+1.0)
    p_values_diff = (count_diff+1.0)/(num_permutations+1.0)

    # applying FDR correction
    for s in range(num_sl_spheres):
        rejected, p_values_individual_variances[:,s] = fdrcorrection(p_values_individual_variances[:,s])
        p_values_diff_1D = p_values_diff[:,:,s][np.triu_indices(len(model_rdms),1)]
        _ , corrected_p_values_diff_1D = fdrcorrection(p_values_diff_1D)
        p_values_diff[:,:,s][np.triu_indices(len(model_rdms),1)] = corrected_p_values_diff_1D
        p_values_diff[:,:,s][np.tril_indices(len(model_rdms),-1)] = p_values_diff[:,:,s].T[np.tril_indices(len(model_rdms),-1)]


    return p_values_individual_variances,p_values_diff

def perform_permutation_test(model_rdms,fmri_rdm,analysis_type,function,\
                             num_permutations = 10000,\
                             bootstrap_ratio = 0.9,\
                             FDR_threshold = 0.05):
    """performs permutation test to compare model with fMRI data.

    Parameters
    ----------
    model_rdms : list
        list of model rdms.
    fmri_rdm :  np.array
        fMRI RDM ROI/searchlight.
    analysis_type : string
        ROI or searchlight..
    function : function
        the comparison function that compares fMRI RDM with model RDMs.

    Returns
    -------
    tuple
        variances, pvalues, and standard deviation

    """
    num_models = len(model_rdms)

    if len(fmri_rdm.shape)==1:
        fmri_rdm = np.expand_dims(fmri_rdm,axis=0)
    print("Function is ", function)
    individual_variances, total_variance = function(model_rdms,fmri_rdm)
    print("shape of individual_variances_averaged_subjects", individual_variances.shape)

    if analysis_type == 'roi':

        error_bars = get_errobars_bootstrap_stimuli(model_rdms,fmri_rdm,function,\
                                                     num_permutations = num_permutations,\
                                                     bootstrap_ratio = bootstrap_ratio)
        p_values,p_values_diff = get_pvalues_bootstrap(model_rdms,fmri_rdm,function,\
                                                       individual_variances,\
                                                       num_permutations = num_permutations,\
                                                       FDR_threshold = FDR_threshold)

        return individual_variances,error_bars,p_values,p_values_diff,total_variance
    elif analysis_type == 'searchlight':

        p_values,p_values_diff = get_pvalues_bootstrap(model_rdms,fmri_rdm,function,\
                                                       individual_variances,\
                                                       num_permutations = num_permutations,\
                                                       FDR_threshold = FDR_threshold)

        return individual_variances,p_values,p_values_diff,total_variance

    return 0
