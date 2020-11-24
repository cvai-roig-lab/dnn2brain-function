# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 10:06:42 2020

@author: kshitij
"""
import numpy as np
import torch
from library.comparison_methods import vpart2,vpart3,vpart_general,multimodel_rsa,return_ranks,return_ranks_sl,multiple_regression_rsa,multiple_regression_rsa_with_baseline
from library.significance_testing import perform_ttest,perform_permutation_test

class variance_partitioning:
    def __init__(self,model_rdms, fmri_rdm,\
                 analysis_type, test_type):
        self.model_rdms = model_rdms
        self.fmri_rdm = fmri_rdm
        self.test_type = test_type
        if self.test_type == 'permutation_labels':
            self.fmri_rdm =  np.mean(fmri_rdm,axis=0)
        self.analysis_type = analysis_type

        if len(self.model_rdms)==2:
            self.function = vpart2
        elif len(self.model_rdms)==3:
            self.function = vpart3
        else:
            self.function = vpart_general

    def get_unique_variance(self,num_permutations = 10000,\
                            bootstrap_ratio = 0.9, FDR_threshold = 0.05):
        if self.test_type == 't-test':
            result =  perform_ttest(self.model_rdms,self.fmri_rdm,\
                                    self.analysis_type,self.function)
        elif self.test_type == 'permutation_labels':
            result =  perform_permutation_test(self.model_rdms,self.fmri_rdm,\
                                               self.analysis_type,self.function,\
                                                num_permutations = num_permutations,\
                                                bootstrap_ratio = bootstrap_ratio,\
                                                FDR_threshold = FDR_threshold)
        return result


class rsa:
    def __init__(self,model_rdms, fmri_rdm,\
                 analysis_type, test_type):
        model_ranks_cuda = []
        for i,model_rdm in enumerate(model_rdms):
            model_rdm_ranks = return_ranks(model_rdm[0])
            model_rdm_ranks = torch.from_numpy(model_rdm_ranks).float().cuda()
            model_ranks_cuda.append(model_rdm_ranks)
        self.model_rdms = model_ranks_cuda

        self.test_type = test_type
        if len (fmri_rdm.shape)==2:
            fmri_rdm = np.expand_dims(fmri_rdm,axis=1)
        if self.test_type == 't-test':
            num_subjects = fmri_rdm.shape[0]
            self.fmri_rdm = np.zeros_like(fmri_rdm)
            for subject in range(num_subjects):
                fmri_rdm_ranks = return_ranks_sl(fmri_rdm[subject,:])
                self.fmri_rdm[subject,:] = fmri_rdm_ranks
            self.fmri_rdm = torch.from_numpy(self.fmri_rdm).float().cuda()
        elif self.test_type == 'permutation_labels':
            #Here we use subject averaged RDMs
            fmri_rdm = np.mean(fmri_rdm,axis=0)
            fmri_rdm_ranks = return_ranks_sl(fmri_rdm)
            fmri_rdm_ranks_cuda = torch.from_numpy(fmri_rdm_ranks).float().cuda()
            self.fmri_rdm = fmri_rdm_ranks_cuda
        self.analysis_type = analysis_type
        self.function = multimodel_rsa

    def get_spearman_correlation(self,num_permutations = 10000,\
                            bootstrap_ratio = 0.9, FDR_threshold = 0.05):
        if self.test_type == 't-test':
            result =  perform_ttest(self.model_rdms,self.fmri_rdm,\
                                    self.analysis_type,self.function)
        elif self.test_type == 'permutation_labels':
            result =  perform_permutation_test(self.model_rdms,self.fmri_rdm,\
                                               self.analysis_type,self.function,\
                                                num_permutations = num_permutations,\
                                                bootstrap_ratio = bootstrap_ratio,\
                                                FDR_threshold = FDR_threshold)
        return result


class multiple_regression:
    def __init__(self,model_rdms, fmri_rdm,\
                 analysis_type, test_type,baseline=None,baseline_rdm = None):
        self.model_rdms = model_rdms
        self.fmri_rdm = fmri_rdm
        self.test_type = test_type
        if self.test_type == 'permutation_labels':
            self.fmri_rdm =  np.mean(fmri_rdm,axis=0)
        self.analysis_type = analysis_type
        if baseline:
            self.function = multiple_regression_rsa_with_baseline
        else:
            self.function = multiple_regression_rsa

    def get_explained_variance(self,num_permutations = 10000,\
                            bootstrap_ratio = 0.9, FDR_threshold = 0.05):
        if self.test_type == 't-test':
            result =  perform_ttest(self.model_rdms,self.fmri_rdm,\
                                    self.analysis_type,self.function)
        elif self.test_type == 'permutation_labels':
            result =  perform_permutation_test(self.model_rdms,self.fmri_rdm,\
                                               self.analysis_type,self.function,\
                                                num_permutations = num_permutations,\
                                                bootstrap_ratio = bootstrap_ratio,\
                                                FDR_threshold = FDR_threshold)
        return result
