# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 10:06:42 2020

@author: kshitij
"""
import numpy as np
import torch
from library.comparison_methods import vpart2,vpart3,vpart_general,multiple_regression_rsa
from library.significance_testing import perform_ttest,perform_permutation_test

class variance_partitioning:
    def __init__(self,model_rdms, fmri_rdm,\
                 analysis_type, test_type):
        """Initializes variance_partitioning class.

        Parameters
        ----------
        model_rdms : list
            list of model rdms.
        fmri_rdm : np.array
            fMRI RDM ROI/searchlight.
        analysis_type : string
            ROI or searchlight.
        test_type : string
            t-test or permutation.
        Returns
        -------
        None

        """
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
        """Returns R2 (unique variance).

        Parameters
        ----------
        num_permutations : int
            number of permutations for permutation test.
        bootstrap_ratio : float
            Ratio of total conditions to select for bootstrap.
        FDR_threshold : float
            threshold for FDR correction.

        Returns
        -------
        tuple
            result containing R2(unique variance) values along with p-values and std.

        """
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
                 analysis_type, test_type):
        """Initializes multiple regression class.

        Parameters
        ----------
        model_rdms : list
            list of model rdms.
        fmri_rdm : np.array
            fMRI RDM ROI/searchlight.
        analysis_type : string
            ROI or searchlight.
        test_type : string
            t-test or permutation.
        Returns
        -------
        None

        """
        self.model_rdms = model_rdms
        self.fmri_rdm = fmri_rdm
        self.test_type = test_type

        # averaging RDMs across subjects
        if self.test_type == 'permutation_labels':
            self.fmri_rdm =  np.mean(fmri_rdm,axis=0)
        self.analysis_type = analysis_type
        self.function = multiple_regression_rsa

    def get_explained_variance(self,num_permutations = 10000,\
                            bootstrap_ratio = 0.9, FDR_threshold = 0.05):
        """Returns R2 (explained variance).

        Parameters
        ----------
        num_permutations : int
            number of permutations for permutation test.
        bootstrap_ratio : float
            Ratio of total conditions to select for bootstrap.
        FDR_threshold : float
            threshold for FDR correction.

        Returns
        -------
        tuple
            result containing R2 values along with p-values and std.

        """
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
