# dnn-to-brain-function
<a href="https://sites.google.com/view/dnn2brainfunction/home"> Unveiling functions of the visual cortex using task-specific deep neural networks
</a><br/>
Kshitij Dwivedi, Michael F. Bonner, Radoslaw Martin Cichy, Gemma Roig <br/>
(Under submission)<br/><br/>


Here we provide the code to reproduce our key results from the paper. 

We generate a functional map of the visual cortex following the below steps:

* Extract activations from multiple DNNs and responses of a cortical region for all the images in the stimulus set

* Create representational dissimilarity matrices (RDMs) by computing pairwise distance between DNN activations (or fMRI responses) of all the images

* Predicting fMRI RDM from DNN RDM using a linear regression

* Highlighting the cortical region by color code corresponding to the best predicting DNN  


<div align="center">
  <img width="80%" alt="DNN-fMRI comparison" src="https://github.com/cvai-repo/dnn2brain-function/blob/main/figures/methods-vid.gif">
</div>

## Setup
* Install anaconda
* Clone the repository ```git clone https://github.com/kshitijd20/dnn-to-brain-function```
* Change working directory ```cd dnn-to-brain-function```
* Add conda channels: 
    * ```conda config --append channels conda-forge```
    * ```conda config --append channels pytorch```
    * ```conda config --append channels default```
* Run ```conda create --name dnn2brain --file requirements.txt``` to setup a new conda environment with required libraries
* Activate environment ```conda activate dnn2brain``` 
* Install <a href="https://nilearn.github.io/introduction.html#installation">nilearn </a> and <a href="https://github.com/tqdm/tqdm">tqdm </a>
* Download the data (searchlight and ROI RDMs) from this <a href="https://www.dropbox.com/s/hehc4h8qale0lo9/data.zip?dl=0">link </a> , and save it in the project root directory (./) 

## Requirements
RAM: 16 GB, NVIDIA-GPU

## Generate results
* Run ```python Fig1_individualdnns_searchlight.py``` to generate searchlight results using individual taskonomy DNNs. After the code is successfully run, a new tab will open in default browser displaying an interactive functional map. The functional map should match with Figure 1d of the paper shown below. 


<div align="center">
  <img width="80%" alt="DNN-fMRI comparison" src="https://github.com/cvai-repo/dnn2brain-function/blob/main/figures/Figure1.png">
</div>

<br/><br/>
* Run ```python Fig2_individualdnns_top3_ROIs.py``` to generate ROI results using best predicting individual taskonomy DNNs. The ROI plots should match with Figure 2b of the paper shown below.


<div align="center">
  <img width="80%" alt="DNN-fMRI comparison" src="https://github.com/cvai-repo/dnn2brain-function/blob/main/figures/Figure2.png">
</div>

<br/><br/>

* Run ```python Fig3a_groupeddnns_searchlight.py``` to generate searchlight results using grouped taskonomy DNNs (2D, 3D, and semantic). After the code is successfully run, a new tab will open in default browser displaying an interactive functional map. The functional map should match with Figure3a of the paper shown below
* Run ```python Fig3b_groupeddnns_ROIs.py``` to generate ROI results using grouped taskonomy DNNs (2D, 3D, and semantic). The ROI plots should match with Figure3b of the paper shown below.


<div align="center">
  <img width="80%" alt="DNN-fMRI comparison" src="https://github.com/cvai-repo/dnn2brain-function/blob/main/figures/Figure3.png">
</div>

## Acknowledgement
Some parts of the code are borrowed from <a href="https://brainiak.org/">Brainiak toolbox</a> 

## Cite

If you use our code, partly or as is,  please cite the paper below

```
To be updated

```

