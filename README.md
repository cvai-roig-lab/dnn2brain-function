# dnn-to-brain-function
<a href="https://arxiv.org/abs/2008.02107"> Unveiling functions of the visual cortex using task-specific deep neural networks
</a><br/>
Kshitij Dwivedi, Michael F. Bonner, Radoslaw Martin Cichy, Gemma Roig <br/>
Journal X 2021<br/><br/>


Here we provide the code to reproduce our key results from the paper. 
We generate a functional map of visual cortex in following steps:

* Extract activations from multiple DNNs and responses of a cortical region for all the images in the stimulus set

* Create representational dissimilarity matrices (RDMs) by computing pairwise distance between DNN activations (or fMRI responses) of all the images

* Predicting fMRI RDM from DNN RDM using a linear regression

* Highlighting the cortical region by color code corresponding to the best predicting DNN  

* To assess similarity between two tasks, we extract the features of the Deep Neural Networks(DNNs) trained on these tasks


<div align="center">
  <img width="80%" alt="DNN-fMRI comparison" src="https://github.com/kshitijd20/dnn-to-brain-function/blob/main/figures/methods-vid.gif">
</div>


<br/><br/>


## Setup
* Install anaconda
* Run ```conda create --name dnn2brain --file requirements.txt``` to setup a new conda environment with required libraries
* Activate environment ```conda activate dnn2brain```
* Download the data (searchlight and ROI RDMs) from this <a href="https://www.dropbox.com/s/hehc4h8qale0lo9/data.zip?dl=0">link </a> , and save it in the root directory (./) 

## Generate results
* Run ```python Fig1_individualdnns_searchlight.py``` to generate searchlight results using individual taskonomy DNNs. After the code is successfully run, a new tab will open in default browser displaying an interactive functional map.
* Run ```python Fig2_individualdnns_top3_ROIs.py``` to generate ROI results using best predicting individual taskonomy DNNs.
* Run ```python Fig3a_groupeddnns_searchlight``` to generate searchlight results using grouped taskonomy DNNs (2D, 3D, and semantic). After the code is successfully run, a new tab will open in default browser displaying an interactive functional map.
* Run ```python Fig3b_groupeddnns_ROIs.py``` to generate ROI results using grouped taskonomy DNNs (2D, 3D, and semantic).

## Cite

If you use our code please consider citing the paper below

```
@inproceedings{dwivedi2020DDS,
  title={Duality Diagram Similarity: a generic framework for initialization
               selection in task transfer learning},
  author={Kshitij Dwivedi and
               Jiahui Huang and
               Radoslaw Martin Cichy and
               Gemma Roig},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2020}
}

```

