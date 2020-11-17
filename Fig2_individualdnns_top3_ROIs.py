import os
import numpy as np
import argparse
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from library.comparison_class import rsa,variance_partitioning,multiple_regression
from library.rdm_loader import get_taskonomy_RDMs_all_blocks_lt,get_fMRI_RDMs_per_subject_lt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
def label_diff(ax,text,r1,r2,max_corr,yer1,yer2,ymax,barWidth):

    dx = int(abs((r1-r2))+0.1)
    y = max(max_corr+ yer2/2, max_corr+ yer1/2) + 0.1*dx*ymax
    x = r1 + dx/2.0
    lx = r1+0.1*barWidth
    rx = r1+dx*barWidth-0.1*barWidth

    barh = 0.05*ymax*dx
    barx = [lx, lx, rx, rx]
    bary = [y, y+barh, y+barh, y]
    mid = ((lx+rx)/2, y+barh)

    ax.plot(barx, bary, c='black',linewidth=0.1)
    #props = {'connectionstyle':'bar','arrowstyle':'-',\
    #             'shrinkA':0.01/dx,'shrinkB':0.05/dx,'linewidth':1}
    ax.annotate(text, xy=(x,y+ 1.2*barh ), zorder=10)
    #ax.annotate('', xy=(r1+0.1*barWidth,y+ 0.02*ymax*dx), xytext=(r1+dx*barWidth-0.1*barWidth,y+ 0.02*ymax*dx), arrowprops=props)


def label_against_zero(ax,i,text,r,bars,yer,ymax):
    x = r - 0.04
    y = bars + yer/2 + 0.1*ymax
    ax.annotate(text, xy=(x,y), zorder=10)

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

    cmaplist =cmap2D+cmap3D+cmapsem + [(.8, .80, 0, 1.0)]
    return cmaplist



def plot_top3(rois,results,top3_dnns_perROI,result_dir):
    top3dnns_roi = {}
    plt.rcParams.update({'font.size': 6})
    fig,ax = plt.subplots( nrows=4, ncols=4 , sharex=True, sharey=True)
    cmaplist = create_color_map()

    ymax = 0.45
    barWidth = 1
    count = 0

    tasks = ['autoencoder','colorization','denoise','edge2d','inpainting_whole','keypoint2d','segment2d',\
         'edge3d','keypoint3d','segment25d','reshade', 'rgb2depth', 'rgb2mist', 'rgb2sfnorm','curvature',\
         'class_1000', 'class_places', 'segmentsemantic','random']

    tasks_names =  ['autoencoding','colorization','denoising','2D edges','inpainting','2D keypoints','2D segment', \
                           '3D edges',  '3D keypoints','2.5d segment', 'reshading', 'z-depth', 'distance', 'normals','curvature',\
                            'object class', 'scene class', 'semantic segm','random' ]

    ventral_temporal = ['V1v','V2v','V3v','hV4','VO1','VO2','PHC1','PHC2']
    dorso_lateral = ['V1d','V2d','V3d','LO1','LO2','V3b','V3a',]
    parietal_frontal = ['IPS0','IPS1','IPS2','IPS3','IPS5','SPL1','FEF']
    good_rois = ventral_temporal + dorso_lateral

    print(good_rois)

    patterns = [ "////" , "\\\\\\\\" , "||||" ]
    for r,roi in enumerate(rois):
        if roi in good_rois or roi == rois[6] or roi == rois[16]:
            individual_variances_mean,error_bars,p_values,p_values_diff,total_variances_mean = results[roi]
            correlation = individual_variances_mean[:3].ravel()

            p_values = p_values.ravel()
            error_bar = error_bars.ravel()

            x=[]
            y=[]
            yers = []
            indices = range(len(correlation))
            colors = []
            legend_elements = []
            for si in indices:
                w=top3_dnns_perROI[roi][si]
                #print(tasks.index(w),"is tje task index")
                x.append(tasks_names[tasks.index(w)])
                y.append(correlation[si])
                yers.append(error_bar[si])
                colors.append(cmaplist[tasks.index(w)])
                p = mpl.patches.Patch(facecolor=cmaplist[tasks.index(w)],\
                                             hatch=patterns[si],edgecolor=[0.7,0.7,0.7],\
                                             label=tasks_names[tasks.index(w)])
                legend_elements.append(p)
                #legend_elements.append(Line2D([0], [0], color=cmaplist[tasks.index(w)], lw=4,\
                #                       hatch=patterns[si],label=tasks_names[tasks.index(w)]),\
                #                       edgecolor=[0.7,0.7,0.7])
            top3dnns_roi[roi] = x
            row = int(count/4)
            col = int(count%4)
            barlist = ax[row,col].bar(range(len(correlation)), list(y),yerr=yers,color = colors,\
                                      align='center',label = x)
            for bar, pattern in zip(barlist, patterns):
                bar.set_hatch(pattern)
                bar.set_edgecolor([0.7,0.7,0.7])
            # Plotting significant values *
            for i,si in enumerate(indices):
                if p_values[si]<0.05:
                    text = '*'
                    label_against_zero(ax[row,col],i,text,range(len(correlation))[i],y[i],yers[i],ymax)

            max_corr = max(y)
            for si1 in indices:
                for si2 in indices:
                    if si1>=si2:
                        continue
                    else:
                        if p_values_diff[si1,si2]<0.05:
                            text = '*'
                            #barplot_annotate_brackets(si1, si2, text, range(len(correlation)), list(y))

                            label_diff(ax[row,col],text,si1,si2,max_corr,yers[si1],yers[si2],ymax,barWidth)

            ax[row,col].spines["top"].set_visible(False)
            ax[row,col].spines["right"].set_visible(False)
            ax[row,col].spines["bottom"].set_position(("data",0))
            ax[row,col].set_xticks(range(len(correlation)), list(x))
            yticks = np.arange(0, 0.5, 0.1)
            ax[row,col].set_yticks(yticks)
            ax[row,col].legend(handles=legend_elements, loc=(0.55,0.72),prop={'size': 4})#'upper right')
            plt.setp(ax[row,col].get_xticklabels(), rotation=90)
            #plt.xlabel('Unique variance', axes=ax[row,col])
            #plt.ylabel('Task type', axes=ax[row,col])
            ax[row,col].set_ylim([0,ymax])
            #plt.legend(labels, ['2D', '3D','semantic','geometric'])
            ax[row,col].set_title(roi + ": $R^{2}=$" +  str(round(total_variances_mean['tv'][0],2)),x=0.25, y=0.89,size = 'small')#,loc = 'left')
            ax[row,col].tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False) # labels along the bottom edge are off
            #ax.legend(handles=legend_elements, loc=(0.50,0.72))#'upper right')
            count+=1

    ax[3,3].spines["top"].set_visible(False)
    ax[3,3].spines["right"].set_visible(False)
    ax[3,3].spines["bottom"].set_visible(False)
    ax[3,3].spines["left"].set_visible(False)
    ax[3,3].tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False)
    ax[3,3].tick_params(
            axis='y',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False)
    #ax[3,3].set_yticks([])
    fig.text(0.04, 0.5, 'Unique variance '+"$ (R^{2})$", va='center', rotation='vertical')

    plots_save_path = os.path.join(result_dir,"top3_dnns.svg")

    plt.savefig(plots_save_path, bbox_inches="tight")
    plt.show()




def get_top3rdms(roi,results,taskonomy_rdms_all_blocks,task_list_nogeometry):
    k = 3
    top3_rdms = []
    top3_dnns = []
    individual_variances_mean,error_bars,p_values,p_values_diff,total_variances_mean = results
    individual_variances_mean[np.isnan(individual_variances_mean)] = 0
    sorted_indices = np.argsort(individual_variances_mean[:,0])[::-1]
    for i in range(k):
        top3_rdms.append(taskonomy_rdms_all_blocks[task_list_nogeometry[sorted_indices[i]]])
        top3_dnns.append(task_list_nogeometry[sorted_indices[i]])

    return top3_rdms, top3_dnns

def main():
    parser = argparse.ArgumentParser(description='ROIs from sabine kastners atlas vs. tasknomy grouped RDMs')
    parser.add_argument('--fMRI_RDMs_dir', help='fMRI_RDMs_dir',\
                        default = "./data/kastner_ROIs_RDMs_pearson", type=str)
    parser.add_argument('--DNN_RDM_dir', help='DNN_RDM_dir', \
                        default = "./data/RDM_taskonomy_bonner50", type=str)
    parser.add_argument('--results_dir', help='results_dir', \
                        default = "./results/individual_DNNS_ROIs/", type=str)
    parser.add_argument('-np','--num_perm', help=' number of permutations to select for bootstrap',\
                        default = 10000, type=int)
    parser.add_argument('--roi_labels', help='roi label file path', \
                        default = "./data/ROIfiles_Labeling.txt", type=str)
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

    # Loading ROI RDMs
    roi_label_file = args['roi_labels']
    roi_labels = pd.read_csv(roi_label_file)
    print(roi_labels)
    rois=roi_labels['roi']
    roi_ids=roi_labels['id']


    # result directory
    rsa_result_dir = os.path.join(args['results_dir'],'multiple_regression','-'.join(layers))
    if not os.path.exists(rsa_result_dir):
        os.makedirs(rsa_result_dir)
    result_file_name = os.path.join(rsa_result_dir,'roi_all_'+ stats_type +'.pkl')


    # individual taskonomy DNN's RDMs
    fmri_rdms = get_fMRI_RDMs_per_subject_lt(fMRI_RDMs_dir,rois)
    taskonomy_rdms = get_taskonomy_RDMs_all_blocks_lt(taskonomy_RDM_dir,layers,task_list_nogeometry)

    # if rsa result file doesn't exists comparing all DNNs with all ROIs
    if not os.path.exists(result_file_name):
        # preparing taskonomy rdms
        individual_rdms = []
        for key,value in taskonomy_rdms.items():
            print(key,np.array(value).shape)
            individual_rdms.append(np.array(value))

        # preparing results for rois, doing regression to find best predicting DNNs
        results = {}
        for roi in rois:
            print(roi,fmri_rdms[roi].shape)
            rsa_object = multiple_regression(individual_rdms,fmri_rdms[roi],'roi',stats_type)
            result = rsa_object.get_explained_variance(num_permutations=2)
            results[roi] = result
            print("---------------------------------------------------------------------")
            print(roi)
        result_list = [rois,results]
        with open(result_file_name, 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump(result_list, f)
    else:
        with open(result_file_name, 'rb') as f:  # Python 3: open(..., 'rb')
            rois,results = pickle.load(f)


    vpart_file_name = os.path.join(rsa_result_dir,'roi_top3_'+ stats_type +'.pkl')
    if not os.path.exists(vpart_file_name):
        vpart_results = {}
        top3_dnns_perROI = {}
        # Performing variance partitioning to find unique and shared variance explained by
        # top-3 best predicting DNNS
        for roi in rois:
            print("---------------------------------------------------------------------")
            top3_rdms, top3_dnns = get_top3rdms(roi,results[roi],taskonomy_rdms,task_list_nogeometry)
            print(roi)
            print(top3_dnns)
            vpart = variance_partitioning(top3_rdms,fmri_rdms[roi],'roi',stats_type)
            result = vpart.get_unique_variance(num_permutations=num_perm)
            vpart_results[roi] = result
            top3_dnns_perROI[roi] = top3_dnns

        result_list = [rois,vpart_results,top3_dnns_perROI]
        with open(vpart_file_name, 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump(result_list, f)
    else:
        with open(vpart_file_name, 'rb') as f:  # Python 3: open(..., 'rb')
            rois,vpart_results,top3_dnns_perROI = pickle.load(f)

    # plottting the unique variance of top3 RDMs
    plot_top3(rois,vpart_results,top3_dnns_perROI,rsa_result_dir)

if __name__ == "__main__":
    main()
