import numpy as np
import nibabel as nib
from scipy.stats import spearmanr
def readnii(niifile):
    img = nib.load(niifile)
    a = np.array(img.dataobj)
    return a
def get_blocks(mask,sl_rad=1, max_blk_edge=10):
    """Divide the volume into a set of blocks
    Ignore blocks that have no active voxels in the mask
    Parameters
    ----------
    mask: a boolean 3D array which is true at every active voxel
    Returns
    -------
    list of tuples containing block information:
       - a triple containing top left point of the block and
       - a triple containing the size in voxels of the block
    """
    blocks = []
    outerblk = max_blk_edge + 2*sl_rad
    for i in range(0, mask.shape[0], max_blk_edge):
        for j in range(0, mask.shape[1], max_blk_edge):
            for k in range(0, mask.shape[2], max_blk_edge):
                block_shape = mask[i:i+outerblk,
                                   j:j+outerblk,
                                   k:k+outerblk
                                   ].shape
                if np.any(
                    mask[i+sl_rad:i+block_shape[0]-sl_rad,
                         j+sl_rad:j+block_shape[1]-sl_rad,
                         k+sl_rad:k+block_shape[2]-sl_rad]):
                    blocks.append(((i, j, k), block_shape))
    return blocks
def get_block_data( mat, block):
    """Retrieve a block from a 3D or 4D volume
    Parameters
    ----------
    mat: a 3D or 4D volume
    block: a tuple containing block information:
      - a triple containing the lowest-coordinate voxel in the block
      - a triple containing the size in voxels of the block
    Returns
    -------
    In the case of a 3D array, a 3D subarray at the block location
    In the case of a 4D array, a 4D subarray at the block location,
    including the entire fourth dimension.
    """
    (pt, sz) = block
    if len(mat.shape) == 3:
        return mat[pt[0]:pt[0]+sz[0],
                   pt[1]:pt[1]+sz[1],
                   pt[2]:pt[2]+sz[2]].copy()
    elif len(mat.shape) == 4:
        return mat[pt[0]:pt[0]+sz[0],
                   pt[1]:pt[1]+sz[1],
                   pt[2]:pt[2]+sz[2],
                   :].copy()

def blocks2sl(brain_mask,sl_result,sl_rad,max_blk_edge):
    mask = readnii(brain_mask)
    blocks = get_blocks(mask,sl_rad=sl_rad,max_blk_edge=max_blk_edge)

    # Coalesce results
    outmat = np.empty(mask.shape, dtype=np.object).astype(np.float)

    for i in range(sl_result.shape[0]):
        pt = blocks[i][0]
        mat = blocks[i][1]
        coords = np.s_[
            pt[0]+sl_rad:pt[0]+sl_rad+mat[0],
            pt[1]+sl_rad:pt[1]+sl_rad+mat[1],
            pt[2]+sl_rad:pt[2]+sl_rad+mat[2]
        ]
        outmat[coords] = sl_result[i]
    return outmat

def RSA_spearman(rdm1,rdm2):
    lt_rdm1 = get_uppertriangular(rdm1)
    lt_rdm2 = get_uppertriangular(rdm2)
    return spearmanr(lt_rdm1, lt_rdm2)[0]

def get_uppertriangular(rdm):
    num_conditions = rdm.shape[0]
    return rdm[np.triu_indices(num_conditions,1)]

def get_uppernoiseceiling(rdm):
    num_subs = rdm.shape[0]
    unc = 0.0
    for i in range(num_subs):
        sub_rdm = rdm[i,:,:]
        mean_sub_rdm = np.mean(rdm,axis=0)
        unc+=RSA_spearman(sub_rdm,mean_sub_rdm)
    unc = unc/num_subs
    return unc

def get_lowernoiseceiling(rdm):
    num_subs = rdm.shape[0]
    lnc = 0.0
    for i in range(num_subs):
        sub_rdm = rdm[i,:,:]
        rdm_sub_removed = np.delete(rdm, i, axis=0)
        mean_sub_rdm = np.mean(rdm_sub_removed,axis=0)
        lnc+=RSA_spearman(sub_rdm,mean_sub_rdm)
    lnc = lnc/num_subs
    return lnc
def get_rdm_from_responses(roi_responses):
    # Pearson: rdm = 1-np.corrcoef(activation)
    # Euclidean: rdm  = euclidean_distances(activation)
    # Cosine: rdm = 1- cosine_similarity(activation)
    rdm = euclidean_distances(roi_responses)
    return rdm

def get_searchlight_rdms(sub_dir,blocks):
    condns = glob.glob(sub_dir+"/*.nii")
    condns.sort()
    condn_responses = []
    for condn in condns:
        condn_response = readnii(condn)
        condn_responses.append(condn_response)
    blocks_rdms = []
    for block in blocks:
        block_data_all_condns = []
        for condn_response in condn_responses:
            block_data = get_block_data(condn_response, block)
            array_sum = np.sum(block_data)
            array_has_nan = np.isnan(array_sum)
            if array_has_nan:
                print("Time to eat nan")
            block_data_all_condns.append(block_data.ravel())
        block_rdm = get_rdm_from_responses(np.array(block_data_all_condns))
        blocks_rdms.append(block_rdm)
    return blocks_rdms
