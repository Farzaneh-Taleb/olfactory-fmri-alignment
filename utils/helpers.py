import pandas as pd
import numpy as np
import ast
import nibabel as nib
import scipy.io as sio
import os
import random
import torch
import pubchempy as pcp
from scipy.io import loadmat
import selfies as sf
# def read_fmri_avgSingleTrial_peak(base_dir, subject_id, selected_roi):
#     # Load the data
#     parent_input_sagar_original = f'{base_dir}/fmri/average_of_singletrial/fmri_{subject_id}_{selected_roi}.csv'
#     df = pd.read_csv(parent_input_sagar_original)
#     df = df.sort_values(by='CID')
#
#     # Create fMRI array (CIDs as rows, voxels as columns)
#     fmri_array = df.pivot(index='CID', columns='voxel', values='fmri')
#
#
#
#
#     # Convert the cleaned DataFrame to a numpy array
#     fmri_array_cleaned = fmri_array.to_numpy()
#
#     #replace nan with 0
#     fmri_array_cleaned = np.nan_to_num(fmri_array_cleaned)
#
#
#
#     return fmri_array_cleaned


#get smiles from CID
def get_smiles_from_cid(CIDs):
    smiles_subject = []
    for cid in CIDs:
        compound =  pcp.Compound.from_cid(cid)
        smiles_subject.append(compound.canonical_smiles)
    return smiles_subject

def read_CIDs(base_dir,subject_id):
    mat1 = loadmat(f'{base_dir}/fmri/Fahime/behavior/behav_ratings_NEMO0{subject_id}.mat')
    CIDs = mat1['behav'][0][0]['cid']
    CIDs = CIDs.squeeze(1)
    CIDs = CIDs.tolist()
    
    smiles_subject = get_smiles_from_cid(CIDs)
    
    return CIDs, smiles_subject

def smiles_to_selfies(smiles):
    """Convert SMILES to SELFIES"""
    selfies=[]
    for smile in smiles:
        selfies_str = sf.encoder(smile)
        selfies.append(selfies_str)
    return selfies


def set_seeds(seed=2024):
    # Set environment variable for hash-based operations
    os.environ['PYTHONHASHSEED'] = str(seed)

    # Set seed for Python's random module
    random.seed(seed)

    # Set seed for NumPy
    np.random.seed(seed)

    # Set seed for PyTorch
    torch.manual_seed(seed)

    # If using GPUs, set seed for CUDA operations
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # Configure cuDNN for deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def create_nc_mask(base_dir,subject,roi,threshold=0.3):

    #loading noise_ceiling
    noise_ceiling = pd.read_csv(f'{base_dir}/fmri/noise_ceiling_session/noise_ceiling_session_{subject}_{roi}.csv')
    noise_ceiling.rename(columns={'fmri':'nc'}, inplace=True)
    noise_ceiling['nc_corrected'] = np.sqrt((2/(1+np.sqrt(1/noise_ceiling['nc']**2))))
    noise_ceiling['nc_corrected'] = np.where(np.isinf(noise_ceiling['nc_corrected']), 0, noise_ceiling['nc_corrected'])
    noise_ceiling['nc_corrected'] = np.nan_to_num(noise_ceiling['nc_corrected'])

    noise_ceiling_avg = noise_ceiling.groupby(['voxel','subject','roi']).mean()

    # noise_ceiling_avg= pd.merge(df,noise_ceiling_avg,on=['subject','roi','voxel'])
    noise_ceiling_avg.sort_values(by=['voxel','subject','roi'], inplace=True)
    noise_ceiling_avg['nc_selected']= noise_ceiling_avg['nc_corrected']>=threshold
    mask_1d= noise_ceiling_avg['nc_selected'].values.tolist()

    mask_1d=np.array(mask_1d)
    # print(mask_1d.shape,"nc shape")
    _,mask_3d = to3d(base_dir,mask_1d,subject,roi)
    return  mask_3d



def mask_fmri(base_dir,fmri,nc_mask_array,subject,roi,nc_mask,func_mask):
    path_ROI = f'{base_dir}/fmri/supportings/S{subject}/'
    # Load the NIfTI mask file
    #load the whole brain mask
    maskfile = f'rw{roi}.nii'
    mask_img = nib.load(path_ROI + maskfile)
    mask_data = mask_img.get_fdata()>0.1


    #load functional mask
    maskfile_func = 'ARC3_fanatgw3_pos.nii'
    mask_img_func = nib.load(path_ROI + maskfile_func)
    mask_img_func=mask_img_func.get_fdata()>0

    #load anatomical mask
    maskfile_anat = 'ARC3_anatgw.nii'
    mask_img_anat = nib.load(path_ROI + maskfile_anat)
    mask_img_anat=mask_img_anat.get_fdata()>0.1

    #count the number of true values in the mask
    print(np.sum(mask_data),np.sum(nc_mask_array),np.sum(mask_img_func),np.sum(mask_img_anat))



    #intersection of all masks
    if func_mask and nc_mask:
        result_reduce = np.logical_and.reduce([mask_data,mask_img_anat, mask_img_func, nc_mask_array])
    elif func_mask:
        result_reduce = np.logical_and.reduce([mask_data,mask_img_anat, mask_img_func])
    elif nc_mask:
        result_reduce = np.logical_and.reduce([mask_data, mask_img_anat, nc_mask_array]) 
    else:
        result_reduce = np.logical_and.reduce([mask_data, mask_img_anat])
    _,mask_1d = from3d(base_dir,result_reduce,subject,roi)


    #read fmri data
    # masked_fmri = fmri[:,mask_1d]

       # Identify voxels that contain NaN values in any trial
    # nan_mask = np.isnan(fmri,axis=0)
     # True if voxel has NaN in any CID

    # Compute mean for each voxel across all CIDs
    mean_values_voxel = fmri.mean(axis=0)
    zero_mean_voxels = abs(mean_values_voxel) == 0  # Near-zero mean voxels

    # Combine NaN and zero-mean voxels
    # print(len(nan_mask),len(zero_mean_voxels),len(mask_1d))
    voxels_to_remove = zero_mean_voxels | ~mask_1d  # Logical OR to combine both conditions
    masked_fmri = fmri[:, ~voxels_to_remove]  # Drop the columns (voxels) to remove

    #index of ~voxels_to_remove
    indices = np.where(~voxels_to_remove)[0]
    indices = pd.DataFrame(indices)





    # print(mask_1d)
    # print(fmri.shape,np.sum(voxels_to_remove),masked_fmri.shape)
    return indices, masked_fmri


def df_to_cid_voxel_array(df,tr):
    # Load the data
   
    values=f'TR_{tr}'
    # Create fMRI array (CIDs as rows, voxels as columns)
    fmri_array = df.pivot(index='CID', columns='voxel', values=values)
    fmri_array_sorted = fmri_array.sort_index().sort_index(axis=1)
    fmri_array_cleaned = fmri_array_sorted.to_numpy()

    fmri_array_cleaned = np.nan_to_num(fmri_array_cleaned)



    return fmri_array_cleaned

def read_fmri_avgSingleTrial_peak(base_dir, subject_id, selected_roi):
    # Load the data
    parent_input_sagar_original = f'{base_dir}/fmri/average_of_singletrial/fmri_{subject_id}_{selected_roi}.csv'
    df = pd.read_csv(parent_input_sagar_original)
    # if dro_voxels is not None:
    #     #drop voxels that are not in the mask
    #     df = df[~df['voxel'].isin(dro_voxels)]
    # df = df.sort_values(by=['CID','voxel'])

    # Create fMRI array (CIDs as rows, voxels as columns)
    fmri_array = df.pivot(index='CID', columns='voxel', values='fmri')




    # Convert the cleaned DataFrame to a numpy array
    fmri_array_cleaned = fmri_array.to_numpy()

    #replace nan with 0
    fmri_array_cleaned = np.nan_to_num(fmri_array_cleaned)



    return fmri_array_cleaned


def read_fmri_avgSingleTrial_dropna(base_dir, subject_id, selected_roi,dro_voxels=None):
    # Load the data
    parent_input_sagar_original = f'{base_dir}/fmri/average_of_singletrial/fmri_{subject_id}_{selected_roi}.csv'
    df = pd.read_csv(parent_input_sagar_original)


    # Create fMRI array (CIDs as rows, voxels as columns)
    fmri_array = df.pivot(index='CID', columns='voxel', values='fmri')




    # Convert the cleaned DataFrame to a numpy array
    fmri_array_cleaned = fmri_array.to_numpy()

    #replace nan with 0
    fmri_array_cleaned = np.nan_to_num(fmri_array_cleaned)



    return fmri_array_cleaned

def read_orig_avg(base_dir, subject_id, selected_roi,tr):
    # % Max FIR times for [PirF, PirT, AMY and OFC];
    # P = [[5, 5, 5], [4, 4, 5], [4, 3, 4], [3, 3, 5]]
    parent_input_sagar_original = base_dir + f'/fmri/NEMO_scripts-master/odor_responses_S1-3_regionized/odor_responses_S{subject_id}.mat'
    data = sio.loadmat(parent_input_sagar_original)

    if selected_roi == 'PirF':
        # pi = P[0][subject_id - 1]
        roi = data['odor_vals'][0][0][:,tr,:]
        # print(roi.shape,roi,"sssss")


    elif selected_roi == 'PirT':
        # pi = P[1][subject_id - 1]
        roi = data['odor_vals'][0][1][:,tr,:]

    elif selected_roi == 'AMY':
        # pi = P[2][subject_id - 1]
        roi = data['odor_vals'][0][2][:,tr,:]
    elif selected_roi == 'OFC':
        # pi = P[3][subject_id - 1]
        roi = data['odor_vals'][0][3][:,tr,:]


    roi = np.moveaxis(roi, -1, 0)
    # print("sssss",roi.shape)
    return roi

def prepare_dataset(ds):
    if 'y' in ds.columns:
        ds['y'] = ds['y'].apply(ast.literal_eval)
    ds['embeddings'] = ds['embeddings'].apply(ast.literal_eval)
    return ds


def read_pom(base_dir, CIDs):
    embedding_open_pom = '/alignment_olfaction_datasets/curated_datasets/embeddings/pom/sagar_pom_embeddings_Apr17.csv'
    ds_pom = pd.read_csv(base_dir + embedding_open_pom)
    ds_pom = prepare_dataset(ds_pom)

    ds_pom = ds_pom.sort_values(by='CID')
    ds_pom = ds_pom[['CID', 'embeddings', 'y', 'subject']]
    # ds_pom = ds_pom[ds_pom['subject'] == subject]
    ds_pom = ds_pom[ds_pom['CID'].isin(CIDs)]

    #filter based on unique CIDs
    ds_pom = ds_pom.drop_duplicates(subset=['CID'])

    pom_embeddings = ds_pom['embeddings']
    pom_embeddings = np.array([np.array(x) for x in pom_embeddings])

    return pom_embeddings


def npy2nii( data_array,roi, subject,dir='' , title='', save=True):
    mask_img, newVOL = to3d(data_array, roi, subject)
    new_img = nib.Nifti1Image(newVOL, affine=mask_img.affine)

    # Save the new NIfTI image
    # output_file_path = 'path/to/output_file.nii'  # Output NIfTI file path
    # nib.save(new_img, f"{base_dir}/fmri/metrics_avgsingletrial/subj_{subject}_{roi}_{model}_{layer}_{title}_metrics.nii")
    if save:
        nib.save(new_img, f"{dir}/{roi}+{subject}_{title}.nii")

    return new_img

def from3d(base_dir,newVOL, subject, roi):
    # path_ROI = f'../../../../T5 EVO/fmri/supportings/S{subject}/'
    path_ROI = f'{base_dir}/fmri/supportings/S{subject}/'
    maskfile = f'rw{roi}.nii'
    # Load the NIfTI mask file
    mask_img = nib.load(path_ROI + maskfile)
    mask_data = mask_img.get_fdata()
    mask_data[np.isnan(mask_data)] = 0
    # Convert to a boolean array (logical)
    mask_data = mask_data.astype(bool)
    voxel_inds = np.flatnonzero(mask_data)
    # Now let's reverse the process and extract the values from newVOL
    data_array_reversed = np.zeros(len(voxel_inds),dtype=bool)

    for idx, voxel_idx in enumerate(voxel_inds):
        xx, yy, zz = np.unravel_index(voxel_idx, newVOL.shape)
        data_array_reversed[idx] = newVOL[xx, yy, zz]


    return mask_img, data_array_reversed

def to3d(base_dir,data_array, subject,roi):
    path_ROI = f'{base_dir}/fmri/supportings/S{subject}/'
    maskfile = f'rw{roi}.nii'
    # Load the NIfTI mask file
    mask_img = nib.load(path_ROI + maskfile)
    mask_data = mask_img.get_fdata()
    mask_data[np.isnan(mask_data)] = 0
    # Convert to a boolean array (logical)
    mask_data = mask_data.astype(bool)
    voxel_inds = np.flatnonzero(mask_data)
    # Initialize new volume with zeros (same shape as mask)
    print(data_array.shape, voxel_inds.shape, subject, roi)
    if data_array.shape[0] != voxel_inds.shape[0]:
        raise ValueError("Data array and mask must have the same shape.", data_array.shape, voxel_inds.shape, subject,
                         roi)
    newVOL = np.zeros(mask_data.shape)
    print(newVOL.shape, data_array.shape)
    # Iterate over voxel indices
    for ii in range(len(voxel_inds)):
        # Convert linear index to 3D subscripts
        xx, yy, zz = np.unravel_index(voxel_inds[ii], mask_data.shape)
        # print(data_array[ii][0][0])
        # Assign data value to new volume
        newVOL[xx, yy, zz] = data_array[ii]
    return mask_img, newVOL


def f_test(X, y, estimator):
    # Predicted values
    y_pred = estimator.predict(X)

    # Residual Sum of Squares (RSS)


    # Total Sum of Squares (SST)
    y_mean = np.mean(y, axis=0)
    # SST = np.sum((y - y_mean) ** 2)
    SSE = np.sum((y - y_pred) ** 2)
    SSM = np.sum((y_pred - y_mean) ** 2)
    SST = np.sum((y - y_mean) ** 2)

    # Regression Sum of Squares (SSR)
    # SSR = SST - SSE

    dfm = X.shape[1] - 1  # Degrees of freedom for the model
    dfe = X.shape[0] - X.shape[1]  # Degrees of freedom for the error
    dft = X.shape[0] - 1  # Total degrees of freedom

    msm = SSM / dfm  # Mean Squares for the model
    mse = SSE / dfe  # Mean Squares for the error
    mst = SST / dft  # Mean Squares for the total


    # Degrees of freedom
    # n = len(y)  # Number of samples
    # k = X.shape[1]  # Number of predictors (features)

    # F-statistic
    F =msm/mse

    # P-value
    from scipy.stats import f
    p_value = 1 - f.cdf(F, dfm, dfe)

    return F, p_value

def find_overlap(base_dir, ds, subject_source):
    subjects = [1, 2, 3]
    subjects.remove(subject_source)
    smiles_df_1 = pd.read_csv(f"{base_dir}/embeddings{ds}/CIDs_smiles_selfies_{subject_source}{ds}.csv")
    # Get the CIDs from the source DataFrame
    cids_1 = smiles_df_1['CIDs']
    cids_rest = []

    # Collect CIDs from the remaining subjects
    for subject in subjects:
        smiles_df_dest = pd.read_csv(f"{base_dir}/embeddings{ds}/CIDs_smiles_selfies_{subject}{ds}.csv")
        cids_2 = smiles_df_dest['CIDs']
        cids_rest.append(set(cids_2))

    # Find overlapping indices
    overlapping_indices = [idx for idx, cid in enumerate(cids_1) if all(cid in cids_set for cids_set in cids_rest)]

    return overlapping_indices,cids_1
