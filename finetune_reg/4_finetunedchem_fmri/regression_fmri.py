base_dir = '/proj' 

import sys
parent_dir = f'{base_dir}/olfactory-fmri-alignment'
sys.path.append(parent_dir)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.metrics import mean_squared_error
import scipy
from sklearn.decomposition import PCA
from utils.helpers import *
from scipy import stats
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from utils import molecular_prep
from sklearn.preprocessing import StandardScaler

seed = 2024

import argparse





parser = argparse.ArgumentParser(description='chem_exploration')

# num_train_epochs=args.num_train_epochs
# seeds=args.seeds

parser.add_argument('--subject', type=int)
parser.add_argument('--num_train_epochs', type=int)
parser.add_argument('--model', type=str)
parser.add_argument('--n_components', type=str)
parser.add_argument('--behavior_embeddings', type=str)
parser.add_argument('--out_dir', type=str)
parser.add_argument('--n_fold', type=int, required=True)
parser.add_argument('--unfreeze_last_n', type=int, required=True)
parser.add_argument('--roi', type=str, required=True)
parser.add_argument('--tr', type=int)
parser.add_argument('--nc_mask', type=str)
parser.add_argument('--func_mask', type=str)
parser.add_argument('--read_style', type=str)
parser.add_argument('--z_score', type=str)

def custom_ridge_regression(X, y,alpha):
    if alpha is None:
        linreg = RidgeCV(alphas=np.logspace(2,7,16),cv=5)
    else:
        linreg = Ridge(alpha=alpha)
    # linreg = MultiOutputRegressor(linreg,n_jobs=-1)

    estimator = linreg.fit(X, y)
    return estimator


def train_and_eval_prekfold(Xs_train,ys_train,Xs_test,ys_test, voxels_retained,n_components=None):
    """
    Train and evaluate a regression model using cross-validation.

    Parameters:
    data (DataFrame): Input data containing embeddings and behavior_average.
    times (int): Number of cross-validation iterations.
    n_components (int, optional): Number of components for dimensionality reduction.

    Returns:
    tuple: Contains CIDs, predicted values, test targets, runs, MSE errors, and correlations.
    """


    # stats_correlations = correlations[:, 0]
    # pvalues_correlations = correlations[:, 1]


    predicteds = []
    y_tests = []
    voxels_predicteds = []
    # alphas = np.logspace(2,7,16)

    # for i in range(times):
    # for fold, (train_index, test_index) in enumerate(kf.split(X), start=1):
    for X_train,y_train,X_test,y_test in zip(Xs_train,ys_train,Xs_test,ys_test):
        if n_components and n_components<X_train.shape[1]:
            print("yes pca")
            pca = PCA(n_components=n_components)
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)

        # for alpha in alphas:
        linreg = custom_ridge_regression(X_train, y_train,None)
        predicted = linreg.predict(X_test)
        if len(predicted.shape)==1:
            predicted = predicted.reshape(-1,1)





        #flatten the predicted and y_test arrays
        voxels_predicted = np.arange(predicted.shape[1])
        # predicted = np.concatenate(predicted)

        # y_test_flatten = np.concatenate(y_test)
        # print(y_test.shape,predicted.shape,"shape_y")


        predicteds.extend(predicted)
        y_tests.extend(y_test)



        voxels_predicted = np.repeat(voxels_predicted,y_test.shape[0])
        voxels_predicteds.extend(voxels_predicted)

    if voxels_retained is None:
        voxels_retained = range(y_test.shape[1])

    predicteds = np.asarray(predicteds)
    y_tests = np.asarray(y_tests)
    voxels = np.asarray(voxels_retained)
    voxels_predicteds = np.asarray(voxels_predicteds)
    # print("yyy",predicteds.shape,y_tests.shape)


    #compute pearson correlation on the whole prediction
    mse_errors =  np.asarray([mean_squared_error(predicteds[:, i], y_tests[:, i]) for i in range(y_tests.shape[1])])
    correlations = np.asarray([scipy.stats.pearsonr(predicteds[:, i], y_tests[:, i]) for i in range(y_tests.shape[1])])
    p_value_correlation= np.zeros(shape=correlations.shape[0])
    p_value_mse = np.zeros(shape=mse_errors.shape)
    y_tests_shuffle = y_tests.copy()

    #shuffle predictions 1000 times and compute correlation coefficient and spearman correlation 1000 times
    times=1000
    for i in range(times):
        print(i)
        np.random.shuffle(y_tests_shuffle)
        mse_error_shuffle = np.asarray([mean_squared_error(predicteds[:, i], y_tests_shuffle[:, i]) for i in range(y_tests_shuffle.shape[1])])
        correlation_shuffle = np.asarray([scipy.stats.pearsonr(predicteds[:, i], y_tests_shuffle[:, i]) for i in range(y_tests_shuffle.shape[1])])
        # print(correlation_shuffle.shape,correlations.shape)
        # ev = correlation_shuffle[:,0]>correlations[:,0]
        # print(ev.shape)
        p_value_correlation += correlation_shuffle[:,0]>correlations[:,0]
        p_value_mse += mse_error_shuffle<mse_errors

    p_value_correlation = p_value_correlation/times
    p_value_mse = p_value_mse/times




    return predicteds, y_tests,correlations,mse_errors,p_value_correlation,p_value_mse,voxels,voxels_predicteds


def pipeline(Xs_train,ys_train,Xs_test,ys_test,voxels_retained,n_components=None):
    """
    Run the pipeline for a specific voxel and model, processing behavior and embeddings data.

    Parameters:
    behavior_ev (DataFrame): behavior event data.
    voxel (int): The voxel identifier.
    model_name (str): Name of the model being evaluated.
    input_file (str): Path to the input CSV file.
    times (int, optional): Number of cross-validation iterations. Default is 30.
    n_components (int, optional): Number of components for dimensionality reduction. Default is None.

    Returns:
    tuple: DataFrames containing predictions and metrics.
    """
    # Filter voxel-specific data
    # Xs_train = embeddings_train
    # ys_train = behavior_train
    # Xs_test = embeddings_test
    # ys_test = behavior_test
    

    predicteds, y_tests,correlations,mse_errors,p_value_correlation,p_value_mse,voxels,voxels_predicteds = train_and_eval_prekfold(Xs_train,ys_train,Xs_test,ys_test,voxels_retained,n_components=n_components)



    stats_correlations = correlations[:, 0]
    pvalues_correlations = correlations[:, 1]
    print(stats_correlations.shape,mse_errors.shape,voxels.shape,p_value_correlation.shape,p_value_mse.shape)
    # (101,) (101,) (505,) (505,) (101,) (101,) (16160,)
    # (101,) (101,) (101,) (505,) (101,) (101,) (16160,)
    metrics_df = pd.DataFrame(
        np.column_stack([stats_correlations, mse_errors,voxels,p_value_correlation,p_value_mse]),
        columns=['correlation', 'mse','target','p_value_correlation','p_value_mse']
    )

    return metrics_df


def compute_correlation(Xs_train,ys_train,Xs_test,ys_test,voxels_retained=None,n_components=None):
    """
    Compute correlations for MolFormer across specified layers and voxels.

    Parameters:
    times (int): Number of cross-validation iterations.
    n_components (int): Number of components for dimensionality reduction.
    behavior_ev (DataFrame): behavior event data.
    input_file_molformer (str): Base path for MolFormer input files.
    layers (list, optional): List of layers to process. Default is [13].

    Returns:
    tuple: Metrics and predictions for MolFormer.
    """

    results = pipeline(Xs_train,ys_train,Xs_test,ys_test, voxels_retained,n_components=n_components)
    df_metric = results
    return df_metric


def main():
    
    set_seeds(seed=seed)
    args = parser.parse_args()
    args.nc_mask = str(args.nc_mask).lower() == 'true'
    args.func_mask = str(args.func_mask).lower() == 'true'
    args.z_score = str(args.z_score).lower() == 'true'  
    # Convert "None" string to actual Python None
    if args.n_components == "None":
        args.n_components = None
    else:
        args.n_components = int(args.n_components)
    layers_end = [1,1,1,8,6,12,12,4,12,12,12,24,24,24
                #   ,12,12
                  ]
    layers_end = [1,1,1,8,6,12,12,4,12,12,12,24,24,24
                #   ,12,12
                  ]
    models = ['openpom','behavior','molecular_descriptors','ChemBERT_ChEMBL_pretrained', 'ChemBERTa-zinc-base-v1','MoLFormer-XL-both-10pct','SELFormer',
    
          'smiles-gpt','decoder_BARTSmiles','encoder_BARTSmiles','molgpt','ChemGPT-4.7M','ChemGPT-19M','ChemGPT-1.2B'
        #   ,'encoder_MolGen-large','decoder_MolGen-large'
          ]
    
    behavior_embeddings = args.behavior_embeddings
    # if behavior_embeddings is not None:
    #     print("behavior_embeddings included:", behavior_embeddings)

    #     if behavior_embeddings.split(',') == ['']:
    #         behavior_embeddings = int(behavior_embeddings)
    #     else:
    #         behavior_embeddings_int = [
    #         int(b.strip()) for b in behavior_embeddings.split(',')
    #         ]
    #     labels_array = labels_array[:, behavior_embeddings_int]
    # else:
    #     behavior_embeddings = ''  

    
    model_name =args.model
    m= models.index(model_name)
    subject = args.subject
    roi= args.roi
    num_train_epochs=args.num_train_epochs
    n_components=args.n_components
    n_fold = args.n_fold
    unfreeze_last_n = args.unfreeze_last_n
    out_dir = args.out_dir
    nc_mask = args.nc_mask
    func_mask = args.func_mask
    tr = args.tr
    read_style = args.read_style
    z_score = args.z_score
    # out_dir='fembeddings_lora'
    out_dir = args.out_dir
    if not os.path.exists(f"{base_dir}/{out_dir}_fmri_metrics"):
        os.makedirs(f"{base_dir}/{out_dir}_fmri_metrics",exist_ok=True)
    
    P = [[5,5,5], [4,4, 5], [4, 3, 4], [3, 3 ,5]]
    rois = ["PirF","PirT","AMY","OFC"]
    i_roi = rois.index(roi)
    tr_orig= tr
    if tr==-1:
        tr = P[i_roi][subject-1]

    # for layer in range(layers_end[m],0,-1):
    for layer in range(1,layers_end[m]+1):
        if read_style == 'avg_computed':
            
            parent_input_sagar_original = f'{base_dir}/fmri/average_of_singletrial_allTRs/fmri_{subject}_{roi}.csv'
            
            df = pd.read_csv(parent_input_sagar_original)
            fmri =df_to_cid_voxel_array(df,tr)
            nc_mask_array =create_nc_mask( base_dir,subject,roi,threshold=0.25)
            remained_indices,fmri =mask_fmri(base_dir,fmri,nc_mask_array,subject,roi,nc_mask,func_mask)
            # print(fmri.shape,fmri_array.shape)
            if fmri.shape[1]==0:
                print("no voxels remained")
                #create empy metrics csv file
                metrics = pd.DataFrame(columns=['correlation', 'mse','target','p_value_correlation','p_value_mse'])
                metrics.to_csv(f"{base_dir}/{out_dir}_fmri_metrics/all_laterfmrimetrics_{model_name}_{n_fold}_{num_train_epochs}_{subject}_{behavior_embeddings}_{unfreeze_last_n}_{layer}_{n_components}_{roi}_{nc_mask}_{func_mask}_{tr_orig}_{read_style}_{z_score}.csv", index=False)
                continue
        elif read_style == 'avg_orig':
            fmri=read_orig_avg(base_dir, subject, roi,tr)
        else:   
            raise ValueError(f"Invalid read_style: {read_style}. Choose 'avg_computed' or 'avg_orig'.")

        fmri = stats.zscore(fmri, axis=0)
        fmri = np.nan_to_num(fmri, nan=0)
        print("after read_orig_avg")
        # behavior = stats.zscore(behavior, axis=0)
        embeddings_train =[]
        embeddings_test= []
        fmris_train=[]
        fmris_test=[]
        pre='May15_'
        for i_fold in range(n_fold):


            filename = (
                f"{pre}finetuned_reg_fembeddings_{model_name}_{n_fold}_{num_train_epochs}_"
                f"{subject}_{behavior_embeddings}_{unfreeze_last_n}_{i_fold}_{layer}.npy"
            )
            print(f"reading {model_name}_{n_fold}_{num_train_epochs}_{subject}_{behavior_embeddings}_{unfreeze_last_n}_{i_fold}_{layer}.npy")
            
            embeddings = np.load(f"{base_dir}/read_orig_avg/{pre}finetuned_reg_fembeddings/{filename}")
            if z_score:
                print("zscored",flush=True)
                embeddings = np.nan_to_num(embeddings, nan=0)
                scaler = StandardScaler()
                embeddings = scaler.fit_transform(embeddings)  
            

            
            cids = pd.read_csv(f"{base_dir}/read_orig_avg/{pre}finetuned_reg_metrics/test_CIDs_{model_name}_{n_fold}_{num_train_epochs}_{subject}_{behavior_embeddings}_{unfreeze_last_n}_{i_fold}.csv")
            cids_test = cids.values
            # all_cids, _ = read_CIDs(base_dir,subject)
            smiles_df = pd.read_csv(f"{base_dir}/embeddings/CIDs_smiles_selfies_{subject}.csv")
            all_cids = smiles_df["CIDs"].values.tolist()
            overlapping_indices,_ = find_overlap(base_dir,'',subject)
            #find indices of cids_test in all_cids
            indices_test = np.where(np.isin(all_cids, cids_test.flatten()))[0]
            # Find indices of cids_test in all_cids
            # Find indices in all_cids that include all_cids_overlapping but not in indices_test
            indices_train = np.setdiff1d(overlapping_indices, indices_test)


           

            embedding_train = embeddings[indices_train]
            embedding_test = embeddings[indices_test]
            fmri_train = fmri[indices_train]
            fmri_test = fmri[indices_test]
            embeddings_train.append(embedding_train)
            embeddings_test.append(embedding_test)
            fmris_train.append(fmri_train)
            fmris_test.append(fmri_test)


            
        
        metrics = compute_correlation(embeddings_train,fmris_train,embeddings_test,fmris_test,None,n_components=n_components)
        metrics.to_csv(f"{base_dir}/{out_dir}_fmri_metrics/all_laterfmrimetrics_{model_name}_{n_fold}_{num_train_epochs}_{subject}_{behavior_embeddings}_{unfreeze_last_n}_{layer}_{n_components}_{roi}_{nc_mask}_{func_mask}_{tr_orig}_{read_style}_{z_score}.csv", index=False)

# #%%
if __name__ == "__main__":
    main()
#%%