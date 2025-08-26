import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

from utils.config import BASE_DIR, SEED
from utils.helpers import *
from utils.regression import compute_correlation
from utils.model_config import MODELS, LAYERS_END
from utils.arg_parser import create_behavior_parser, parse_common_args
from utils.data_loader import load_behavior_embeddings,load_model_embeddings,load_fold_cids
from datetime import datetime
from pathlib import Path

parser = create_behavior_parser('chem_exploration')


def main():
    set_seeds(seed=SEED)
    args = parser.parse_args()
    args = parse_common_args(args)
    
    model_name = args.model
    m = MODELS.index(model_name)
    participant_id = args.participant_id
    n_components = args.n_components
    n_fold = args.n_fold
    out_dir = args.out_dir
    z_score = args.z_score
    ds= args.ds
    embed_type='can'
    embed_cols = args.behavior_embeddings or get_descriptors(ds)





    for layer in range(1, LAYERS_END[m] + 1):
        # Load behavior embeddings
        for i_fold in range(n_fold):
            print(i_fold,layer)
            
            train_cids, test_cids = load_fold_cids( n_fold, i_fold, ds)
         
            # print(test_cids.shape,"shape")
            
            train_behavior = load_behavior_embeddings(ds,train_cids,participant_id, embed_cols, group_by_cid=True)
            test_behavior = load_behavior_embeddings(ds,test_cids,participant_id, embed_cols, group_by_cid=True)
            train_embeddings = load_model_embeddings( ds,model_name,train_cids,layer,embed_type='can')
            test_embeddings=load_model_embeddings( ds,model_name,test_cids,layer,embed_type='can')

            # Prepare data for all folds
       

            out_base = Path(BASE_DIR) / f"{out_dir}_metrics"
            out_base.mkdir(parents=True, exist_ok=True)
            out_file = out_base / f"metrics_model-{model_name}_ds-{ds}.csv"




            # Compute correlations
            metrics = compute_correlation(
                train_embeddings, train_behavior, test_embeddings, test_behavior, 
             n_components=n_components
            )


            metrics = metrics.assign(
                model=model_name,
                ds=ds,
                participant_id=participant_id,
                layer=layer,
                n_fold=n_fold,
                i_fold=i_fold,
                n_components=n_components,
                z_score=bool(z_score),
                behavior_embeddings=embed_cols,
                date = datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                run_id=os.environ.get("RUN_ID", "UNKNOWN")
                
            )

            write_header = not out_file.exists()
            metrics.to_csv(out_file, mode="a", index=False, header=write_header)
            print("****")


if __name__ == "__main__":
    main()



