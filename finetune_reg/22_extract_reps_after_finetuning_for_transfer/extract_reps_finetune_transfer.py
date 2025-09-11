import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = '/cfs/klemming/projects/supr/olfactory_alignment/olfactory-fmri-alignment'
sys.path.insert(0, project_root)

from utils.config import BASE_DIR, SEED
# BASE_DIR now imported from config

import sys
parent_dir=f"{BASE_DIR}/olfactory-fmri-alignment"
sys.path.append(parent_dir)
from transformers import AutoModel, AutoTokenizer
from utils.helpers import *
import argparse



def extract_representations(output_dir, tokenizer, model, model_name,n_fold, i_fold,
                             num_train_epochs, subject_source,subject_dest, behavior_embedding, unfreeze_last_n,
                              ds,lr,batch_size,weight_decay,input_type='smiles', token=0):
    model.eval()
    input_molecules = pd.read_csv(f'{BASE_DIR}/embeddings/CIDs_smiles_selfies_{subject_dest}.csv')[input_type].tolist()
    inputs = tokenizer(input_molecules, padding=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        for i, output in enumerate(outputs.hidden_states):
            filename = (
                f"{output_dir}_{model_name}_{n_fold}_{num_train_epochs}_"
                f"{subject_source}_{subject_dest}_{behavior_embedding}_{unfreeze_last_n}_{i_fold}_{i}{ds}_{lr}_{batch_size}_{weight_decay}.npy"
            )

            np.save(f"{BASE_DIR}/read_orig_avg/{output_dir}/{filename}", output[:, token, :].cpu().numpy())
def get_latest_checkpoint(path):
    checkpoints = [
        os.path.join(path, d)
        for d in os.listdir(path)
        if d.startswith("checkpoint-")
    ]
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint found in {path}")
    latest_checkpoint = sorted(checkpoints, key=os.path.getmtime)[-1]
    return latest_checkpoint


parser = argparse.ArgumentParser(description='chem_exploration')
parser.add_argument('--subject', type=int)
parser.add_argument('--n_fold', type=int)
parser.add_argument('--model_name', type=str)
parser.add_argument('--behavior_embedding', type=str)
parser.add_argument('--unfreeze_last_n', type=int)
parser.add_argument('--input_dir', type=str)

def main():
    # seed now imported from config
    set_seeds(seed=seed)
    args = parser.parse_args()
    model_name =args.model_name 
    if model_name == 'SELFormer':
        input_type = 'selfies'
    else:
        input_type = 'smiles'
    input_dir = args.input_dir
    output_dir=input_dir+'_fembeddings_transfer'
    input_dir=input_dir+'_models'
    subject_source = args.subject
    n_fold = args.n_fold
    behavior_embedding = args.behavior_embedding
    unfreeze_last_n = args.unfreeze_last_n
    nums_train_epochs = 40
    subjects = [1,2,3]
    subjects_dest =  [s for s in subjects if s != subject_source]


    # model_name= 'ChemBERT_ChEMBL_pretrained'
    


    top5_csv = os.path.join(f"{BASE_DIR}/best_hparam_selection_logs/top5_hparams_{model_name}_{subject_source}_{behavior_embedding}_{unfreeze_last_n}.csv")
    top5 = pd.read_csv(top5_csv)
    top5 = top5.query("Model == @model_name and subject == @subject_source")
    
    ds=''
    row = top5.iloc[0]
    lr = row['lr']
    batch_size   = int(row['batch_size'])
    weight_decay = row['weight_decay']

        
    for i_fold in range(n_fold):     
        
        for subject_dest in subjects_dest:
            print(f"Extracting representations for {model_name} {i_fold} {nums_train_epochs} {subject_source} {subject_dest} {behavior_embedding} {unfreeze_last_n} {input_type} {output_dir} {input_dir}", flush=True)
            
            path = os.path.join(
                BASE_DIR,
                input_dir,
                f"model_{model_name}_{n_fold}_{nums_train_epochs}_{subject_source}_{behavior_embedding}_{unfreeze_last_n}_{i_fold}{ds}_{lr}_{batch_size}_{weight_decay}"
            )

            checkpoint_path = get_latest_checkpoint(path)
            model = AutoModel.from_pretrained(checkpoint_path, trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
            if not os.path.exists(f"{BASE_DIR}/read_orig_avg/{output_dir}"):
                os.makedirs(f"{BASE_DIR}/read_orig_avg/{output_dir}",exist_ok=True)

            extract_representations(
                output_dir, tokenizer, model, model_name,n_fold, i_fold, nums_train_epochs, subject_source,subject_dest, behavior_embedding,unfreeze_last_n,ds,lr,batch_size,weight_decay,
                input_type=input_type
            )

if __name__ == "__main__":
    main()