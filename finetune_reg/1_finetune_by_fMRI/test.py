import os
import re
# Directory where your .out files are located
out_dir = "/proj/rep-learning-robotics/users/x_farzt/olfactory_alignment/olfactory-fmri-alignment-NEW/finetune_reg/1_finetune_by_behavior/logs"


# Message to search for
success_msg = "Fine-tuning completed successfully!"

# Regex patterns to extract values
patterns = {
    # "RUN_ID": r"RUN_ID=(\S+)",
    # "Global index": r"Global index=.*",
    # "DS": r"DS=.*",
    "participant_id": r"participant_id=\d+",
    # "model": r"model=.*",
    "behavior_embeddings": r"behavior_embeddings=.*",
    "unfreeze_last_n": r"unfreeze_last_n=.*",
    # "lr": r"lr=.*",
    # "weight_decay": r"weight_decay=.*",
    # "batch_size": r"batch_size=.*",
}

# Scan files
for fname in sorted(os.listdir(out_dir)):
    if fname.endswith(".out"):
        fpath = os.path.join(out_dir, fname)
        with open(fpath, "r") as f:
            content = f.read()
            if success_msg not in content:
                print(f"\n--- Missing success message: {fname} ---")
                # Extract run metadata
                for key, pat in patterns.items():
                    match = re.search(pat, content)
                    if match:
                        print(match.group(0))
                # Find folds that started
                folds = re.findall(r"Training fold \d+, Model .*", content)
                if folds:
                    print("Folds started:")
                    for fold in folds:
                        print("  " + fold)
                else:
                    print("No folds started")