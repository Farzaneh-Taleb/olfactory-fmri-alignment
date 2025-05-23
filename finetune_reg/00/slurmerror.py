import os

def find_slurmstepd_errors(directory="./logs"):
    # print("Searching for slurmstepd errors in .error files...",os.listdir('./'))
    # List all files ending with .error
    error_files = [f for f in os.listdir(directory) if f.endswith(".out")]
    
    for filename in error_files:
        # print(filename)
        filepath = os.path.join(directory, filename)
        try:
            with open(filepath, "r") as file:
                content = file.read()
                if "TR: -1" in content and "ROI: OFC" in content and 'Components: 30' in content and 'Model: ChemBERT_ChEMBL_pretrained' in content:
                # if 'Err' in content:
                    out_filename = filename.replace(".err", ".out")
                    out_filename = out_filename.replace("error", "output")
                    print(out_filename)
                    # read what is in front of roi int the content
                    # with open(os.path.join(directory, out_filename), "r") as out_file:
                    #     out_content = out_file.read()
                    #     print(out_content)
                    #     roi_line = [line for line in out_content.splitlines() if "ROI" in line]
                    #     if roi_line:
                    #         print(roi_line[0])
                    #     else:
                    #         print(f"Error found in {filename}, but no 'roi' line found.")
                    #     model_line = [line for line in out_content.splitlines() if "Model" in line]
                    #     if model_line:
                    #         print(model_line[0])
                    #     else:
                    #         print(f"Error found in {model_line}, but no 'roi' line found.")
                    #     #count how many times there is "after read_orig_avg" in the file
                    #     after_read_count = out_content.count("after read_orig_avg")
                    #     if after_read_count > 0:
                    #         print(f"Found {after_read_count} occurrences of 'after read_orig_avg' in {out_filename}.")
                    #     else:
                    #         print(f"No occurrences of 'after read_orig_avg' found in {out_filename}.")
                    # print(filename)
        except Exception as e:
            print(f"Could not read {filename}: {e}")

if __name__ == "__main__":
    find_slurmstepd_errors()