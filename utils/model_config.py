"""
Configuration constants for models and layers.
"""

# Model configurations
MODELS = [
    'openpom', 'behavior', 'molecular_descriptors', 'ChemBERT_ChEMBL_pretrained',
    'ChemBERTa-zinc-base-v1', 'MoLFormer-XL-both-10pct', 'SELFormer',
    'smiles-gpt', 'decoder_BARTSmiles', 'encoder_BARTSmiles', 'molgpt',
    'ChemGPT-4.7M', 'ChemGPT-19M', 'ChemGPT-1.2B'
]

# Layer endpoints for each model
LAYERS_END = [1, 1, 1, 8, 6, 12, 12, 4, 12, 12, 12, 24, 24, 24]

# ROI configurations  
ROIS = ["PirF", "PirT", "AMY", "OFC"]
P_VALUES = [[5, 5, 5], [4, 4, 5], [4, 3, 4], [3, 3, 5]]

INPUT_TYPES_CAN = {''}
INPUT_TYPES_CAN = {m: ("canonicalselfies" if m == "SELFormer" else "canonicalsmiles") for m in MODELS}

INPUT_TYPES_ISO = {''}
INPUT_TYPES_ISO = {m: ("isomericselfies" if m == "SELFormer" else "isomericsmiles") for m in MODELS}

# def get_input_type(model_name: str) -> str:
#     """Return the input type for a model (defaults to 'smiles' if unknown)."""
#     return INPUT_TYPES.get(model_name, "smiles")