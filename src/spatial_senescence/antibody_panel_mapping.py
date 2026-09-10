# -*- coding: utf-8 -*-
"""
antibody_panel_mapping.py
=========================
Defines the official 38-plex PhenoCycler antibody panel extracted directly from
the PerkinElmer QPTIFF metadata (Study: SenNet_Pancreas).
"""

PANEL_38 = {
    0: {"marker": "DAPI", "category": "Nuclear", "target": "DNA / Nucleus", "is_senescence": True},
    1: {"marker": "CD45RA", "category": "Immune", "target": "Naive T cells / Leukocytes", "is_senescence": False},
    2: {"marker": "S100B", "category": "Neural", "target": "Schwann cells / Glia", "is_senescence": False},
    3: {"marker": "Insulin", "category": "Endocrine", "target": "Beta cells", "is_senescence": False},
    4: {"marker": "Carbonic Anhydrase 9", "category": "Ductal", "target": "CA9 / Duct epithelium", "is_senescence": False},
    5: {"marker": "Vimentin", "category": "Stroma", "target": "Mesenchymal / Fibroblasts", "is_senescence": False},
    6: {"marker": "Glucagon", "category": "Endocrine", "target": "Alpha cells", "is_senescence": False},
    7: {"marker": "CD45RO", "category": "Immune", "target": "Memory T cells", "is_senescence": False},
    8: {"marker": "Collagen I", "category": "Stroma", "target": "Extracellular matrix / Fibrosis", "is_senescence": False},
    9: {"marker": "p21", "category": "Senescence", "target": "CDKN1A (G1/S arrest)", "is_senescence": True},
    10: {"marker": "MPO", "category": "Immune", "target": "Myeloperoxidase / Neutrophils", "is_senescence": False},
    11: {"marker": "MECOM", "category": "TranscriptionFactor", "target": "PRDM3 / Transcriptional regulator", "is_senescence": False},
    12: {"marker": "CFTR", "category": "Ductal", "target": "Ductal ion channel", "is_senescence": False},
    13: {"marker": "PGP9.5", "category": "Neural", "target": "UCHL1 / Neuronal / Neuroendocrine", "is_senescence": False},
    14: {"marker": "Keratin 19", "category": "Ductal", "target": "CK19 / Ductal epithelium", "is_senescence": False},
    15: {"marker": "53BP1", "category": "DDR_Senescence", "target": "DNA Damage Checkpoint", "is_senescence": True},
    16: {"marker": "p16", "category": "Senescence", "target": "CDKN2A / p16INK4a", "is_senescence": True},
    17: {"marker": "Cadherin 11", "category": "Stroma", "target": "Mesenchymal cadherin", "is_senescence": False},
    18: {"marker": "Somatostatin", "category": "Endocrine", "target": "Delta cells", "is_senescence": False},
    19: {"marker": "CD8", "category": "Immune", "target": "Cytotoxic T cells", "is_senescence": False},
    20: {"marker": "Lamin B1", "category": "Senescence", "target": "Nuclear Lamina (Loss in Senescence)", "is_senescence": True},
    21: {"marker": "CD3e", "category": "Immune", "target": "Pan-T cells", "is_senescence": False},
    22: {"marker": "C-Peptide", "category": "Endocrine", "target": "Beta cells / Proinsulin", "is_senescence": False},
    23: {"marker": "HMGB1", "category": "Senescence", "target": "Alarmin / DAMP / SASP trigger", "is_senescence": True},
    24: {"marker": "Perilipin", "category": "Adipose", "target": "Lipid droplets / Adipocytes", "is_senescence": False},
    25: {"marker": "Podoplanin", "category": "Stroma_Lymph", "target": "Lymphatic endothelium / FRCs", "is_senescence": False},
    26: {"marker": "Pancreatic Polypeptide", "category": "Endocrine", "target": "PPY / Gamma cells", "is_senescence": False},
    27: {"marker": "CD4", "category": "Immune", "target": "Helper T cells", "is_senescence": False},
    28: {"marker": "E-cadherin", "category": "Epithelial", "target": "Cell-cell adhesion", "is_senescence": False},
    29: {"marker": "CD31", "category": "Endothelial", "target": "PECAM-1 / Blood vessels", "is_senescence": False},
    30: {"marker": "CD68", "category": "Immune", "target": "Macrophages / Monocytes", "is_senescence": False},
    31: {"marker": "Ki67", "category": "Proliferation", "target": "MKI67 / Proliferating cells", "is_senescence": True},
    32: {"marker": "EpCAM", "category": "Epithelial", "target": "Epithelial adhesion", "is_senescence": False},
    33: {"marker": "CD45", "category": "Immune", "target": "PTPRC / Pan-leukocyte", "is_senescence": False},
    34: {"marker": "CD56", "category": "Immune_Neural", "target": "NCAM1 / NK cells", "is_senescence": False},
    35: {"marker": "CD20", "category": "Immune", "target": "MS4A1 / B cells", "is_senescence": False},
    36: {"marker": "FOXP3", "category": "Immune", "target": "Regulatory T cells (Tregs)", "is_senescence": False},
    37: {"marker": "gH2AX", "category": "DDR_Senescence", "target": "Phospho-Histone H2A.X (DSB)", "is_senescence": True},
}

CHANNEL_NAME_TO_IDX = {v["marker"]: k for k, v in PANEL_38.items()}
IDX_TO_CHANNEL_NAME = {k: v["marker"] for k, v in PANEL_38.items()}

# Senescence & Proliferation Markers
SENESCENCE_MARKERS = {
    "p16": 16,
    "p21": 9,
    "Lamin_B1": 20,
    "HMGB1": 23,
    "Ki67": 31,
    "gH2AX": 37,
    "53BP1": 15,
}

# Key Lineage & Microenvironment Markers
ISLET_ENDOCRINE_MARKERS = {"Insulin": 3, "Glucagon": 6, "Somatostatin": 18, "C-Peptide": 22, "PPY": 26}
IMMUNE_MARKERS = {"CD68_Macrophage": 30, "CD8_Tcell": 19, "CD4_Tcell": 27, "CD3e_Tcell": 21, "CD20_Bcell": 35, "CD45_PanImmune": 33, "MPO_Neutrophil": 10}
STROMA_FIBROSIS_MARKERS = {"Collagen_I": 8, "Vimentin": 5}
DUCTAL_MARKERS = {"CFTR": 12, "Keratin_19": 14, "CA9": 4}
ENDOTHELIAL_MARKERS = {"CD31": 29}
