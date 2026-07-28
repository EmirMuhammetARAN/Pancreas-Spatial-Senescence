# Pancreatic Cellular Senescence Spatial Proteomics

This project aims to investigate cellular senescence in human pancreas tissues using spatial proteomics data (PhenoCycler/CODEX) with computer vision and a PyTorch-based Cellpose deep learning model.

The project analyzes 38-channel high-resolution images obtained from the NIH SenNet database to map how senescence varies across anatomical regions (Head, Body, Tail) at single-cell resolution.

## Core Methodology

*   **Cell Segmentation:** Fully automated cell boundary detection using DAPI and E-cadherin channels via the PyTorch-based `CellposeSAM v2` model.
*   **Big Data Processing:** Splitting high-resolution (e.g., 28800x50400 pixels) QPTIFF images into manageable 2048x2048 tiles using `zarr` and `tifffile`.
*   **Spatial Analysis & Clustering:** `DBSCAN`-based clustering for tissue boundary detection and anatomical region separation (Head, Tail) based on y-axis coordinates.
*   **Feature Extraction:** Calculating signal intensities for 38 protein channels (Insulin, p16, Lamin B1, CD68, etc.) for hundreds of thousands of segmented cells.
*   **Robust Normalization:** An age- and region-specific reference normalization strategy that is highly robust to outliers.

## Installation and Requirements

A Python virtual environment with the following libraries is required to run the project:

```bash
# Install dependencies
pip install numpy pandas tifffile zarr scikit-image scikit-learn matplotlib seaborn pyarrow fastparquet

# PyTorch and Cellpose (CUDA support highly recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install cellpose
```

## Project Structure & Pipeline

The pipeline execution order is detailed below. The entire process can be run seamlessly using `process_pipeline.py`.

1.  **`src/cutting/extract_all_tiles.py`**: Reads massive QPTIFF tissue images via Zarr and slices them into smaller, parallel-processable `tiff` tiles.
2.  **`src/cutting/run_cellpose.py`**: Takes the extracted tiles and runs the PyTorch-based CellposeSAM v2 model on the GPU to generate segmentation masks (mask_tiff) for every cell.
3.  **`src/cutting/extract_features.py`**: Matches the generated masks with the original raw images, computes the mean signal intensity for 38 channels per cell (feature extraction), and saves the tabular data in `.parquet` format.
4.  **`src/utils/filtering.py`**: Loads the parquet files, cleans tissue artifacts using DBSCAN, and splits the pancreas into spatial regions based on y-coordinates.
5.  **`src/analysis/bolgeler_arasi_karsilastirma.py`**: Analyzes the cleaned and regionally split data, calculates senescence ratios (p16 increase, Lamin B1 loss), and generates visualization plots (bar plots, box plots).

## Running the Automated Pipeline

To execute all steps sequentially via a single script:

```bash
python src/cutting/process_pipeline.py
```

## Dataset

The data utilized in this study is the Human Pancreas PhenoCycler/CODEX Atlas provided by the NIH SenNet Consortium. Due to extreme file sizes, the raw `.qptiff` images, generated `.parquet` data files, and `.tiff` masks are not included in this repository (See `.gitignore`).

## License

This project is open-source and intended for research and educational purposes.