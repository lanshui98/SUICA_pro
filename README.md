# SUICA_pro

This repository extends the SUICA framework proposed by Qingtian Zhu et al., adapting and expanding it for gene imputation for 3D Spatial Transcriptomics.

<p align="center">
  <img src="fig.png" width="1000">
</p>

## Environment

```shell
conda create -n SUICA_pro python=3.9 -y && conda activate SUICA_pro
pip install -r requirements.txt
```

## Package structure

The typical data structure is as follows:

```
|-- SUICA_pro/                     
    |-- configs                    
        |-- ST
            |-- embedder_gae.yaml         
            |-- embedder_gae_3d_sparse.yaml
            |-- inr_embd.yaml
            |-- inr_embd_3d_sparse.yaml
            |-- inr_pred.yaml              # prediction with custom coordinates
            |-- inr_pred_3d_sparse.yaml
    |-- data
        |-- 2D_data.h5ad                  
        |-- 3D_data.h5ad                   
        |-- preprocessed_data             
    |-- logs                             
    |-- networks
    |-- systems
    |-- datasets.py
    |-- train.py                          
    |-- utils.py
    |-- predict.py                        
    |-- prepare_custom_coords.py           # normalize custom coords for prediction
    |-- map_coords_back.py                 # map reconstructed coords back to original space
    |-- requirements.txt
```

**Note:** Update the YAML config to point to your `.h5ad` file.

## Training

**Train the Graph AutoEncoder (GAE)**

```
# 2D data
python train.py --mode embedder --conf ./configs/ST/embedder_gae.yaml
```

```
# 3D sparse data
python train.py --mode embedder --conf ./configs/ST/embedder_gae_3d_sparse.yaml
```

**Train the GAE-INR**

```
# 2D
python train.py --mode inr --conf ./configs/ST/inr_embd.yaml
```

```
# 3D sparse
python train.py --mode inr --conf ./configs/ST/inr_embd_3d_sparse.yaml
```

## Prediction

**Predict on custom coordinates** (after GAE and GAE-INR training)

```
# --- 2D ---
# Prepare normalized custom coords first
python prepare_custom_coords.py --mode 2d --reference data/2D_data.h5ad --coords your_coords.xyz --output data/preprocessed_data/custom_coords_2d_norm.npy

# Run prediction
python predict.py --mode inr --conf ./configs/ST/inr_pred.yaml

# Map reconstructed coords back to original space
python map_coords_back.py --reconstructed reconstructed-custom_2d.h5ad --reference data/2D_data.h5ad --output reconstructed-original.h5ad --mode 2d
```

```
# --- 3D sparse ---
# Prepare normalized custom coords (with z-scale options)
python prepare_custom_coords.py --mode 3d --reference data/3D_data.h5ad --coords your_coords.xyz --output data/preprocessed_data/custom_coords_3d_norm.npy --keep_ratio True --preserve_z_scale True --z_scale_factor 1.5

# Run prediction
python predict.py --mode inr --conf ./configs/ST/inr_pred_3d_sparse.yaml

# Map reconstructed coords back to original space
python map_coords_back.py --reconstructed reconstructed-custom-3d.h5ad --reference data/3D_data.h5ad --output reconstructed-original-3d.h5ad --mode 3d --keep_ratio True --preserve_z_scale True --z_scale_factor 1.5
```

### Acknowledgements
If you find this useful, please cite our work:

Shui, L., Liu, Y., Julio, I.C., Clemenceau, J.R., Hoi, X.P., Dai, Y., Lu, W., Min, J., Khan, K., Roemer, B. and Jiang, M., 2026. UniST: A Unified Computational Framework for 3D Spatial Transcriptomics Reconstruction. bioRxiv, pp.2026-03.

This project is inspired by [SUICA](https://github.com/Szym29/SUICA). Please also consider citing SUICA:

Zhu, Q., Zheng, Y., Sang, Y., Zhan, Y., Zhu, Z., Ding, J. and Zheng, Y., 2024. Suica: Learning super-high dimensional sparse implicit neural representations for spatial transcriptomics. arXiv preprint arXiv:2412.01124.
