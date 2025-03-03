# EGNNMDA
EGNNMDA: Predict MiRNA-Disease Association Based on Ensemble Graph Neural Networks

## Requirements
  * xgboost==2.0.3
  * matplotlib==0.1.6
  * pandas==1.5.3
  * joblib==1.2.0
  * torch==2.0.0+cu118
  * pandas==1.5.3
  * scipy==1.9.3
  * scikit-learn==1.6.1
  * numpy==1.22.4
  * torch-geometric==2.3.1
  * xlrd==1.2.0
  * openpyxl==3.1.4


## Usage
  * If ```DATA``` and ```OUT``` directory does not exist, unzip ```DATA.rar``` and ```OUT.rar```.
  * Open ```params.py``` and choose ```dataset```, ```type_eval``` and ```nloop```, respectively.  ```number of repeat times```.
    (- dataset: "HMDD v2.0" / "HMDD v3.2" / "INDE_TEST", respectively to HMDD v2.0, HMDD v3.2 and Independent dataset.
     - type_eval = "KFOLD"  / "DIS_K" / "DENO_MI", respectively to 5-fold-CV, Case studies and Denovo miRNAs.
     - nloop = ```number of repeat times```.
     - Defaut: 5fold-CV in HMDD v2.0 dataset)
  * Run ```main.py```.
  
