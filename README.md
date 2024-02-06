![](https://github.com/satellogic/iquaflow/blob/main/docs/source/iquaflow_logo_mini.png)
Check [QMRNet's article](https://www.mdpi.com/2072-4292/15/9/2451) and [IQUAFLOW's preprint](https://arxiv.org/abs/2210.13269) documentation

# IQUAFLOW - QMRNet's Loss for Super Resolution Optimization

- Note: Use any our shell scripts to train MSRN with QMR+[rer](train_new_rer.sh),[snr](train_new_snr.sh),[blur](train_new_sigma.sh),[sharpness](train_new_sharpness.sh),[gsr](train_new_scale.sh) to run the use case (train and validate the network). Also you can train and validate with the original [vanilla MSRN](train_new_vanilla_hd.sh).

- The rest of code is distributed in distinct repos [IQUAFLOW framework](https://github.com/satellogic/iquaflow), [QMRNet EO Dataset Evaluation Use Case](https://github.com/dberga/iquaflow-qmr-eo), [QMRNet's Super-Resolution Use case](https://github.com/dberga/iquaflow-qmr-sisr) and [QMRNet standalone code](https://github.com/satellogic/iquaflow/tree/main/iquaflow/quality_metrics).

# MSRN optimization

In this repo we add a novel QMRLoss which is able to adapt or regularize the training of MSRN upon a specific metric objective (e.g. blur, sharpness, rer, snr, etc.) with respect GT. It can also be used to regularize to explicitly minimize or maximize a specific metric.
 - *rer* is a measure of the edge response ( mesures the degree of the transition ) which also informs on image sharpness.
 - *snr* - Signal to noise (gaussian) ratio.
 - *sigma* - of a gaussian distribution. It measures blur by defining its kernel.
 - *sharpness* - Edge response (lower is blurred, higher is oversharpened)
 - *scale* - resolution proportion scale (x2 from 0.30 is 0.15 m/px)

____________________________________________________________________________________________________


## To reproduce the experiments:

1. `git clone https://YOUR_GIT_TOKEN@github.com/dberga/iquaflow-qmr-loss.git`
2. `cd iquaflow-qmr-loss`
3. Then build the docker image with `make build`.(\*\*\*) This will also download required datasets and weights.
4. In order to execute the experiments:
    - `make dockershell` (\*)
    - Inside the docker terminal execute `sh train_new_rer.sh`
5. Start the mlflow server by doing `make mlflow` (\*)
6. Notebook examples can be launched and executed by `make notebookshell NB_PORT=[your_port]"` (\**)
7. To access the notebook from your browser in your local machine you can do:
    - If the executions are launched in a server, make a tunnel from your local machine. `ssh -N -f -L localhost:[your_port]:localhost:[your_port] [remote_user]@[remote_ip]`  Otherwise skip this step.
    - Then, in your browser, access: `localhost:[your_port]/?token=sisr`


____________________________________________________________________________________________________

## Notes

   - The results of the IQF experiment can be seen in the MLflow user interface.
   - For more information please check the IQF_expriment.ipynb or IQF_experiment.py.
   - There are also examples of dataset Sanity check and Stats in SateAirportsStats.ipynb
   - The default ports are `8888` for the notebookshell, `5000` for the mlflow and `9197` for the dockershell
   - (*)
        Additional optional arguments can be added. The dataset location is:
        >`DS_VOLUME=[path_to_your_dataset]`
   - To change the default port for the mlflow service:
     >`MLF_PORT=[your_port]`
   - (**)
        To change the default port for the notebook: 
        >`NB_PORT=[your_port]`
   - A terminal can also be launched by `make dockershell` with optional arguments such as (*)
   - (***)
        Depending on the version of your cuda drivers and your hardware you might need to change the version of pytorch which is in the Dockerfile where it says:
        >`pip3 install torch==1.7.0+cu110 torchvision==0.8.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html`.
   - (***)
        The dataset is downloaded with all the results of executing the dataset modifiers already generated. This allows the user to freely skip the `.execute` as well as the `apply_metric_per_run` which __take long time__. Optionally, you can remove the pre-executed records folder (`./mlruns `) for a fresh start.
        
Note: make sure to replace "YOUR_GIT_TOKEN" to your github access token, also in [Dockerfile](Dockerfile).

# Design and Train the QMRNet (regressor.py)

In [QMRNet standalone code](https://github.com/satellogic/iquaflow/tree/main/iquaflow/quality_metrics) you can find several scripts for training and testing the QMRNet, mainly integrated in `regressor.py`. Using `run_spec.sh` you can specify any of the `cfgs\` folder where the architecture design and hyperparameters are defined. You can create new versions by adding new `.cfg` files.

# Cite

If you use content of this repo, please cite:

```
@article{rs15092451,
AUTHOR = {Berga, David and Gallés, Pau and Takáts, Katalin and Mohedano, Eva and Riordan-Chen, Laura and Garcia-Moll, Clara and Vilaseca, David and Marín, Javier},
TITLE = {QMRNet: Quality Metric Regression for EO Image Quality Assessment and Super-Resolution},
JOURNAL = {Remote Sensing},
VOLUME = {15},
YEAR = {2023},
NUMBER = {9},
ARTICLE-NUMBER = {2451},
URL = {https://www.mdpi.com/2072-4292/15/9/2451},
ISSN = {2072-4292},
ABSTRACT = {The latest advances in super-resolution have been tested with general-purpose images such as faces, landscapes and objects, but mainly unused for the task of super-resolving earth observation images. In this research paper, we benchmark state-of-the-art SR algorithms for distinct EO datasets using both full-reference and no-reference image quality assessment metrics. We also propose a novel Quality Metric Regression Network (QMRNet) that is able to predict the quality (as a no-reference metric) by training on any property of the image (e.g., its resolution, its distortions, etc.) and also able to optimize SR algorithms for a specific metric objective. This work is part of the implementation of the framework IQUAFLOW, which has been developed for the evaluation of image quality and the detection and classification of objects as well as image compression in EO use cases. We integrated our experimentation and tested our QMRNet algorithm on predicting features such as blur, sharpness, snr, rer and ground sampling distance and obtained validation medRs below 1.0 (out of N = 50) and recall rates above 95%. The overall benchmark shows promising results for LIIF, CAR and MSRN and also the potential use of QMRNet as a loss for optimizing SR predictions. Due to its simplicity, QMRNet could also be used for other use cases and image domains, as its architecture and data processing is fully scalable.},
DOI = {10.3390/rs15092451}
}
@article{galles2022,
  title={IQUAFLOW: A NEW FRAMEWORK TO MEASURE IMAGE QUALITY},
  author={Gallés, Pau and Takáts, Katalin and Hernández-Cabronero, Miguel and Berga, David and Pega, Luciano and Riordan-Chen, Laura and Garcia-Moll, Clara and Becker, Guillermo and Garriga, Adán and Bukva, Anica and Serra-Sagristà, Joan and Vilaseca, David and Marín, Javier},
  journal={arXiv preprint arXiv:2210.13269},
  year={2022}
}
```
