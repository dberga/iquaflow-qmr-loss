![](https://github.com/satellogic/iquaflow/blob/main/docs/source/iquaflow_logo_mini.png)
Check [QMRNet's preprint](https://arxiv.org/abs/2210.06618) and [IQUAFLOW's preprint](https://arxiv.org/abs/XXXX.XXXXX) documentation

# IQUAFLOW - QMRNet's Loss for Super Resolution Optimization

- Note: Use any our [jupyter notebook](IQF-UseCase-QMRLOSS.ipynb) to run the use case.

- The rest of code is distributed in distinct repos [IQUAFLOW framework](https://github.com/satellogic/iquaflow), [QMRNet EO Dataset Evaluation Use Case](https://github.com/dberga/iquaflow-qmr-eo) and [QMRNet's Super-Resolution Use case](https://github.com/dberga/iquaflow-qmr-sisr).

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
2. `cd iq-sisr-use-case`
3. Then build the docker image with `make build`.(\*\*\*) This will also download required datasets and weights.
4. In order to execute the experiments:
    - `make dockershell` (\*)
    - Inside the docker terminal execute `python ./IQF-UseCase-QMRLOSS.py`
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

# Cite

If you use content of this repo, please cite:

```
@article{berga2022,
  title={QMRNet: Quality Metric Regression for EO Image Quality Assessment and Super-Resolution},
  author={Berga, David and Gallés, Pau and Takáts, Katalin and Mohedano, Eva and Riordan-Chen, Laura and Garcia-Moll, Clara and Vilaseca, David and Marín, Javier},
  journal={arXiv preprint arXiv:2210.06618},
  year={2022}
}
@article{galles2022,
  title={IQUAFLOW: A NEW FRAMEWORK TO MEASURE IMAGE QUALITY},
  author={Gallés, Pau and Takáts, Katalin and Hernández-Cabronero, Miguel and Berga, David and Pega, Luciano and Riordan-Chen, Laura and Garcia-Moll, Clara and Becker, Guillermo and Garriga, Adán and Bukva, Anica and Serra-Sagristà, Joan and Vilaseca, David and Marín, Javier},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2022}
}
```
