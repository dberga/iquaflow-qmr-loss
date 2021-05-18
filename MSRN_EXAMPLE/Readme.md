# Inference MSRN

Example of loading and running the inference of MSRN trained model.

To run the model execute:
`python main.py --filename data/sample_1mpx.tif --gpu_device '0'
`

It will generate a super resolved version at 0.7m/px within the same folder.
Remove the `gpu_device` flag device to run the code on CPU


## Building the Docker image

Execute:
```
docker build -t srai_msrn .
```

## Run inference (interactive) with Docker

```
docker run --rm -it --entrypoint /bin/bash --gpus all  -v "$PWD":/MSRN_EXAMPLE  srai_msrn 
```

#### Benchmark memory/time for sample image

| GPU memory (MiB) | Time GPU (s) | Total time one image (s) |
|------------|--------------|--------------------------|
| 9194    | 3.8          | 35                      |  
