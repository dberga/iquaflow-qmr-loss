find msrn/experiment/*/ -name "events.out.tfevents.*" -type 'f' -size -900k -delete
tensorboard --logdir=msrn/experiment/
