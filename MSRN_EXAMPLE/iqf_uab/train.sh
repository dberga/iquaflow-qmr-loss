: '
python3 training.py \
	--trainid "vanilla_20epoch_crops128_bsize16_lr0001" \
	--nEpochs 20 \
	--crop_size 128 \
	--batchSize 16 \
	--lr 0.001 \
	--cuda --nockpt --gpus "0" > "log_1.txt" ; 
	#--resume \
	#--add_noise \
	#--vgg_loss \
	#--regressor_loss "sigma" \ 
	#--regressor_criterion "BCELoss_mean" \ 
	#--regressor_loss_factor=1.0 \
	#--regressor_gt2onehot \
	#--regressor_onorm \
	#--regressor_zeroclamp
	#--colorjitter
	#--trainds_input test_datasets/AerialImageDataset/train/images
	#--valds_input test_datasets/AerialImageDataset/test/images
	#--step 50
	
python3 training.py \
	--trainid "sigma_MSEloss_20epoch_crops128_bsize16_lr0001_factor010_clamp" \
	--nEpochs 20 \
	--crop_size 128 \
	--batchSize 16 \
	--lr 0.001 \
	--regressor_loss "sigma" \
	--regressor_criterion "MSELoss_mean" \
	--regressor_loss_factor 0.10 \
	--regressor_zeroclamp \
	--cuda --nockpt --gpus "0" > "log_2.txt"  ;

python3 training.py \
	--trainid "snr_MSEloss_20epoch_crops128_bsize16_lr0001_factor010_clamp" \
	--nEpochs 20 \
	--crop_size 128 \
	--batchSize 16 \
	--lr 0.001 \
	--regressor_loss "snr" \
	--regressor_criterion "MSELoss_mean" \
	--regressor_loss_factor 0.10 \
	--regressor_zeroclamp \
	--cuda --nockpt --gpus "0" > "log_3.txt"  ;

python3 training.py \
	--trainid "sigma_BCEloss_20epoch_crops128_bsize16_lr0001_gt2onehot_factor010" \
	--nEpochs 20 \
	--crop_size 128 \
	--batchSize 16 \
	--lr 0.001 \
	--regressor_loss "sigma" \
	--regressor_criterion "BCELoss_mean" \
	--regressor_loss_factor 0.10 \
	--regressor_gt2onehot \
	--cuda --nockpt --gpus "0" > "log_4.txt"  ; 

python3 training.py \
	--trainid "snr_BCEloss_20epoch_crops128_bsize16_lr0001_gt2onehot_factor010" \
	--nEpochs 20 \
	--crop_size 128 \
	--batchSize 16 \
	--lr 0.001 \
	--regressor_loss "snr" \
	--regressor_criterion "BCELoss_mean" \
	--regressor_loss_factor 0.10 \
	--regressor_gt2onehot \
	--cuda --nockpt --gpus "0" > "log_5.txt"  ; 


python3 training.py \
	--trainid "sigma_L1loss_20epoch_crops128_bsize16_lr0001_factor010_clamp" \
	--nEpochs 20 \
	--crop_size 128 \
	--batchSize 16 \
	--lr 0.001 \
	--regressor_loss "sigma" \
	--regressor_criterion "L1Loss_mean" \
	--regressor_loss_factor 0.10 \
	--regressor_zeroclamp \
	--cuda --nockpt --gpus "0" > "log_6.txt"  ; 

python3 training.py \
	--trainid "snr_L1loss_20epoch_crops128_bsize16_lr0001_factor010_clamp" \
	--nEpochs 20 \
	--crop_size 128 \
	--batchSize 16 \
	--lr 0.001 \
	--regressor_loss "sigma" \
	--regressor_criterion "L1Loss_mean" \
	--regressor_loss_factor 0.10 \
	--regressor_zeroclamp \
	--cuda --nockpt --gpus "0" > "log_7.txt"  ; 

'


python3 training.py \
	--trainid "rer_L1loss_20epoch_crops128_bsize16_lr0001_010_clamp" \
	--nEpochs 20 \
	--crop_size 128 \
	--batchSize 16 \
	--lr 0.001 \
	--regressor_loss "rer" \
	--regressor_criterion "L1Loss_mean" \
	--regressor_loss_factor 0.10 \
	--regressor_zeroclamp \
	--cuda --nockpt --gpus "0" > "log_7b.txt"  ; 


python3 training.py \
	--trainid "sharpness_L1loss_20epoch_crops128_bsize16_lr0001_010_clamp" \
	--nEpochs 20 \
	--crop_size 128 \
	--batchSize 16 \
	--lr 0.001 \
	--regressor_loss "sharpness" \
	--regressor_criterion "L1Loss_mean" \
	--regressor_loss_factor 0.10 \
	--regressor_zeroclamp \
	--cuda --nockpt --gpus "0" > "log_7c.txt"  ; 

python3 training.py \
	--trainid "scale_L1loss_20epoch_crops128_bsize16_lr0001_010_clamp" \
	--nEpochs 20 \
	--crop_size 128 \
	--batchSize 16 \
	--lr 0.001 \
	--regressor_loss "scale" \
	--regressor_criterion "L1Loss_mean" \
	--regressor_loss_factor 0.10 \
	--regressor_zeroclamp \
	--cuda --nockpt --gpus "0" > "log_7d.txt"  ; 

python3 training.py \
	--trainid "sigma_L1loss_30epoch_crops128_bsize16_lr0001_onorm_clamp" \
	--nEpochs 30 \
	--crop_size 128 \
	--batchSize 16 \
	--lr 0.001 \
	--regressor_loss "sigma" \
	--regressor_criterion "L1Loss_mean" \
	--regressor_onorm \
	--regressor_zeroclamp \
	--cuda --nockpt --gpus "0" > "log_8.txt"  ; 

python3 training.py \
	--trainid "snr_L1loss_30epoch_crops128_bsize16_lr0001_onorm_clamp" \
	--nEpochs 30 \
	--crop_size 128 \
	--batchSize 16 \
	--lr 0.001 \
	--regressor_loss "snr" \
	--regressor_criterion "L1Loss_mean" \
	--regressor_onorm \
	--regressor_zeroclamp \
	--cuda --nockpt --gpus "0" > "log_9.txt"  ; 

