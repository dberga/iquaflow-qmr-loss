python3 training.py \
	--trainid "vanilla_200epoch_crops512_bsize16_lr0001" \
	--nEpochs 200 \
	--crop_size 512 \
	--batchSize 16 \
	--lr 0.001 \
	--cuda --nockpt --gpus "0,1" > "log_vanilla.txt" ; 
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
	--trainid "sigma_L1loss_200epoch_crops512_bsize16_lr0001_factor010_clamp" \
	--nEpochs 200 \
	--crop_size 512 \
	--batchSize 16 \
	--lr 0.001 \
	--regressor_loss "sigma" \
	--regressor_criterion "L1Loss_mean" \
	--regressor_loss_factor 0.10 \
	--regressor_zeroclamp \
	--cuda --nockpt --gpus "0,1" > "log_sigma.txt"  ; 

python3 training.py \
	--trainid "snr_L1loss_200epoch_crops512_bsize16_lr0001_factor010_clamp" \
	--nEpochs 200 \
	--crop_size 512 \
	--batchSize 16 \
	--lr 0.001 \
	--regressor_loss "snr" \
	--regressor_criterion "L1Loss_mean" \
	--regressor_loss_factor 0.10 \
	--regressor_zeroclamp \
	--cuda --nockpt --gpus "0,1" > "log_snr.txt"  ; 


python3 training.py \
	--trainid "rer_L1loss_200epoch_crops512_bsize16_lr0001_factor010_clamp" \
	--nEpochs 200 \
	--crop_size 512 \
	--batchSize 16 \
	--lr 0.001 \
	--regressor_loss "rer" \
	--regressor_criterion "L1Loss_mean" \
	--regressor_loss_factor 0.10 \
	--regressor_zeroclamp \
	--cuda --nockpt --gpus "0,1" > "log_rer.txt"  ; 

