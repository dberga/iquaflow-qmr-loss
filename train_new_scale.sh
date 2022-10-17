
python3 training.py \
	--trainid "vanilla_200epoch_crops128_bsize32_lr0001" \
	--nEpochs 200 \
	--crop_size 128 \
	--batchSize 32 \
	--lr 0.001 \
	--cuda --gpus "1";
	#--resume \
	#--add_noise \
	#--vgg_loss \
	#--regressor_loss "scale" \ 
	#--regressor_criterion "BCELoss_mean" \ 
	#--regressor_loss_factor=1.0 \
	#--regressor_gt2onehot \
	#--regressor_onorm \
	#--regressor_zeroclamp
	#--colorjitter
	#--trainds_input test_datasets/inria-aid_short/train/images
	#--valds_input test_datasets/inria-aid_short/test/images
	#--step 50

# L1
python3 training.py \
	--trainid "scale_L1loss_200epoch_crops64_bsize128_lr0001_factor010_clamp" \
	--nEpochs 200 \
	--crop_size 64 \
	--batchSize 128 \
	--lr 0.001 \
	--regressor_loss "scale" \
	--regressor_criterion "L1Loss_mean" \
	--regressor_loss_factor 0.10 \
	--regressor_zeroclamp \
	--cuda --gpus "1";

python3 training.py \
	--trainid "scale_L1loss_200epoch_crops128_bsize32_lr0001_factor010_clamp" \
	--nEpochs 200 \
	--crop_size 128 \
	--batchSize 32 \
	--lr 0.001 \
	--regressor_loss "scale" \
	--regressor_criterion "L1Loss_mean" \
	--regressor_loss_factor 0.10 \
	--regressor_zeroclamp \
	--cuda --gpus "1";

python3 training.py \
	--trainid "scale_L1loss_200epoch_crops256_bsize16_lr0001_factor010_clamp" \
	--nEpochs 200 \
	--crop_size 256 \
	--batchSize 16 \
	--lr 0.001 \
	--regressor_loss "scale" \
	--regressor_criterion "L1Loss_mean" \
	--regressor_loss_factor 0.10 \
	--regressor_zeroclamp \
	--cuda --gpus "1";

python3 training.py \
	--trainid "scale_L1loss_200epoch_crops512_bsize4_lr0001_factor010_clamp" \
	--nEpochs 200 \
	--crop_size 512 \
	--batchSize 4 \
	--lr 0.001 \
	--regressor_loss "scale" \
	--regressor_criterion "L1Loss_mean" \
	--regressor_loss_factor 0.10 \
	--regressor_zeroclamp \
	--cuda --gpus "1";

# L2

python3 training.py \
	--trainid "scale_MSELoss_200epoch_crops64_bsize128_lr0001_factor010_clamp" \
	--nEpochs 200 \
	--crop_size 64 \
	--batchSize 128 \
	--lr 0.001 \
	--regressor_loss "scale" \
	--regressor_criterion "MSELoss_mean" \
	--regressor_loss_factor 0.10 \
	--regressor_zeroclamp \
	--cuda --gpus "1";

python3 training.py \
	--trainid "scale_MSELoss_200epoch_crops128_bsize32_lr0001_factor010_clamp" \
	--nEpochs 200 \
	--crop_size 128 \
	--batchSize 32 \
	--lr 0.001 \
	--regressor_loss "scale" \
	--regressor_criterion "MSELoss_mean" \
	--regressor_loss_factor 0.10 \
	--regressor_zeroclamp \
	--cuda --gpus "1";

python3 training.py \
	--trainid "scale_MSELoss_200epoch_crops256_bsize16_lr0001_factor010_clamp" \
	--nEpochs 200 \
	--crop_size 256 \
	--batchSize 16 \
	--lr 0.001 \
	--regressor_loss "scale" \
	--regressor_criterion "MSELoss_mean" \
	--regressor_loss_factor 0.10 \
	--regressor_zeroclamp \
	--cuda --gpus "1";

python3 training.py \
	--trainid "scale_MSELoss_200epoch_crops512_bsize4_lr0001_factor010_clamp" \
	--nEpochs 200 \
	--crop_size 512 \
	--batchSize 4 \
	--lr 0.001 \
	--regressor_loss "scale" \
	--regressor_criterion "MSELoss_mean" \
	--regressor_loss_factor 0.10 \
	--regressor_zeroclamp \
	--cuda --gpus "1";

# BCE


python3 training.py \
	--trainid "scale_BCELoss_200epoch_crops64_bsize128_lr0001_factor010_clamp" \
	--nEpochs 200 \
	--crop_size 64 \
	--batchSize 128 \
	--lr 0.001 \
	--regressor_loss "scale" \
	--regressor_criterion "BCELoss_mean" \
	--regressor_loss_factor 0.10 \
	--regressor_zeroclamp \
	--cuda --gpus "1";

python3 training.py \
	--trainid "scale_BCELoss_200epoch_crops128_bsize32_lr0001_factor010_clamp" \
	--nEpochs 200 \
	--crop_size 128 \
	--batchSize 32 \
	--lr 0.001 \
	--regressor_loss "scale" \
	--regressor_criterion "BCELoss_mean" \
	--regressor_loss_factor 0.10 \
	--regressor_zeroclamp \
	--cuda --gpus "1";

python3 training.py \
	--trainid "scale_BCELoss_200epoch_crops256_bsize16_lr0001_factor010_clamp" \
	--nEpochs 200 \
	--crop_size 256 \
	--batchSize 16 \
	--lr 0.001 \
	--regressor_loss "scale" \
	--regressor_criterion "BCELoss_mean" \
	--regressor_loss_factor 0.10 \
	--regressor_zeroclamp \
	--cuda --gpus "1";

python3 training.py \
	--trainid "scale_BCELoss_200epoch_crops512_bsize4_lr0001_factor010_clamp" \
	--nEpochs 200 \
	--crop_size 512 \
	--batchSize 4 \
	--lr 0.001 \
	--regressor_loss "scale" \
	--regressor_criterion "BCELoss_mean" \
	--regressor_loss_factor 0.10 \
	--regressor_zeroclamp \
	--cuda --gpus "1";

