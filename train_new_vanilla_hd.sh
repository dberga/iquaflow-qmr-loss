# scale 4
python3 training.py \
	--trainid "vanilla_200epoch_crops256_bsize16_lr0001_scale4" \
	--nEpochs 200 \
	--crop_size 256 \
	--batchSize 16 \
	--lr 0.001 \
	--n_scale 4 \
	--cuda --gpus "0";

python3 training.py \
	--trainid "vanilla_200epoch_crops128_bsize32_lr0001_scale4" \
	--nEpochs 200 \
	--n_scale 4 \
	--crop_size 128 \
	--batchSize 32 \
	--lr 0.001 \
	--cuda --gpus "0";
'''
# scale 3
python3 training.py \
	--trainid "vanilla_200epoch_crops256_bsize16_lr0001_scale3" \
	--nEpochs 200 \
	--crop_size 256 \
	--batchSize 16 \
	--lr 0.001 \
	--n_scale 3 \
	--cuda --gpus "0";

python3 training.py \
	--trainid "vanilla_200epoch_crops128_bsize32_lr0001_scale3" \
	--nEpochs 200 \
	--n_scale 3 \
	--crop_size 128 \
	--batchSize 32 \
	--lr 0.001 \
	--cuda --gpus "0";
'''
# scale 4 + noise
python3 training.py \
	--trainid "vanilla_200epoch_crops256_bsize16_lr0001_scale4_addnoise" \
	--nEpochs 200 \
	--crop_size 256 \
	--batchSize 16 \
	--lr 0.001 \
	--n_scale 4 \
        --add_noise \
	--cuda --gpus "0";

python3 training.py \
	--trainid "vanilla_200epoch_crops128_bsize32_lr0001_scale4_addnoise" \
	--nEpochs 200 \
	--n_scale 4 \
	--crop_size 128 \
	--batchSize 32 \
	--lr 0.001 \
        --add_noise \
	--cuda --gpus "0";
'''
# scale 3 + noise
python3 training.py \
	--trainid "vanilla_200epoch_crops256_bsize16_lr0001_scale3_addnoise" \
	--nEpochs 200 \
	--crop_size 256 \
	--batchSize 16 \
	--lr 0.001 \
	--n_scale 3 \
        --add_noise \
	--cuda --gpus "0";

python3 training.py \
	--trainid "vanilla_200epoch_crops128_bsize32_lr0001_scale3_addnoise" \
	--nEpochs 200 \
	--n_scale 3 \
        --add_noise \
	--crop_size 128 \
	--batchSize 32 \
	--lr 0.001 \
	--cuda --gpus "0";
'''

# scale 4 + vgg_loss
python3 training.py \
	--trainid "vanilla_200epoch_crops256_bsize16_lr0001_scale4_vggloss" \
	--nEpochs 200 \
	--crop_size 256 \
	--batchSize 16 \
	--lr 0.001 \
	--n_scale 4 \
        --vgg_loss \
	--cuda --gpus "0";

python3 training.py \
	--trainid "vanilla_200epoch_crops128_bsize32_lr0001_scale4_vggloss" \
	--nEpochs 200 \
	--n_scale 4 \
	--crop_size 128 \
	--batchSize 32 \
	--lr 0.001 \
        --vgg_loss \
	--cuda --gpus "0";
'''
# scale 3 + vgg_loss
python3 training.py \
	--trainid "vanilla_200epoch_crops256_bsize16_lr0001_scale3_vggloss" \
	--nEpochs 200 \
        --vgg_loss \
	--crop_size 256 \
	--batchSize 16 \
	--lr 0.001 \
	--n_scale 3 \
	--cuda --gpus "0";

python3 training.py \
	--trainid "vanilla_200epoch_crops128_bsize32_lr0001_scale3_vggloss" \
	--nEpochs 200 \
	--n_scale 3 \
        --vgg_loss \
	--crop_size 128 \
	--batchSize 32 \
	--lr 0.001 \
	--cuda --gpus "0";

'''

# old
'''
python3 training.py \
	--trainid "vanilla_200epoch_crops64_bsize128_lr0001" \
	--nEpochs 200 \
	--crop_size 64 \
	--batchSize 128 \
	--lr 0.001 \
	--n_scale 4 \
	--cuda --gpus "0";


python3 training.py \
	--trainid "vanilla_200epoch_crops512_bsize4_lr0001" \
	--nEpochs 200 \
	--crop_size 512 \
	--batchSize 4 \
	--lr 0.001 \
	--n_scale 4 \
	--cuda --gpus "0";
'''

