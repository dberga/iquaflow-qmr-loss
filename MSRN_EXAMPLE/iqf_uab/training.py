#to do: incluir summarywriter de tensorboard para iqf
#to do: ejecutar con crops de 512 y 1024 con menor batch size?
#to do: incluir patience para no depender del nEpochs
#to do: fix regressor scale i sharpness
#to do: poner una funcionalidad para hacer un target parameter 'optimo' en términos de rer, snr, etc (ya que el regressor se entrenó con onehot). se deberia probar este caso para downscalings mayores, seguramente mejoraría
import argparse, os, json
import sys
import torch
import math, random
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
import numpy as np
from datetime import datetime;

from metrics import PSNR, SSIM, FID
from models.msrn import MSRN_Upscale
from dataset_hr_lr import DatasetHR_LR
from torch.utils.data import DataLoader
from models.perceptual_loss import VGGPerceptualLoss

from loss_regressor import argparse_regressor_loss, init_regressor_loss, apply_regressor_loss

# Training settings
parser = argparse.ArgumentParser(description="PyTorch MSRN")
parser.add_argument("--trainid", default=None, type=str, help="Training id to save tensorboard logs")
parser.add_argument("--batchSize", type=int, default=16, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=1500, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=0.001, help="Learning Rate. Default=1e-3")
parser.add_argument("--cuda", action="store_true", help="Use cuda?")
parser.add_argument("--colorjitter", action="store_true", help="Use colorjitter?")
parser.add_argument("--add_noise", action="store_true", help="Use cuda?")
parser.add_argument("--vgg_loss", action="store_true", help="Use perceptual loss?")
parser.add_argument("--resume", action="store_true", help="take last epoch available")
parser.add_argument("--threads", type=int, default=0, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--start_epoch", type=int, default=0, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--step", type=int, default=50, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=500")
parser.add_argument("--gpus", default="1", type=str, help="gpu ids (default: 0)")
parser.add_argument("--seed", default="12345", type=str, help="random seed")
parser.add_argument("--path_out", default="msrn/experiment/", type=str, help="path output")
parser.add_argument("--trainds_input", default="test_datasets/AerialImageDataset/train/images", type=str, help="path input training")
parser.add_argument("--valds_input", default="test_datasets/AerialImageDataset/test/images", type=str, help="path input val")
parser.add_argument("--crop_size", type=int, default=512, help="Crop size")
parser.add_argument("--nockpt", action="store_true", help="Flag to not save checkpoint")
parser = argparse_regressor_loss(parser)

class noiseLayer_normal(nn.Module):
    def __init__(self, noise_percentage, mean=0, std=0.2):
        super(noiseLayer_normal, self).__init__()
        self.n_scale = noise_percentage
        self.mean=mean
        self.std=std

    def forward(self, x):
        if self.training:
            noise_tensor = torch.normal(self.mean, self.std, size=x.size()).to(x.get_device()) 
            x = x + noise_tensor * self.n_scale
        
            mask_high = (x > 1.0)
            mask_neg = (x < 0.0)
            x[mask_high] = 1
            x[mask_neg] = 0

        return x

def main():

    global opt, model
    opt = parser.parse_args()
    os.makedirs(opt.path_out, exist_ok=True)
    tt = datetime.now()
    ttdate = tt.strftime("%m-%d-%Y_%H:%M:%S")
    if opt.trainid == None:
        opt.trainid = "run_"+ttdate
    path_logs = os.path.join(opt.path_out,opt.trainid)
    path_checkpoints = os.path.join(path_logs, "checkpoint_"+opt.trainid)
    os.makedirs(path_logs, exist_ok=True)
    os.makedirs(path_checkpoints, exist_ok=True)
    writer = SummaryWriter(path_logs)
    with open(os.path.join(path_logs,'config.txt'), 'w') as f: #save argparse params config in text
        json.dump(opt.__dict__,f,indent=2)

    print(opt)

    cuda = opt.cuda
    if cuda:
        print("=> use gpu id: '{}'".format(opt.gpus))
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"]=opt.gpus
        if not torch.cuda.is_available():
                raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)


    print("===> Loading datasets")
    train_set = DatasetHR_LR("training", crop_size=(opt.crop_size,opt.crop_size), apply_color_jitter=opt.colorjitter, input_path=opt.trainds_input)
    validation_set = DatasetHR_LR("validation", crop_size=(opt.crop_size,opt.crop_size), input_path=opt.valds_input)

    dataloaders ={
        'training': DataLoader(dataset=train_set, num_workers=opt.threads, \
                        batch_size=opt.batchSize, shuffle=True),
        'validation': DataLoader(dataset=validation_set, num_workers=opt.threads, \
            batch_size=opt.batchSize, shuffle=False)}

    print("===> Building model")
    model = MSRN_Upscale(n_scale=2)
    # pretrained
    criterion = nn.L1Loss(reduction='none')
    
    print("INIT PIXEL SHUFFLE!!")
    model._init_pixel_shuffle()
    
    if opt.vgg_loss:
        global perceptual_loss
        perceptual_loss = VGGPerceptualLoss()
        perceptual_loss.eval()
        perceptual_loss.cuda()
        print("Using perceptual loss")

    if opt.regressor_loss is not None:
        global quality_metric
        global quality_metric_criterion
        quality_metric , quality_metric_criterion = init_regressor_loss(opt)
        print("Using regressor loss")

    print("===> Setting GPU")
    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()
        
    # optionally resume from a checkpoint
    if opt.resume:
        list_epochs = [int(f.split('.')[0].split('_')[-1]) for f in os.listdir(path_checkpoints)]
        list_epochs.sort()
        last_epoch = list_epochs[-1]
        print(" resume from ", last_epoch)
        weights = torch.load(os.path.join(path_checkpoints, f"model_epoch_{last_epoch}.pth"))
        model.load_state_dict(weights["model"].state_dict())
        opt.start_epoch = weights["epoch"] + 1
        
    
#     for name,param in model.named_parameters():
#         param.requires_grad = False
#         if 'conv_up' in name:
#             param.requires_grad = True
#         if 'conv_output' in name:
#             param.requires_grad = True
            
    for name,param in model.named_parameters():
        print(name, param.requires_grad)
    print("===> Setting Optimizer")
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    print("===> Training")
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        for mode in ['training', 'validation']:
            train(mode, dataloaders, optimizer, model, criterion, epoch, writer)
            if (mode == 'training') & (opt.nockpt != True):
                save_checkpoint(model, epoch, path_checkpoints)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10"""
    lr = opt.lr * (0.1 ** (epoch // opt.step))
    return lr 

def train(mode, dataloader, optimizer, model, criterion, epoch, writer):
    
    metric_psnr = PSNR()
    metric_ssim = SSIM()
    metric_fid = FID()

#     lr = adjust_learning_rate(optimizer, epoch-1)
#     print("learning rate", lr)
#     for param_group in optimizer.param_groups:
#         param_group["lr"] = lr

    print("{}\t Epoch={}, lr={}".format(mode, epoch, optimizer.param_groups[0]["lr"]))
    model.train()
    if mode=='validation':
        model.eval()

    for iteration, batch in enumerate(dataloader[mode], 1):

        img_lr, img_hr = batch
        
        if opt.cuda:
            img_lr = img_lr.cuda()
            img_hr = img_hr.cuda()
            
        if opt.add_noise:
            scale_noise = np.random.choice(np.arange(0.05, 0.2, 0.01))
            add_noise = noiseLayer_normal(scale_noise, mean=0, std=0.2)
            img_lr = add_noise(img_lr)

        output = model(img_lr)
        loss_spatial = criterion(img_hr, output)
        loss = torch.mean(loss_spatial)
        
        if opt.vgg_loss:
            vgg_loss,_ = perceptual_loss(output, img_hr)
            loss = loss + 10*vgg_loss

        if opt.regressor_loss is not None:
            regressor_loss = apply_regressor_loss(img_hr,output,quality_metric,quality_metric_criterion,opt,loss,loss_spatial)
            loss = loss + regressor_loss    
        
        #Metrics
        psnr = metric_psnr(img_hr, output)
        ssim = metric_ssim(img_hr, output)
        fid = metric_fid(img_hr, output)

        if mode=='training':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        grid_lr = torchvision.utils.make_grid(img_lr)[[2, 1, 0],...]
        grid_hr = torchvision.utils.make_grid(img_hr)[[2, 1, 0],...]
        grid_pred = torchvision.utils.make_grid(output)[[2, 1, 0],...]

        if iteration%10 == 0:
            print("===>{}\tEpoch[{}]({}/{}): Loss: {:.5} \t PSNR: {:.5} \t SSIM: {:.5} \t FID: {:.5}".format(mode, epoch, iteration, len(dataloader[mode]), loss.item(), psnr.item(), ssim.item(), fid.item()))
            writer.add_scalar(f'{mode}/LOSS/', loss.item(), epoch*len(dataloader[mode])+iteration)
            writer.add_scalar(f'{mode}/PSNR', psnr.item(), epoch*len(dataloader[mode])+iteration)
            writer.add_scalar(f'{mode}/SSIM', ssim.item(), epoch*len(dataloader[mode])+iteration)
            writer.add_scalar(f'{mode}/FID', fid.item(), epoch*len(dataloader[mode])+iteration)
            writer.add_image(f'{mode}/lr', grid_lr, iteration)
            writer.add_image(f'{mode}/hr', grid_hr, iteration)
            writer.add_image(f'{mode}/pred', grid_pred, iteration)
            if opt.regressor_loss is not None:
                writer.add_scalar(f'{mode}/REG_LOSS_{type(quality_metric_criterion).__name__}/', regressor_loss.item(), epoch*len(dataloader[mode])+iteration)
                writer.add_scalar(f'{mode}/LOSS-REG_LOSS_{type(quality_metric_criterion).__name__}/', (loss-regressor_loss).item(), epoch*len(dataloader[mode])+iteration)
                for i in range(len(pred_reg)):
                    writer.add_histogram(f'{mode}/REG_pred_{opt.regressor_loss}/', quality_metric.regressor.yclasses[opt.regressor_loss][torch.argmax(pred_reg, dim=1)[i].item()], epoch*len(dataloader[mode])+iteration)
                    writer.add_histogram(f'{mode}/REG_HR_{opt.regressor_loss}/', quality_metric.regressor.yclasses[opt.regressor_loss][torch.argmax(img_reg, dim=1)[i].item()], epoch*len(dataloader[mode])+iteration)
                
def save_checkpoint(model, epoch, path_checkpoints):       
    os.makedirs(path_checkpoints, exist_ok=True)
    model_out_path = os.path.join(path_checkpoints + f"model_epoch_{epoch}.pth".format(epoch))
    state = {"epoch": epoch ,"model": model}
    torch.save(state, model_out_path)

    print("Checkpoint saved to {}".format(model_out_path))

if __name__ == "__main__":
    main()
