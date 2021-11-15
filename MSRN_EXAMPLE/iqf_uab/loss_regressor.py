 
import torch.nn as nn
from iq_tool_box.quality_metrics import (
    GaussianBlurMetrics,
    NoiseSharpnessMetrics,
    RERMetrics,
    ResolScaleMetrics,
    SNRMetrics,
)

def argparse_regressor_loss(argparser):
    argparser.add_argument("--regressor_loss", default=None, type=str, help="Regressor quality metric")
    argparser.add_argument("--regressor_criterion", default=None, type=str, help="Criterion for regressor quality metric loss")
    argparser.add_argument("--regressor_loss_factor", type=float, default=1.0, help="Constant to multiply by loss")
    argparser.add_argument("--regressor_zeroclamp", action="store_true", help="Clamp negative regressor losses to 0")
    argparser.add_argument("--regressor_onorm", action="store_true", help="Normalize to original loss mean")
    argparser.add_argument("--regressor_gt2onehot", action="store_true", help="Make HR regressor outputs to binary, upon max")
    return argparser


def init_regressor_loss(opt): # opt must be the output of argparse after parse_args(), containing "regressor_loss", "regressor_criterion" and "cuda" 
    if opt.regressor_loss == "rer":
        quality_metric = RERMetrics()
    elif opt.regressor_loss == "snr":
        quality_metric = SNRMetrics()
    elif opt.regressor_loss == "sigma":
        quality_metric = GaussianBlurMetrics()
    elif opt.regressor_loss == "sharpness":
        quality_metric = NoiseSharpnessMetrics()
    elif opt.regressor_loss == "scale":
        quality_metric = ResolScaleMetrics() 
    if opt.regressor_criterion == None:
        quality_metric_criterion = nn.BCELoss(reduction='mean') 
    elif opt.regressor_criterion == "BCELoss":
        quality_metric_criterion = nn.BCELoss(reduction='none')
    elif opt.regressor_criterion == "BCELoss_mean":
        quality_metric_criterion = nn.BCELoss(reduction='mean')
    elif opt.regressor_criterion == "BCELoss_sum":
        quality_metric_criterion = nn.BCELoss(reduction='sum')
    elif opt.regressor_criterion == "L1Loss":
        quality_metric_criterion = nn.L1Loss(reduction='none')
    elif opt.regressor_criterion == "L1Loss_mean":
        quality_metric_criterion = nn.L1Loss(reduction='mean')
    elif opt.regressor_criterion == "L1Loss_sum":
        quality_metric_criterion = nn.L1Loss(reduction='sum')
    elif opt.regressor_criterion == "MSELoss":
        quality_metric_criterion = nn.MSELoss(reduction='none')
    elif opt.regressor_criterion == "MSELoss_mean":
        quality_metric_criterion = nn.MSELoss(reduction='mean')
    elif opt.regressor_criterion == "MSELoss_sum":
        quality_metric_criterion = nn.MSELoss(reduction='sum')
    quality_metric_criterion.eval()
    quality_metric.regressor.net.eval()
    if opt.cuda:
        quality_metric_criterion.cuda()
        quality_metric.regressor.net.cuda()
    return quality_metric, quality_metric_criterion

def apply_regressor_loss(gt,pred,quality_metric,quality_metric_criterion,opt,loss=None,loss_spatial=None):
    output_reg = quality_metric.regressor.net(pred)
    pred_reg = nn.Sigmoid()(output_reg)
    img_reg = quality_metric.regressor.net(gt)
    regressor_loss = quality_metric_criterion(pred_reg,img_reg.detach())
    print("Original Loss")
    print(loss)
    # checking other hyperparams
    if opt.regressor_gt2onehot == True:
        img_reg_bin = torch.zeros_like(img_reg)
        for idx, hot in enumerate(img_reg):
            img_reg_bin[idx,hot.argmax()] = torch.ones_like(img_reg_bin[idx,hot.argmax()])
        regressor_loss = quality_metric_criterion(pred_reg,img_reg_bin.detach()) 
    if (opt.regressor_zeroclamp == True) and (regressor_loss < 0):
        regressor_loss = -regressor_loss * 0 #make 0 conserving tensor type
    if opt.regressor_loss_factor != 1.0: #multiply by constant factor
        regressor_loss = regressor_loss * opt.regressor_loss_factor
    if opt.regressor_onorm == True:
        regressor_loss = regressor_loss*torch.mean(loss_spatial)
    print("Regressor Loss")
    print(regressor_loss)
    return regressor_loss, img_reg, pred_reg