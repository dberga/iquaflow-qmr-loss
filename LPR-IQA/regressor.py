# matplotlib inline
import argparse
import configparser
import glob
import json
import os
import shutil
import time
from bisect import bisect_right

import numpy as np
import torch
import torchvision.models as models
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from torchvision.utils import save_image

from iq_tool_box.datasets import (
    DSModifier,
    DSModifier_blur,
    DSModifier_resol,
    DSModifier_sharpness,
    DSWrapper,
)


def parse_params_cfg(cfg_path="config.cfg", recursive=False):
    if recursive is False:
        default_config = configparser.ConfigParser()
        default_config.read(cfg_path)

        aux_parser = argparse.ArgumentParser()
        aux_parser.add_argument(
            "--resume",
            default=default_config["RUN"].getboolean("resume"),
            action="store_true",
        )
        aux_parser.add_argument("--trainid", default=default_config["RUN"]["trainid"])
        aux_parser.add_argument(
            "--outputpath", default=default_config["PATHS"]["outputpath"]
        )
        run_args = aux_parser.parse_args()

        ckpt_cfg_path = os.path.join(run_args.outputpath, run_args.trainid, cfg_path)

        if run_args.resume is True and os.path.exists(
            ckpt_cfg_path
        ):  # if resume read cfg file from cfg_path if provided
            return parse_params_cfg(ckpt_cfg_path, True)
        else:
            return parse_params_cfg(cfg_path, True)

    # extract cfg config
    else:
        print(cfg_path)
        config = configparser.ConfigParser()
        config.read(cfg_path)
        parser = argparse.ArgumentParser()
        parser.add_argument("--trainid", default=config["RUN"]["trainid"])
        parser.add_argument(
            "--resume", default=config["RUN"].getboolean("resume"), action="store_true"
        )
        parser.add_argument(
            "--trainds", default=config["PATHS"]["trainds"]
        )  # inria-aid, "xview", "ds_coco_dataset"
        parser.add_argument("--traindsinput", default=config["PATHS"]["traindsinput"])
        parser.add_argument("--valds", default=config["PATHS"]["valds"])
        parser.add_argument("--valdsinput", default=config["PATHS"]["valdsinput"])
        parser.add_argument("--outputpath", default=config["PATHS"]["outputpath"])
        try:
            parser.add_argument(
                "--modifier_params",
                default=eval(config["HYPERPARAMS"]["modifier_params"]),
            )  # for numpy commands (e.g. np.linspace(min,max,num_reg))
        except Exception:
            parser.add_argument(
                "--modifier_params",
                default=json.loads(config["HYPERPARAMS"]["modifier_params"]),
            )  # dict format
        parser.add_argument(
            "--num_regs", default=json.loads(config["HYPERPARAMS"]["num_regs"])
        )
        parser.add_argument(
            "--epochs", default=json.loads(config["HYPERPARAMS"]["epochs"])
        )
        parser.add_argument(
            "--num_crops", default=json.loads(config["HYPERPARAMS"]["num_crops"])
        )
        parser.add_argument(
            "--splits", default=json.loads(config["HYPERPARAMS"]["splits"])
        )
        parser.add_argument(
            "--input_size", default=json.loads(config["HYPERPARAMS"]["input_size"])
        )
        parser.add_argument(
            "--batch_size", default=json.loads(config["HYPERPARAMS"]["batch_size"])
        )  # samples per batch (* NUM_CROPS * DEFAULT_SIGMAS). (eg. 2*4*4)
        parser.add_argument("--lr", default=json.loads(config["HYPERPARAMS"]["lr"]))
        parser.add_argument(
            "--momentum", default=json.loads(config["HYPERPARAMS"]["momentum"])
        )
        parser.add_argument(
            "--weight_decay", default=json.loads(config["HYPERPARAMS"]["weight_decay"])
        )
        parser.add_argument(
            "--workers", default=json.loads(config["HYPERPARAMS"]["workers"])
        )
        parser.add_argument(
            "--data_shuffle", default=config["HYPERPARAMS"]["data_shuffle"]
        )

        # save config in ckpt folder
        run_args = parser.parse_args()
        ckpt_folder = os.path.join(run_args.outputpath, run_args.trainid)
        ckpt_cfg_path = os.path.join(ckpt_folder, cfg_path)

        if not os.path.exists(run_args.outputpath):  # create main output folder
            os.mkdir(run_args.outputpath)
        if not os.path.exists(ckpt_folder):
            os.mkdir(ckpt_folder)
        if not ckpt_cfg_path == cfg_path:  # if not read from resume
            if not os.path.exists(ckpt_cfg_path):  # new
                shutil.copyfile(cfg_path, ckpt_cfg_path)
            else:
                os.remove(ckpt_cfg_path)  # overwrite old
                shutil.copyfile(cfg_path, ckpt_cfg_path)
    return parser


def get_modifiers_from_params(modifier_params):
    ds_modifiers = []
    if len(modifier_params.items()) == 0:
        ds_modifiers = DSModifier()
    else:
        for key, elem in modifier_params.items():
            for gidx in range(len(elem)):
                if key == "sigma":
                    ds_modifiers.append(
                        DSModifier_blur(params={key: modifier_params[key][gidx]})
                    )
                elif key == "scale":
                    if not "interpolation" in modifier_params.keys(): # default rescale interpolation
                        ds_modifiers.append(
                            DSModifier_resol(params={key: modifier_params[key][gidx], "interpolation": 2})
                        )
                    else:
                        ds_modifiers.append(
                            DSModifier_resol(params={key: modifier_params[key][gidx], "interpolation": modifier_params["interpolation"]})
                        )
                elif key == "sharpness":
                    ds_modifiers.append(
                        DSModifier_sharpness(params={key: modifier_params[key][gidx]})
                    )
    return ds_modifiers


def get_regression_interval_classes(modifier_params, num_regs):
    yclasses = {}
    params = list(modifier_params.keys())
    for idx, param in enumerate(params):
        param_items = modifier_params[param]
        yclasses[param] = np.linspace(
            np.min(param_items), np.max(param_items), num_regs[idx]
        )
    return yclasses


def force_rgb(x):
    if x.shape[0] < 3:  # if grayscale, replicate channel to rgb
        x = torch.cat([x] * 3)
    elif (
        x.shape[0] > 3
    ):  # if >3 dimensions, use first 3 to discard other dimensions (e.g. depth)
        x = x[0:3]
    return x


def pred2soft(prediction, threshold=0.7):
    output_soft = (torch.sigmoid(prediction) > threshold).float()
    for idx, hot in enumerate(output_soft):
        if hot.sum() == 0:
            hot[prediction[idx].argmax()] = 1
            output_soft[idx] = hot
    return output_soft


def get_precision(output_soft, target, threshold=0.5):
    # calculate precision
    TP = float(torch.sum((output_soft >= threshold) & ((target == 1))))
    FP = float(torch.sum((output_soft >= threshold) & (target == 0)))
    if (TP + FP) > 0:
        prec = torch.tensor((TP) / (TP + FP))
    else:
        prec = torch.tensor(0.0)
    return prec



class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, data_input, num_crops, crop_size):
        self.data_path = data_path
        self.data_input = data_input
        self.num_crops = num_crops
        # list images
        self.lists_files = [
            self.data_input + "/" + filename for filename in os.listdir(self.data_input)
        ]
        self.lists_mod_files = []
        self.lists_crop_files = []
        # keys
        self.mod_keys = []
        self.crop_mod_keys = []
        # params
        self.mod_params = []
        self.crop_mod_params = []
        self.mod_resol = []

        #transforms
        self.tCROP = transforms.Compose(
            [
                transforms.RandomCrop(size=(crop_size[0], crop_size[1])),
            ]
        )
        self.cCROP = transforms.Compose(
            [
                transforms.CenterCrop(size=(crop_size[0], crop_size[1])),
            ]
        )

    def __len__(self):
        """
        if len(self.lists_mod_files) is 0:
            return len(self.lists_mod_files)*self.num_crops
        else:
            return len(self.lists_files)*self.num_crops*len(self.lists_mod_files)
        """
        return int(
            np.max(
                [
                    len(self.lists_files),
                    len(self.lists_mod_files) * len(self.lists_files),
                    len(self.lists_crop_files),
                ]
            )
        )

    def __getitem__(self, idx):
        if (
            len(self.lists_crop_files) > 0 and len(self.lists_mod_files) > 0
        ):  # cropped and modified
            filename = self.lists_crop_files[idx]
            # filename_noext = os.path.splitext(os.path.basename(filename))[0]
            image = Image.open(filename)
            image_tensor = transforms.functional.to_tensor(image)  # .unsqueeze_(0)
            x = force_rgb(image_tensor)  # usgs case (nth dimensions, e.g. depth)
            y = torch.tensor(self.crop_mod_params[idx])
            param = self.crop_mod_keys[idx]
        elif (
            len(self.lists_crop_files) == 0 and len(self.lists_mod_files) != 0
        ):  # modified but not cropped
            filename = self.lists_mod_files[idx]
            # filename_noext = os.path.splitext(os.path.basename(filename))[0]
            image = Image.open(filename)
            image_tensor = transforms.functional.to_tensor(image).unsqueeze_(0)
            x = force_rgb(image_tensor)  # usgs case (nth dimensions, e.g. depth)
            y = torch.tensor(self.mod_params[idx])
            param = self.mod_keys[idx]
        elif (
            len(self.lists_crop_files) > 0 and len(self.lists_mod_files) == 0
        ):  # cropped but no modified
            filename = self.lists_crop_files[idx]
            # filename_noext = os.path.splitext(os.path.basename(filename))[0]
            image = Image.open(filename)
            image_tensor = transforms.functional.to_tensor(image)  # .unsqueeze_(0)
            x = force_rgb(image_tensor)  # usgs case (nth dimensions, e.g. depth)
            y = torch.tensor(0)
            param = ""
        else:
            filename = self.lists_files[idx]
            # filename_noext = os.path.splitext(os.path.basename(filename))[0]
            image = Image.open(filename)
            image_tensor = transforms.functional.to_tensor(image).unsqueeze_(0)
            x = force_rgb(image_tensor)
            y = torch.tensor(0)
            param = ""

        return param, Variable(x), Variable(y)

    def __modify__(self, ds_modifiers):
        self.lists_mod_files = []  # one per each modifier
        self.mod_keys = []  # one per each modifier
        self.mod_params = []  # one per each modifier
        for midx in range(len(ds_modifiers)):
            ds_modifier = ds_modifiers[midx]
            mod_key = next(
                iter(ds_modifier.params.keys())
            )  # parameter name of modifier
            mod_param = next(
                iter(ds_modifier.params.values())
            )  # parameter value of modifier
            print("Preprocessing files with " + mod_key + " " + str(mod_param))
            ds_wrapper = DSWrapper(data_path=self.data_path, data_input=self.data_input)
            # ds_wrapper.data_input=self.data_input
            output_dir = self.data_path + "#" + ds_modifier.name
            split_name = os.path.basename(self.data_input)
            split_dir = os.path.join(output_dir, split_name)
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            if os.path.exists(split_dir):
                shutil.rmtree(split_dir)
            else:
                os.mkdir(split_dir)
            ds_wrapper_modified = ds_modifier.modify_ds_wrapper(ds_wrapper=ds_wrapper)
            # if 'test_images' in split_dir:
            #    import pdb; pdb.set_trace()
            os.listdir(ds_wrapper_modified.data_input)
            self.lists_mod_files.append(
                [
                    ds_wrapper_modified.data_input + "/" + filename
                    for filename in os.listdir(ds_wrapper_modified.data_input)
                ]
            )
            self.mod_keys.append(mod_key)
            self.mod_params.append(mod_param)
            # print(self.lists_mod_files[-1])

    def __crop__(
        self,
    ):  # todo: use random or exact same crop positions for all modifiers? https://discuss.pytorch.org/t/cropping-batches-at-the-same-position/24550/5
        self.list_crop_files = []
        self.crop_mod_keys = []  # one per each modifier
        self.crop_mod_params = []  # one per each modifier
        self.mod_resol = []  # one per each modifier
        if len(self.lists_mod_files) != 0:
            for midx, list_mod_files in enumerate(
                self.lists_mod_files
            ):  # for each modifier
                for idx, mod_files in enumerate(list_mod_files):  # for each sample
                    filename = self.lists_mod_files[midx][idx]
                    filename_noext = os.path.splitext(os.path.basename(filename))[0]
                    image = Image.open(filename)
                    image_tensor = transforms.functional.to_tensor(image).unsqueeze_(0)
                    # if "train_images" in filename:
                    #     import pdb; pdb.set_trace()
                    for cidx in range(self.num_crops):
                        # print("Generating crop ("+str(cidx+1)+"/"+str(self.num_crops)+")")
                        preproc_image = self.tCROP(image_tensor)
                        crops_folder = os.path.dirname(filename) + "_crops"
                        if not os.path.exists(crops_folder):
                            os.mkdir(crops_folder)
                        filename_cropped = (
                            crops_folder
                            + "/"
                            + filename_noext
                            + "_crop"
                            + str(cidx + 1)
                            + ".png"
                        )
                        self.lists_crop_files.append(filename_cropped)
                        self.crop_mod_keys.append(self.mod_keys[midx])
                        self.crop_mod_params.append(self.mod_params[midx])
                        self.mod_resol.append(image_tensor.shape[2:4])
                        save_image(preproc_image, filename_cropped)
                        # print(self.lists_crop_files[-1])  # print last sample name
                    # os.remove(filename) # remove modded image to clean disk
        else:
            for idx, file in enumerate(self.lists_files):
                filename = self.lists_files[idx]
                filename_noext = os.path.splitext(os.path.basename(filename))[0]
                image = Image.open(filename)
                image_tensor = transforms.functional.to_tensor(image).unsqueeze_(0)
                for cidx in range(self.num_crops):
                    preproc_image = self.tCROP(image_tensor)
                    crops_folder = os.path.dirname(filename) + "_crops"
                    if not os.path.exists(crops_folder):
                        os.mkdir(crops_folder)
                    filename_cropped = (
                        crops_folder
                        + "/"
                        + filename_noext
                        + "_crop"
                        + str(cidx + 1)
                        + ".png"
                    )
                    self.lists_crop_files.append(filename_cropped)
                    self.mod_resol.append(image_tensor.shape[2:4])
                    save_image(preproc_image, filename_cropped)
                    # print(self.lists_crop_files[-1])  # print last sample name
                os.remove(filename)  # remove modded image to clean disk


class MultiHead(torch.nn.Module):  # deprecated
    def __init__(self, *modules: torch.nn.Module):
        super().__init__()
        self.modules = modules

    def forward(self, inputs):
        return [module(inputs) for module in self.modules]


class MultiHead_ResNet(torch.nn.Module):
    def __init__(
        self, network=models.resnet18(pretrained=True), *heads: torch.nn.Module
    ):
        super().__init__()
        # import pdb; pdb.set_trace()
        self.network = network
        self.network.fc = torch.nn.Sequential()  # remove fc
        self.network.heads = heads

    def forward(self, inputs):
        x = self.network(inputs)
        return [head(x) for head in self.network.heads]


class Regressor:
    def __init__(self, args):
        self.train_ds = args.trainds
        self.train_ds_input = args.traindsinput
        self.val_ds = args.valds
        self.val_ds_input = args.valdsinput
        self.epochs = int(args.epochs)
        self.lr = float(args.lr)
        self.momentum = float(args.momentum)
        self.weight_decay = float(args.weight_decay)
        self.batch_size = int(args.batch_size)

        # Get Regressor Params from dicts
        self.modifier_params = args.modifier_params
        self.num_regs = list(args.num_regs)
        self.num_crops = args.num_crops  # used only by Dataset

        # Define modifiers and regression intervals
        self.ds_modifiers = get_modifiers_from_params(self.modifier_params)
        self.params = list(self.modifier_params.keys())
        self.dict_params = {par: idx for idx, par in enumerate(self.params)}
        self.yclasses = get_regression_interval_classes(
            self.modifier_params, self.num_regs
        )

        crop_size=str(args.input_size)
        self.tCROP = transforms.Compose(
            [
                transforms.RandomCrop(size=(crop_size[0], crop_size[1])),
            ]
        )  # define torch transform
        self.cCROP = transforms.Compose(
            [
                transforms.CenterCrop(size=(crop_size[0], crop_size[1])),
            ]
        )  # define torch transform

        # Create Network
        if len(self.modifier_params.keys()) == 1:  # Single Head
            self.net = models.resnet18(pretrained=True)
            self.net.fc = torch.nn.Linear(512, self.num_regs[0])
        else:  # MultiHead
            self.net = MultiHead_ResNet(
                models.resnet18(pretrained=True),
                *[
                    torch.nn.Linear(512, self.num_regs[idx])
                    for idx in range(len(self.params))
                ]
            )
            # self.net=models.resnet18(pretrained=True)
            # self.net.fc=MultiHead(*[torch.nn.Linear(512, self.num_regs[idx]) for idx in range(len(self.params))])

        # Training HyperParams
        self.optimizer = torch.optim.SGD(
            self.net.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        self.criterion = torch.nn.BCELoss()

        # Output Paths
        self.output_path = args.outputpath
        self.train_id = args.trainid
        self.output_path = os.path.join(self.output_path, self.train_id)
        self.checkpoint_name = "checkpoint" + "_epoch" + str(self.epochs) + ".pth"
        self.checkpoint_path = os.path.join(
            self.output_path, self.checkpoint_name
        )  # add join names for params

    # GPU CASE
    def train_val(self, train_loader, val_loader):
        best_loss = np.inf
        best_prec = 0.0
        train_losses = []
        val_losses = []
        train_precs = []
        val_precs = []
        for epoch in range(self.epochs):  # epoch
            train_loss, train_prec = self.train(train_loader, epoch)
            val_loss, val_prec = self.validate(val_loader, epoch)
            if val_loss is None or train_loss is None:
                import pdb

                pdb.set_trace()
            is_best = (val_loss < best_loss) & (val_prec > best_prec)
            if is_best:
                best_loss = val_loss
                best_prec = val_prec
                # remove previous checkpoint
                if os.path.exists(
                    self.checkpoint_path
                ):  # 'checkpoint_path' in dir(self) &
                    os.remove(self.checkpoint_path)
                # save checkpoint
                self.checkpoint_name = "checkpoint" + "_epoch" + str(epoch) + ".pth"
                self.checkpoint_path = os.path.join(
                    self.output_path, self.checkpoint_name
                )
                print("Found best model, saving checkpoint " + self.checkpoint_path)
                torch.save(self.net, self.checkpoint_path)  # self.net.state_dict()

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_precs.append(train_prec)
            val_precs.append(val_prec)
            print(
                "Train Loss: "
                + str(train_loss)
                + " Train Precision: "
                + str(train_prec)
            )
            print(
                "Validation Loss: "
                + str(val_loss)
                + " Validation Precision: "
                + str(val_prec)
            )
            np.savetxt(
                os.path.join(self.output_path, "stats.csv"),
                np.asarray([train_losses, val_losses, train_precs, val_precs]),
                delimiter=",",
            )

    def deploy(self, image_files):  # load checkpoint and run an image path
        # load latest checkpoint
        if not os.path.exists(
            self.checkpoint_path
        ):  # if doesnt exist, list all and read latest
            list_of_checkpoints = glob.glob(self.output_path + "/*.pth")
            if len(list_of_checkpoints) > 0:
                self.checkpoint_path = max(list_of_checkpoints, key=os.path.getctime)
            else:
                self.checkpoint_path = None

        if self.checkpoint_path:
            self.net = torch.load(self.checkpoint_path)
            # self.net.load_state_dict(torch.load(self.checkpoint_path))
        else:
            print("Could not find any checkpoint, closing deploy...")
            return []

        # print(image_files)

        """ # Center crop
        # prepare data
        x = []
        for idx in range(len(image_files)):
            filename = image_files[idx]
            filename_noext = os.path.splitext(os.path.basename(filename))[0]
            # image = io.imread(fname=filename)
            image = Image.open(filename)
            image_tensor = transforms.functional.to_tensor(image).unsqueeze_(0)
            preproc_image = self.cCROP(
                image_tensor
            )  # todo: maybe replace this deploy by several crops and select most frequent?
            preproc_image=torch.unsqueeze(force_rgb(preproc_image[0,:]),dim=0)
            x.append(preproc_image)
            save_image(preproc_image, os.path.join(self.output_path,os.path.basename(filename)))
        x = torch.cat(x, dim=0)
        """
        # N Random Crops
        x = []
        for idx in range(len(image_files)):
            filename = image_files[idx]
            # filename_noext = os.path.splitext(os.path.basename(filename))[0]
            image = Image.open(filename)
            image_tensor = transforms.functional.to_tensor(image).unsqueeze_(0)
            if image_tensor.shape[2] != CROP_SIZE[0]:  # whole satellite image or crop?
                xx = []
                for cidx in range(self.num_crops):
                    # print("Generating crop ("+str(cidx+1)+"/"+str(self.num_crops)+")")
                    preproc_image = self.tCROP(image_tensor)
                    preproc_image = torch.unsqueeze(
                        force_rgb(preproc_image[0, :]), dim=0
                    )
                    xx.append(preproc_image)
                    save_image(
                        preproc_image,
                        os.path.join(self.output_path, os.path.basename(filename)),
                    )
                xx = torch.cat(xx, dim=0)
                x.append(xx)
            else:
                x.append(image_tensor)
        # run for each image crop
        reg_values = dict((par, []) for par in self.params)
        for idx, crops in enumerate(x):
            pred = self.net(crops)
            if len(self.params) == 1:  # Single head prediction
                par = self.params[0]
                pmax = []
                for i, prediction in enumerate(pred):
                    pmax.append(pred[i].argmax())
                reg_values[par].append(
                    max(set(pmax), key=pmax.count)
                )  # save reg index values with most occurencies (for each image crops, select most common param in pred list)
            else:  # Multi head prediction
                for pidx, par in enumerate(self.params):
                    pmax = []
                    for i, prediction in enumerate(pred[pidx]):
                        pmax.append(pred[pidx][i].argmax())
                    reg_values[par].append(
                        max(set(pmax), key=pmax.count)
                    )  # save reg index values with most occurencies (for each image crops, select most common param in pred list)
        # prepare output_json
        if len(self.params) == 1:
            output_values = []
            par = self.params[0]
            for i, regval in enumerate(reg_values[par]):
                value = self.yclasses[par][regval]
                output_values.append(value)
                output_json = {par: value, "path": image_files[i]}
                print(" pred " + str(output_json))
        else:
            output_values = [[] for par in self.params]
            for i, crops in enumerate(x):
                output_json = {}
                for pidx, par in enumerate(self.params):
                    regval = reg_values[par][i]
                    value = self.yclasses[par][regval]
                    output_json[par] = value
                    output_values[pidx].append(value)
                output_json["path"] = image_files[i]
                print(" pred " + str(output_json))
        return output_values

    def train(self, train_loader, epoch):
        self.net.train()  # train mode
        losses = np.array([])
        precs = np.array([])
        # xbatches=[x for bix,(x,y) in enumerate(train_loader)]
        # ybatches=[y for bix,(x,y) in enumerate(train_loader)]
        for bidx, (param, x, y) in enumerate(
            train_loader
        ):  # ongoing: if net is outputting several outputs, separate target and prediction for each param and make sure evaluate that specific head
            # join regression batch for crops and modifiers
            x.requires_grad = True
            # transform sigmas to regression intervals (yreg)
            param = [
                self.params[0] if par == "" else par for par in param
            ]  # if param is empty, set to first param in params list
            yreg = torch.stack(
                [
                    torch.tensor(
                        bisect_right(self.yclasses[param[i]], y[i]) - 1,
                        dtype=torch.long,
                    )
                    for i in range(len(y))
                ]
            )
            yreg = Variable(yreg)
            prediction = self.net(x)  # input x and predict based on x, [b,:,:,:]
            if len(self.params) == 1:
                target = torch.eye(self.num_regs[0])[yreg]
                pred = torch.nn.Sigmoid()(prediction)

                # calc loss
                loss = self.criterion(pred, target)  # yreg as alternative (classes)

                # output to soft encoding (threshold output and compute) to get TP,FP...
                output_soft = pred2soft(prediction)
                prec = get_precision(output_soft, target)
                """
                par = self.params[0]
                for i, tgt in enumerate(target):
                    output_json = {par: self.yclasses[par][prediction[i].argmax()]}
                    target_json = {par: self.yclasses[par][target[i].argmax()]}
                    # print("target "+str(target_json)+" pred "+str(output_json)+" batch "+str(bidx+1))
                """
            else:
                # compute losses differently for each head
                loss = 0.0
                pprec = []
                param_ids = [self.dict_params[par] for par in param]
                for pidx, par in enumerate(self.params):
                    param_indices = [
                        ppidx for ppidx, pid in enumerate(param_ids) if pid == pidx
                    ]
                    if len(param_indices) == 0:
                        continue
                    param_yreg = yreg[param_indices]
                    param_target = torch.eye(self.num_regs[pidx])[
                        param_yreg
                    ]  # one-hot encoding
                    param_prediction = prediction[pidx][param_indices]
                    param_pred = torch.nn.Sigmoid()(param_prediction)
                    param_loss = self.criterion(
                        param_pred, param_target
                    )  # one loss for each param

                    loss += param_loss  # final loss is sum of all param BCE losses
                    # todo: check if this loss computation works, or losses need to be adapted to each param/head?
                    # output to soft encoding (threshold output and compute) to get TP,FP...
                    output_soft = pred2soft(param_prediction)
                    param_prec = get_precision(output_soft, param_target)
                    pprec.append(param_prec)

                    """
                    # print json values (converted from onehot to param interval values)
                    for i, tgt in enumerate(param_target):
                        output_json = {par: self.yclasses[par][param_pred[i].argmax()]}
                        target_json = {
                            par: self.yclasses[par][param_target[i].argmax()]
                        }
                        # print("target "+str(target_json)+" pred "+str(output_json)+" batch "+str(bidx+1))
                    """
                prec = np.nanmean(pprec)  # use mean or return separate precs?
            precs = np.append(precs, prec)
            losses = np.append(losses, loss.data.numpy())

            # backprop
            self.optimizer.zero_grad()  # clear gradients for next train
            loss.backward()  # backpropagation, compute gradients
            self.optimizer.step()  # apply gradients

            # print("Debug (check sigma intervals (one-hot encoding) of target and prediction)")
            # print("Target:")
            # print(target.squeeze())
            # print("Prediction:")
            # print(pred.squeeze())

        epoch_loss = np.nanmean(losses)
        epoch_prec = np.nanmean(precs)
        print(
            "Train Step = %d" % epoch
            + " Loss = %.4f" % epoch_loss
            + " Precision = %0.4f" % epoch_prec
        )  # +' Acc = %.4f' % epoch_acc
        return epoch_loss, epoch_prec

    def validate(self, val_loader, epoch):
        with torch.no_grad():
            # val
            self.net.eval()  # val mode
            losses = np.array([])
            precs = np.array([])
            # xbatches=[x for bix,(x,y) in enumerate(val_loader)]
            # ybatches=[y for bix,(x,y) in enumerate(val_loader)]
            for bidx, (param, x, y) in enumerate(val_loader):
                # join regression batch for crops and modifiers
                x.requires_grad = True
                # transform sigmas to regression intervals (yreg)
                param = [
                    self.params[0] if par == "" else par for par in param
                ]  # if param is empty, set to first param in params list
                yreg = torch.stack(
                    [
                        torch.tensor(
                            bisect_right(self.yclasses[param[i]], y[i]) - 1,
                            dtype=torch.long,
                        )
                        for i in range(len(y))
                    ]
                )
                yreg = Variable(yreg)
                prediction = self.net(x)  # input x and predict based on x, [b,:,:,:]
                if len(self.params) == 1:
                    target = torch.eye(self.num_regs[0])[yreg]
                    pred = torch.nn.Sigmoid()(prediction)
                    loss = self.criterion(pred, target)  # yreg as alternative (classes)
                    # output to soft encoding (threshold output and compute) to get TP,FP...
                    output_soft = pred2soft(prediction)
                    prec = get_precision(output_soft, target)
                    """
                    par = self.params[0]
                    for i, tgt in enumerate(target):
                        output_json = {par: self.yclasses[par][prediction[i].argmax()]}
                        target_json = {par: self.yclasses[par][target[i].argmax()]}
                        # print("target "+str(target_json)+" pred "+str(output_json)+" batch "+str(bidx+1))
                    """
                else:
                    # compute losses differently for each head
                    loss = 0.0
                    pprec = []
                    param_ids = [self.dict_params[par] for par in param]
                    for pidx, par in enumerate(self.params):
                        param_indices = [
                            ppidx for ppidx, pid in enumerate(param_ids) if pid == pidx
                        ]

                        if len(param_indices) == 0:
                            continue
                        param_yreg = yreg[param_indices]
                        param_target = torch.eye(self.num_regs[pidx])[
                            param_yreg
                        ]  # one-hot encoding
                        param_prediction = prediction[pidx][param_indices]
                        param_pred = torch.nn.Sigmoid()(param_prediction)
                        param_loss = self.criterion(
                            param_pred, param_target
                        )  # one loss for each param
                        loss += param_loss  # final loss is sum of all param BCE losses

                        # output to soft encoding (threshold output and compute) to get TP,FP...
                        output_soft = pred2soft(param_prediction)
                        param_prec = get_precision(output_soft, param_target)
                        pprec.append(param_prec)
                        """
                        par = self.params[0]
                        # print json values (converted from onehot to param interval values)
                        for i, tgt in enumerate(param_target):
                            output_json = {
                                par: self.yclasses[par][param_pred[i].argmax()]
                            }
                            target_json = {
                                par: self.yclasses[par][param_target[i].argmax()]
                            }
                            # print("target "+str(target_json)+" pred "+str(output_json)+" batch "+str(bidx+1))
                        """
                    prec = np.mean(pprec)
                precs = np.append(precs, prec)
                losses = np.append(losses, loss.data.numpy())

            epoch_loss = np.nanmean(losses)
            epoch_prec = np.nanmean(precs)
            print(
                "Val Step = %d" % epoch
                + " Loss = %.4f" % epoch_loss
                + " Precision = %0.4f" % epoch_prec
            )  # +' Acc = %.4f' % epoch_acc
            return epoch_loss, epoch_prec


if __name__ == "__main__":
    parser = parse_params_cfg()
    args = parser.parse_args()
    print(args)
    print("Preparing Regressor")
    reg = Regressor(args)
    # TRAIN+VAL (depending if checkpoint exists)
    if args.resume is False:
        crops_train=int(np.round(args.num_crops*args.splits[0]))
        train_dataset = Dataset(args.trainds, args.traindsinput, crops_train, args.input_size) #set num_crops as split proportion
        train_dataset.__modify__(reg.ds_modifiers)
        train_dataset.__crop__()
        print(
            "Prepared Train Dataset "
            + "("
            + str(train_dataset.__len__())
            + ") images x modifiers x crops"
        )
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=reg.batch_size,
            shuffle=args.data_shuffle,
            num_workers=args.workers,
            pin_memory=True,
        )
        crops_val=int(np.round(args.num_crops*args.splits[1]))
        val_dataset = Dataset(args.valds, args.valdsinput, crops_val, args.input_size) #set num_crops as split proportion
        val_dataset.__modify__(reg.ds_modifiers)
        val_dataset.__crop__()
        print(
            "Prepared Validation Dataset "
            + "("
            + str(val_dataset.__len__())
            + ") images x modifiers x crops"
        )
        val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset,
            batch_size=reg.batch_size,
            shuffle=args.data_shuffle,
            num_workers=args.workers,
            pin_memory=True,
        )
        # exit()
        # Train and validate regressor
        reg.train_val(train_loader, val_loader)
    else:
        # DEPLOY
        gt_path = args.valdsinput
        image_paths = os.listdir(gt_path)
        image_files = []
        for idx, image_name in enumerate(image_paths):
            image_files.append(gt_path + "/" + image_name)  # abs_images_folder
        start_time = time.time()
        reg.deploy(image_files)
        print("--- %s seconds ---" % (time.time() - start_time))
