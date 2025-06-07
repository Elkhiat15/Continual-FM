import os
from collections import OrderedDict
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lightning import LightningModule
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.train_utils import make_optimizer
from transformers import ViTForImageClassification
import torch.nn.functional as F
from utils.train_utils import largestConnectComponent
from segmentation_models_pytorch.losses import DiceLoss
from medpy.metric import dc, jc
import imageio
from skimage.segmentation import mark_boundaries
import cv2


def init_model(args):
    if args.model == 'vit_large':
        model = ViTForImageClassification.from_pretrained('google/vit-large-patch16-224', num_labels=2, ignore_mismatched_sizes=True)
    elif args.model == 'cae_seg':
        from models.cae_seg import CAEv2_seg
        model = CAEv2_seg()
    return model


class Base_Module(LightningModule):
    def __init__(self, args):
        super(Base_Module, self).__init__()

        self.args = args

    def init_weights(self, pretrained):
        if os.path.isfile(pretrained):
            print("=> loading pretrained model {}".format(pretrained))
            pretrained_dict = torch.load(pretrained, map_location="cpu")
            pretrained_dict = pretrained_dict["state_dict"]
            model_dict = self.state_dict()
            print('Model dict: ', model_dict.keys())
            available_pretrained_dict = {}

            for k, v in pretrained_dict.items():
                print('Pretrained dict: ', k)
                if k in model_dict.keys():
                    if pretrained_dict[k].shape == model_dict[k].shape:
                        available_pretrained_dict[k] = v
                if k[7:] in model_dict.keys():
                    if pretrained_dict[k].shape == model_dict[k[7:]].shape:
                        available_pretrained_dict[k[7:]] = v

            for k, _ in available_pretrained_dict.items():
                print("loading {}".format(k))
            model_dict.update(available_pretrained_dict)
            self.load_state_dict(model_dict, strict=True)

    def load_weights(self, path):
        if os.path.isfile(path):
            print("=> Loading model from {}".format(path))
            checkpoint = torch.load(path, map_location="cpu")
            state_dict = checkpoint["state_dict"]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if "module." in k:
                    name = k[7:]  # remove `module.`
                else:
                    name = k
                print(name)
                new_state_dict[name] = v
            self.load_state_dict(new_state_dict)
            print("=> trained model loaded")

    def initialize(self, decoder, pred):
        if decoder is not None:
            for m in decoder.modules():

                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_uniform_(
                        m.weight, mode="fan_in", nonlinearity="relu"
                    )
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

                elif isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

        if pred is not None:
            for m in pred.modules():
                if isinstance(m, (nn.Linear, nn.Conv2d)):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def configure_optimizers(self):
        optim, kwargs_optimizer = make_optimizer(args=self.args)
        opt = optim(self.parameters(), **kwargs_optimizer)
        scheduler = CosineAnnealingLR(opt, T_max=self.args.epoch)

        return [opt], [scheduler]


class Segmentation_Module(Base_Module):
    def __init__(self, args):
        super(Segmentation_Module, self).__init__(args)

        self.model = init_model(self.args)

        if self.args.model == 'medsam':
            self.loss_ce = nn.BCEWithLogitsLoss()
        else:
            self.loss_ce = nn.CrossEntropyLoss()
        if not self.args.model == 'medsam':
            self.loss_dc = DiceLoss(mode='multiclass')

    def forward(self, x):
        if self.args.model == 'medsam':
            return self.model(x, multimask_output=False)
        return self.model(x)

    def training_step(self, batch, batch_idx):
        input, target = batch

        output = self(input)

        loss = self.loss_ce(output, target)

        self.log("Train/CE_Loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    # def on_train_batch_end(self, outputs, batch, batch_idx, unused=None):
    #     for name, param in self.named_parameters():
    #         if param.grad is None:
    #             print(name)

    def validation_step(self, batch, batch_idx):
        input, target, _ = batch

        output = self(input)

        dsc = 0
        jac = 0
        for j in range(output.size(0)):
            output_temp = torch.argmax(output[j], dim=0).cpu().detach().numpy()
            output_temp = largestConnectComponent(output_temp)
            target_temp = target[j].cpu().detach().numpy()
            dsc += dc(output_temp, target_temp)
            jac += jc(output_temp, target_temp)

        self.log("Val/Dice", dsc / output.size(0), on_epoch=True, sync_dist=True, on_step=False, prog_bar=True)
        self.log("Val/Jac", jac / output.size(0), on_epoch=True, sync_dist=True, on_step=False, prog_bar=True)

        return dsc

    def on_test_start(self):
        self.names = []
        self.dscs = []
        self.jacs = []

    def test_step(self, batch, batch_idx):
        images, targets_subset, names_subset = batch
        output = self(images)
        if self.args.model == 'medsam':
            output = output.pred_masks
            output = output.sigmoid()
            output = output.squeeze(1)
            output = F.interpolate(output, size=(1024, 1024), mode='bilinear', align_corners=False)
            output = output.squeeze()

        dirname = "{}results_{}_{}".format(self.args.save_name, self.args.dataset, self.args.seed)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        for idx, name in enumerate(names_subset):
            image_save = images[idx].cpu().detach().numpy()
            image_save = image_save * 0.5 + 0.5
            image_save = np.transpose(image_save, (1, 2, 0))
            if self.args.model == 'medsam':
                output_np = output[idx].cpu().detach().numpy()
                output_np = np.round(output_np)
            else:
                output_np = torch.argmax(output[idx], dim=0).cpu().detach().numpy()
            binary_output = np.array(output_np)
            binary_output = largestConnectComponent(binary_output)
            binary_output = cv2.resize(binary_output, (224, 224), interpolation=cv2.INTER_NEAREST)
            target_np = targets_subset[idx].cpu().detach().numpy().astype(np.uint8)
            target_np = cv2.resize(target_np, (224, 224), interpolation=cv2.INTER_NEAREST)
            if np.amax(target_np) == 0:
                # change the value of center pixel of target_np to 1
                target_np[target_np.shape[0] // 2, target_np.shape[1] // 2] = 1
            if self.args.save_results:
                filename = os.path.join(dirname, str(name) + ".png")
                output_save = mark_boundaries(image_save, binary_output, (0, 1, 1))
                output_save = mark_boundaries(output_save, target_np, (1, 0, 0))
                output_save = (output_save * 255).astype(np.uint8)
                imageio.imwrite(filename, output_save)

            dsc = dc(binary_output, target_np)
            jac = jc(binary_output, target_np)

            self.names.append(name)
            self.dscs.append(dsc)
            self.jacs.append(jac)

            self.log("Test/Dice", dsc)
            self.log("Test/Jac", jac)

    def on_test_end(self):
        print('=> Writing results to local file')
        dataframe = pd.DataFrame(
            {
                "name": self.names,
                "dice": self.dscs,
                "jac": self.jacs,
            }
        )
        dataframe.to_excel(
            os.path.join(
                self.args.save_name,
                "count_results_{}_{}_{}.xlsx".format(
                    self.args.dataset, self.args.fold, self.args.seed
                ),
            )
        )
        print('=> Dataframe saved')
        results = {
            "Dsc": np.average(self.dscs),
            "Jac": np.average(self.jacs),
        }
        results_json = json.dumps(results, indent=4)
        with open(os.path.join(self.args.save_name, 'results_{}_{}_{}.json'.format(self.args.dataset, self.args.fold, self.args.seed)), 'w') as f:
            f.write(results_json)
        print('=> Json result saved')

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.args.epoch)

        return [opt], [scheduler]


def load_model(args):

    return Segmentation_Module(args=args)
