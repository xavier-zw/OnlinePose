import argparse
import mmcv
import numpy as np
import torch
from mmcv.runner import load_checkpoint
from PIL import Image
from models import build_posenet
from inference import get_max_preds
from torchvision import transforms
from torch import nn


class Pose(nn.Module):
    def __init__(self):
        super(Pose, self).__init__()
        self._init_model()

    def parse_args(self):
        parser = argparse.ArgumentParser(
            description='Convert models to ONNX')
        parser.add_argument('--config',
                            default="./config/litehrnet_18_coco_256x192.py",
                            help='config file path')
        parser.add_argument('--checkpoint',
                            default="./weight/litehrnet_18_coco_256x192.pth",
                            help='checkpoint file')
        args = parser.parse_args()
        return args

    def _init_model(self):
        args = self.parse_args()
        cfg = mmcv.Config.fromfile(args.config)
        # build the model
        self.model = build_posenet(cfg.model)
        if hasattr(self.model, 'forward_dummy'):
            self.model.forward = self.model.forward_dummy
        else:
            raise NotImplementedError(
                'Please implement the forward method for exporting.')
        load_checkpoint(self.model, args.checkpoint, map_location='cpu')
        self.model.eval()

    def forward(self, imgs):
        pose = []
        for img in imgs:
            img = Image.fromarray(img)
            transform = transforms.Compose([
                transforms.Resize((256, 192)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            with torch.no_grad():
                out = self.model(transform(img).unsqueeze(dim=0))
                h, w = out.shape[-2], out.shape[-1]
                out = get_max_preds(out.numpy())[0] / (np.array([w, h]))
            pose.append(out)
        return np.concatenate(pose, axis=0)