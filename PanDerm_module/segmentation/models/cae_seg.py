import torch
import torch.nn as nn
from mmengine.config import Config
from mmseg.models import build_segmentor
from mmseg.models.utils import resize


class CAEv2_seg(nn.Module):
    def __init__(self):
        super(CAEv2_seg, self).__init__()

        config = Config.fromfile('/data/wangzh/code/skinfm/models/cae_config.py')
        self.segmentor = build_segmentor(config.model)
        self.segmentor.init_weights()

        print('=> Loading CAE weights from /data2/wangzh/checkpoints/caev2_large_checkpoint.pth')
        cae_weight = torch.load('/data2/wangzh/checkpoints/caev2_large_checkpoint.pth', map_location='cpu')['model']
        new_state_dict = {k.replace('encoder.', ''): v for k, v in cae_weight.items() if 'encoder' in k}

        model_dict = self.segmentor.backbone.state_dict()
        matched_dict = {}

        for k, v in new_state_dict.items():
            if k in model_dict and model_dict[k].size() == v.size():
                matched_dict[k] = v
            else:
                print(f'Skipping {k} due to size mismatch: {v.size()} vs {model_dict[k].size()}')

        model_dict.update(matched_dict)
        self.segmentor.backbone.load_state_dict(model_dict, strict=False)
        print('=> CAE weights loaded')

    def forward(self, x):

        output = self.segmentor._forward(x)
        output = resize(
            input=output,
            size=x.shape[2:],
            mode='bilinear',
            align_corners=False,
            warning=False
        )

        return output


if __name__ == '__main__':
    model = CAEv2_seg()
    model.eval()
    input_tensor = torch.randn(1, 3, 224, 224)
    output = model(input_tensor)
    print(output.shape)
