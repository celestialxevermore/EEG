import torch
import torch.nn as nn

from Net import *

def build_net(args, shape):
    print("[Build Net]")

    # Build model with band count
    net = EEGNet_net.EEGNet(args, shape)

    # 🧠 Load pretrained parameters only if use_pretrained is True
    if args.use_pretrained:
        if args.pretrained_path is None:
            raise ValueError("use_pretrained을 했는데 pretrained_path가 None입니다.")

        print(f"Loading pretrained weights from {args.pretrained_path}")
        param = torch.load(args.pretrained_path)

        model_dict = net.state_dict()
        pretrained_dict = {k: v for k, v in param['net_state_dict'].items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_dict)
        net.load_state_dict(model_dict)

    # Device config
    if args.gpu != 'cpu':
        assert torch.cuda.is_available(), "Check GPU"
        if args.gpu == "multi":
            net = nn.DataParallel(net)
        else:
            device = torch.device(f'cuda:{args.gpu}')
            torch.cuda.set_device(device)
        net = net.cuda()
        device = net.device if hasattr(net, 'device') else device
    else:
        device = torch.device("cpu")

    print(f"device: {device}\n")
    return net
