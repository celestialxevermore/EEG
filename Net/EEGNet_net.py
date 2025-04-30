import torch
import torch.nn as nn
from layers import LinearWithConstraint, Conv2dWithConstraint

class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() 



class EEGNet(nn.Module):
    def __init__(self, args, shape):
        super(EEGNet, self).__init__()
        self.args = args
        
        self.use_mutual_learning = args.use_mutual_learning #if args.n_bands > 1 else False
        self.use_multi_source_align = args.use_multi_source_align
        self.n_bands = shape[1]
        self.num_ch = shape[2]
        self.T = shape[3]

        self.F1 = 16
        self.F2 = 32
        self.D = 2
        self.P1 = 4
        self.P2 = 8
        self.t1 = 16

        self.temporal_conv = nn.Sequential(
            nn.Conv2d(self.n_bands, self.F1, kernel_size=(1, self.T // 2), bias=False, padding='same'),
            nn.BatchNorm2d(self.F1)
        )
        self.spatial_conv = nn.Sequential(
            Conv2dWithConstraint(self.F1, self.F1 * self.D, kernel_size=(self.num_ch, 1), groups=self.F1, bias=False),
            nn.BatchNorm2d(self.F1 * self.D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, self.P1))
        )
        self.separable_conv = nn.Sequential(
            nn.Conv2d(self.F1 * self.D, self.F2, kernel_size=(1, self.t1), groups=self.F1 * self.D, bias=False),
            nn.Conv2d(self.F2, self.F2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(self.F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, self.P2))
        )
        self.linear = nn.Sequential(
            nn.Flatten(),
            LinearWithConstraint(self.F2 * 33, out_features=4, max_norm=0.25)
        )

        if self.use_mutual_learning:
            freq_feature_dim = (self.T // 2 + 1) * self.n_bands
            self.freq_extractor = nn.Sequential(
                nn.Linear(freq_feature_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU()
            )
            self.freq_classifier = nn.Linear(64, 4)
        else:
            self.freq_extractor = None
            self.freq_classifier = None

        if self.use_multi_source_align:
            self.domain_discriminator = nn.Sequential(
                nn.Linear(self.F2 * 33, 100),
                nn.ReLU(),
                nn.Linear(100, 9)
            )
        else:
            self.domain_discriminator = None 

    def forward(self, x):
        result = {}

        # --- Time series branch ---
        out = self.temporal_conv(x)
        out = self.spatial_conv(out)
        out = self.separable_conv(out)
        cls_output = self.linear(out)
        result["cls"] = cls_output

        if self.use_mutual_learning and self.freq_extractor is not None:
            freq_input = torch.fft.rfft(x, dim=-1)
            freq_input = freq_input.abs()
            freq_input = freq_input.mean(dim=2)
            freq_input = freq_input.flatten(start_dim=1)
            freq_feat = self.freq_extractor(freq_input)
            freq_pred = self.freq_classifier(freq_feat)
            result["freq"] = freq_pred

        if self.use_multi_source_align and self.domain_discriminator is not None:
            domain_input = GradientReversalLayer.apply(out.flatten(start_dim=1))
            domain_pred = self.domain_discriminator(domain_input)
            result["domain"] = domain_pred

        return result