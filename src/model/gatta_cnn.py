import torch.nn as nn
from attention.global_query import compute_global_query
from attention.agreement import agreement_conv, agreement_dense
from modulation.gain_control import GainControl

class GAttACNN(nn.Module):
    def __init__(self, backbone, conv_layers, dense_layers, d):
        super().__init__()
        self.backbone = backbone

        self.conv_kq = nn.ModuleDict({
            name: KQProjectionConv(ch, d)
            for name, ch in conv_layers.items()
        })

        self.dense_kq = nn.ModuleDict({
            name: KQProjectionLinear(ch, d)
            for name, ch in dense_layers.items()
        })

        self.gain = nn.ModuleDict({
            name: GainControl()
            for name in list(conv_layers.keys()) + list(dense_layers.keys())
        })

    def forward(self, x):
        logits, feats = self.backbone(x)

        conv_q, dense_q = [], []
        conv_k, dense_k = {}, {}

        for name, f in feats.items():
            if f.dim() == 4:
                k, q = self.conv_kq[name](f)
                conv_k[name] = k
                conv_q.append(q)
            else:
                k, q = self.dense_kq[name](f)
                dense_k[name] = k
                dense_q.append(q)

        q_global = compute_global_query(conv_q, dense_q)

        for name in conv_k:
            gatta = agreement_conv(conv_k[name], q_global)
            feats[name] = self.gain[name](feats[name], gatta)

        for name in dense_k:
            gatta = agreement_dense(dense_k[name], q_global)
            feats[name] = self.gain[name](feats[name], gatta)

        return logits
