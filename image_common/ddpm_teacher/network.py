import torch
import torch.nn as nn
from .module import DownSample, ResBlock, Swish, TimeEmbedding, UpSample
from torch.nn import init


class UNet(nn.Module):
    def __init__(self, T=1000, image_resolution=64, ch=128, ch_mult=[1,2,2,2], attn=[1], num_res_blocks=4, dropout=0.1, use_cfg=False, cfg_dropout=0.1, num_classes=None):
        super().__init__()
        self.image_resolution = image_resolution
        assert all([i < len(ch_mult) for i in attn]), 'attn index out of bound'
        tdim = ch * 4
        # self.time_embedding = TimeEmbedding(T, ch, tdim)
        self.time_embedding = TimeEmbedding(tdim)

        # classifier-free guidance
        self.use_cfg = use_cfg
        self.cfg_dropout = cfg_dropout
        if use_cfg:
            assert num_classes is not None
            cdim = tdim
            # num_classes+1 because 0 is reserved for the null (unconditional) token
            self.class_embedding = nn.Embedding(num_classes+1, cdim)

        self.head = nn.Conv2d(3, ch, kernel_size=3, stride=1, padding=1)
        self.downblocks = nn.ModuleList()
        chs = [ch]  # record output channel when dowmsample for upsample
        now_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(ResBlock(
                    in_ch=now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)

        self.middleblocks = nn.ModuleList([
            ResBlock(now_ch, now_ch, tdim, dropout, attn=True),
            ResBlock(now_ch, now_ch, tdim, dropout, attn=False),
        ])

        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(ResBlock(
                    in_ch=chs.pop() + now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample(now_ch))
        assert len(chs) == 0

        self.tail = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            Swish(),
            nn.Conv2d(now_ch, 3, 3, stride=1, padding=1)
        )
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.head.weight)
        init.zeros_(self.head.bias)
        init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
        init.zeros_(self.tail[-1].bias)

    def forward(self, x, timestep, class_label=None):
        # Timestep embedding
        temb = self.time_embedding(timestep)

        if self.use_cfg and class_label is not None:
            if self.training:
                assert not torch.any(class_label == 0) # 0 for null.
                
                # Create a mask for dropping labels.
                # We drop with probability self.cfg_dropout.
                # So, we *keep* with probability (1 - self.cfg_dropout).
                mask = torch.rand_like(class_label, dtype=torch.float) > self.cfg_dropout
                
                # Create a null label tensor (all zeros)
                null_label = torch.tensor(0, device=class_label.device, dtype=class_label.dtype)
                
                # Use the mask to select between the original label and the null label
                # If mask is True (1), keep class_label.
                # If mask is False (0), use null_label (0).
                class_label = torch.where(mask, class_label, null_label)            
            
            # Get the embedding for the class label (which could be the null token)
            class_emb = self.class_embedding(class_label) # Shape: [B, cdim]
            
            # Add the class embedding to the time embedding
            temb = temb + class_emb # Shape: [B, tdim]

        # Downsampling
        h = self.head(x)
        hs = [h]
        for layer in self.downblocks:
            h = layer(h, temb)
            hs.append(h)
        # Middle
        for layer in self.middleblocks:
            h = layer(h, temb)
        # Upsampling
        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
            h = layer(h, temb)
        h = self.tail(h)

        assert len(hs) == 0
        return h
