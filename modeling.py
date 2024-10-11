"""
Author: Ankit Gupta
"""

import math, random
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

import logging
from functools import partial
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint_sequential

import torchaudio

from einops import rearrange, repeat
import opt_einsum as oe
einsum = contract = oe.contract

from local_utils import dotdict

from adaptive_cross_entropy_loss import AdaptiveCrossEntropyLoss


@torch.no_grad()
@torch.cuda.amp.autocast(enabled=False)
def speed_aug(wave, orig_freq, factor):
    source_sample_rate, target_sample_rate = torchaudio.transforms._transforms._source_target_sample_rate(
            orig_freq, factor)
    return torchaudio.functional.resample(wave, source_sample_rate, target_sample_rate)


class LogMels(nn.Module):
    def __init__(self, sample_rate=16000, n_fft=400, win_length=400, hop_length=160, 
                 n_mels=80, num_frames_to_stack=4):
        super().__init__() 
        self.sample_rate = sample_rate
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels)
        self.n_mels = n_mels
        self.num_frames_to_stack = num_frames_to_stack
        self.prob_speed_aug = .0
        self.prob_noise, self.snr = .0, 10
        
    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, wave):
        B, T = wave.shape
        
        if self.training and random.random() < self.prob_speed_aug:
            wave = speed_aug(wave, self.sample_rate, 1 + random.random()) # [1,1.1]
            wave = F.pad(wave, (0,T-wave.shape[-1]))
        
        if self.training and random.random() < self.prob_noise:
            # not sure how useful this is as if there's silence and 
            # we noise it then model might learn to delete stuff
            # TODO: maybe we should first detect silence and only noise non-silence parts
            snr = torch.ones_like(wave[:,0]) * self.snr
            wave = torchaudio.functional.add_noise(wave, wave.roll(-1, dims=0), snr)
        
        l = T // self.melspec.hop_length
        spec = self.melspec(wave)[...,:l]                      # [B n_mels t]
        spec = spec.transpose(-1,-2).clip_(min=1e-10).log10_() # [B t n_mels]
        mx = spec.amax(dim=(-2,-1), keepdim=True)              # [B 1 1]
        spec = torch.maximum(spec, mx - 8.0).div_(4).add_(1)
        return spec.view(spec.shape[0], -1, self.num_frames_to_stack * spec.shape[-1])  # [B t/4 n_mels*4]


class FF(nn.Module):
    def __init__(self, 
                 dim=None,               # required
                 dim_ff=None,            # default 4*dim
                 activation=F.gelu, 
                 **kwargs):
        super().__init__()
        
        self.dim = dim
        self.dim_ff = dim_ff if dim_ff else dim * 4

        self.wi = nn.Parameter(self.init(torch.empty(1, dim, self.dim_ff)))
        self.wo = nn.Parameter(self.init(torch.empty(1, self.dim_ff, dim)))
        
        self.act = activation
        
    def init(self, t, scale=1):
        std = (t.shape[-2] * scale) ** -.5
        return t.uniform_(-std, std)
    
    def forward(self, x, implementation=None):
        """x : [... d]"""
        return self.act(x.matmul(self.wi.squeeze(0))).matmul(self.wo.squeeze(0))
        

class SinEmbed(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(-1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, pos):
        """pos [seq_len]"""
        return self.pe[pos]


    
class Attn(nn.Module):
    def __init__(self, i_layer, d_model, H, d_head=64, bidirec_prefix_len=0, **kernel_args):
        super().__init__()
        assert H % d_head == 0
        self.d_model, self.d_head = d_model, d_head
        self.QKV = nn.Linear(d_model, 3*H, bias=False)
        self.O = nn.Linear(H, d_model, bias=False)
        self.bidirec_prefix_len = bidirec_prefix_len
        self.i_layer = i_layer
        
    def forward(self, u, cache=None, **kwargs):
        """u: (B L d_model)   returns: same shape as u
        """
        assert u.shape[-1] == self.d_model
        B, L, _  = u.shape
        bidirec_prefix_len = self.bidirec_prefix_len
        
        form_heads = lambda x: x.view(B, L, -1, self.d_head).transpose(-3,-2)
        
        Q, K, V = map(form_heads, self.QKV(u).chunk(3,-1))          # [B n L dh]
        
        if cache is not None and self.i_layer in cache:  # inference
            assert L == 1
            pre_K, pre_V = cache[self.i_layer]
            K, V = torch.cat((pre_K, K), dim=-2), torch.cat((pre_V, V), dim=-2) # [B n pre+1 dh]
            O = F.scaled_dot_product_attention(Q, K, V)                 # [B n L dh]
        
        elif bidirec_prefix_len <= 0:
            O = F.scaled_dot_product_attention(Q, K, V, is_causal=True) # [B n L dh]
        else:
            assert bidirec_prefix_len <= L             
            O_pfx = F.scaled_dot_product_attention(
                Q[...,:bidirec_prefix_len,:], 
                K[...,:bidirec_prefix_len,:], 
                V[...,:bidirec_prefix_len,:], is_causal=False)      # [B n L dh]
            
            # upper right causal mask
            i, j = (torch.arange(s, device=Q.device) for s in [L-bidirec_prefix_len, L])
            mask = (i + len(j) - len(i)).view(-1,1) >= j.view(1,-1)            
            O_sfx = F.scaled_dot_product_attention(Q[...,bidirec_prefix_len:,:], K, V, 
                                                   attn_mask=mask, is_causal=False) # [B n L-pfx_len dh]
            O = torch.cat((O_pfx, O_sfx), dim=-2)                   # [B n L dh]
            assert O.shape == Q.shape
        
        y = O.transpose(-3,-2).reshape(B, L, -1)                       # [B L H]
        
        if cache is not None:
            cache[self.i_layer] = (K, V)
                                        
        return self.O(y)                                            # [B L d_model]


class Block(nn.Module):
    def __init__(self, config, i_layer):
        super().__init__()
        self.config = config
        self.context = Attn(i_layer, config.d_model, config.H, d_head=64, bidirec_prefix_len=config.bidirec_prefix_len)
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.ff = FF(dim=config.d_model)
        self.ln_2 = nn.LayerNorm(config.d_model)
        
    def forward(self, x, **kwargs):
        x = x + self.context(self.ln_1(x), **kwargs)
        return x + self.ff(self.ln_2(x))


class ModelConfig:
    vocab_size = 2**15
    d_emb = 128
    d_model = 256
    H = 256
    N = 512
    n_layers = 16
    gradient_checkpointing = False
    bidirec_prefix_len = 0
    

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.emb = nn.Embedding(config.vocab_size, config.d_emb)
        self.emb_to_model = nn.Linear(config.d_emb, config.d_model, bias=False)
        self.pos = SinEmbed(2048, config.d_model)
        self.stack = nn.Sequential(*[Block(config, i) for i in range(config.n_layers)])
        self.logmel = LogMels(num_frames_to_stack=config.num_frames_to_stack)
        d_mel = self.logmel.num_frames_to_stack * self.logmel.n_mels
        self.mel_to_model = nn.Linear(d_mel, config.d_model, bias=False)
        self.ln = nn.LayerNorm(config.d_model)
        self.gradient_checkpointing = config.gradient_checkpointing
        self.loss_fn = AdaptiveCrossEntropyLoss(config.d_emb, [4096, 8192, config.vocab_size-4096-8192], 
                                                weight=self.emb.weight)
        
    def forward(self, wave=None, inputs=None, targets=None, **kwargs):
        device = wave.device
        B, L = inputs.shape
        assert targets.shape == inputs.shape
        
        logmel = self.logmel(wave)                      # [B La n_mels*4]
        
        x_audio = self.mel_to_model(logmel)             # [B La d]
        x_text = self.emb_to_model(self.emb(inputs))    # [B L d]
        assert self.config.bidirec_prefix_len in [0, x_audio.shape[-2]], f'{x_audio.shape[-2]}'
        
        x = torch.cat((x_audio, x_text), dim=-2)        # [B La+L d]
        x = x + self.pos(torch.arange(x.shape[-2], device=x.device))
        
        if self.gradient_checkpointing and self.training:
            x = checkpoint_sequential(self.stack, len(self.stack), x, use_reentrant=False)
        else:
            x = self.stack(x)
        
        xt = self.ln(x[:,-L:]).matmul(self.emb_to_model.weight) # [B L e]    
        xt = xt * xt.size(-1)**-.5 
        loss, *_ = self.loss_fn(xt, targets)
        preds = self.loss_fn.predict_approximately(xt)   # [B L]
        
        is_not_ignore = targets != -100                 
        is_err = (preds != targets) & is_not_ignore     
        err = is_err.float().sum() / is_not_ignore.float().sum().clip(min=1e-3)
        
        return loss, err, preds
    
    
    @torch.no_grad()
    def inference(self, wave=None, inputs=None, max_text_len=-1, **kwargs):
        assert max_text_len > 1
        device = wave.device
        x_audio = self.mel_to_model(self.logmel(wave))       # [B La d]
        B, La, d = x_audio.shape
        _B, Lt = inputs.shape
        assert _B == B
        assert self.config.bidirec_prefix_len in [0, x_audio.shape[-2]], f'{x_audio.shape[-2]}'
        
        pos = self.pos(torch.arange(La + max_text_len+1, device=device)) # [_ d]
        x_text = self.emb_to_model(self.emb(inputs))         # [B Lt d]
        x = torch.cat((x_audio, x_text), dim=-2)             # [B La+Lt d]
        x += pos[:La+Lt]
        
        preds, cache = [], {}
        for t in range(La+Lt, La+max_text_len+1):
            for i_layer, layer in enumerate(self.stack):
                x = layer(x, cache=cache)
                assert cache[i_layer][0].shape[-2] == t
            xt = self.ln(x[:,-1:]).matmul(self.emb_to_model.weight) # [B 1 e]    
            xt = xt * xt.size(-1)**-.5 
            pred = self.loss_fn.predict_approximately(xt)    # [B 1]
            preds.append(pred)    
            # pred at timestep t becomes next input
            x = self.emb_to_model(self.emb(pred))            # [B 1 d]
            x += pos[t:t+1]                                  # [1 d]
        
        preds = torch.cat(preds, -1)                         # [B max_text_len-Lt+1]
        return preds

