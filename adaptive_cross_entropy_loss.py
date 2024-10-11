"""
Author: Ankit Gupta
"""

from itertools import accumulate

import torch
from torch import nn
from torch.nn import functional as F


class AdaptiveCrossEntropyLoss(nn.Module):
    def __init__(
        self,
        dim: int,                    # embedding dim
        vocab_sizes,                 # [V0, ... Vc-1]
        ignore_index=-100,
        weight=None,
    ):
        """a simpler case of nn.AdaptiveLogSoftmaxWithLoss. A vocab of size V is assumed to be composed of individual vocabs
           of sizes V0,..,Vc-1. Given a vector, a router is used to determine the approprate Vi and within Vi the appropriate
           token is determined:  Pr(w in Vi | x) = Pr(Vi | x) * Pr(w within Vi | x).
           
           Depending on the cluster sizes and distribution of targets, this can be significantly faster than vanilla 
           nn.Linear(dim, sum(vocab_sizes) + F.cross_entropy.
        """
        super().__init__()
        assert all(int(c) == c and c > 0 for c in vocab_sizes), "vocab_sizes should be sequence of positive ints"
        assert ignore_index is None or ignore_index < 0
        
        self.dim = dim
        self.vocab_sizes = vocab_sizes
        self.ignore_index = ignore_index
        self.router = nn.Linear(dim, len(vocab_sizes), bias=False)
        self.weight = nn.Linear(dim, sum(vocab_sizes), bias=False).weight if weight is None else weight
        assert self.weight.shape == (sum(vocab_sizes), dim)
        
    def vocabs(self, i, x):
        cutoffs = [0] + list(accumulate(self.vocab_sizes))  # [0  V0  ...  V0+..Vc-1]
        low_idx = cutoffs[i]             # V0+..Vi-1
        high_idx = cutoffs[i + 1]        # V0+..Vi
        return F.linear(x, self.weight[low_idx:high_idx])           
    
    def forward(self, input, target):
        """input:  [..., self.dim], target: [...]
           output: (), (), [...], [...]
        """
        assert input.ndim >= 2 and target.shape == input.shape[:-1] and input.shape[-1] == self.dim
        in_shape = target.shape
        
        input = input.view(-1, self.dim)    # [b d]
        target = target.view(-1)            # [b]
        
        vocab_sizes = self.vocab_sizes                 # [V0 ... Vc-1]
        cutoffs = [0] + list(accumulate(vocab_sizes))  # [0  V0  ...  V0+..Vc-1]
        
        used_rows = target.new_zeros(1)
        batch_size = target.size(0)
        
        output_dtype = torch.float32 if torch.is_autocast_enabled() else None
        output = input.new_zeros(batch_size, dtype=output_dtype)  # [b]  router log prob + intra vocab log prob
        
        ignore_index = self.ignore_index
        closest_in_target_cluster = input.new_zeros(batch_size, dtype=torch.long)  # [b]
        if ignore_index is not None:
            closest_in_target_cluster += ignore_index
        
        # intra target vocab log prob
        for i in range(len(cutoffs) - 1):
            
            low_idx = cutoffs[i]             # V0+..Vi-1
            high_idx = cutoffs[i + 1]        # V0+..Vi

            target_mask = (low_idx <= target) & (target < high_idx)  # [b] : is i'th cluster
            row_indices = target_mask.nonzero().squeeze()            # rows inds for i'th cluster

            if row_indices.numel() == 0:
                continue
            
            input_subset = input.index_select(0, row_indices)       # [bi d] subset of inputs for vocab i
            vocab_output = self.vocabs(i, input_subset)             # [bi Vi] 
            
            relative_target = target[target_mask] - low_idx         # [bi] intra vocab index
            # intra vocab loss
            assert vocab_output.ndim == relative_target.ndim + 1
            vocab_nll = F.cross_entropy(vocab_output, relative_target, reduction='none') # [bi] 
            
            # vocab id loss
            router_output = self.router(input_subset)                         # [bi  num_vocabs]
            router_nll = F.cross_entropy(router_output, i + 0*relative_target, reduction='none')  # [bi] 
            vocab_nll = vocab_nll.to(output.dtype) + router_nll.to(output.dtype)  # [bi]
            
            output.index_copy_(0, row_indices, vocab_nll)                     # [b]
            
            # pred within target cluster
            closest_in_cluster_i = vocab_output.argmax(-1) + low_idx                     # [bi]
            closest_in_target_cluster.index_copy_(0, row_indices, closest_in_cluster_i)  # [b]
            
            used_rows += row_indices.numel()
            
            if used_rows == batch_size:
                break
        
        used_rows_non_ignore = used_rows
        
        # handle ignore_index
        if ignore_index is not None and used_rows < batch_size:
            target_mask = target == ignore_index                    # [b]
            row_indices = target_mask.nonzero().squeeze()           # rows inds with ignore_index
            used_rows += row_indices.numel()
        
        if used_rows < batch_size:
            raise RuntimeError("Target values should be in [0 .. {}] or {}"
                               "but values in range [{}, {}] "
                               "were found. ".format(cutoffs[-1]-1,
                                                     ignore_index,
                                                     target.min().item(),
                                                     target.max().item()))

        loss = output.sum() / used_rows_non_ignore.clip(min=1)
        
        output = output.view(in_shape)
        closest_in_target_cluster = closest_in_target_cluster.view(in_shape)
        
        # print(loss, output.sum())
        # exit()
        
        return loss, used_rows_non_ignore, output, closest_in_target_cluster   # (), (), [...], [...]
        
    def full_vocab_log_prob(self, input):
        """input: [..., d]   out: [..., V0+..Vc-1]
        """
        cutoffs = [0] + list(accumulate(self.vocab_sizes))        # [0, V0, ... V0+..Vc-1]
        
        out = input.new_zeros(input.shape[:-1] + (cutoffs[-1],))  # [... V0+..Vc-1]
        router_logprob = self.router(input).log_softmax(-1)       # [... num_vocabs]
        
        # intra vocab log prob
        for i in range(len(cutoffs) - 1):
            
            low_idx = cutoffs[i]             # V0+..Vi-1
            high_idx = cutoffs[i + 1]        # V0+..Vi
        
            vocab_logprob = self.vocabs(i, input).log_softmax(-1)     # [... Vi]
            out_logprob = vocab_logprob + router_logprob[..., i:i+1]  # [... Vi]
            out[..., low_idx:high_idx] = out_logprob                  # [... V0+..Vc-1]
        
        return out
    
    @torch.no_grad()
    def predict_with_full_vocab_log_prob(self, input):
        """input: [... d]   output: [...], [... V0+..Vc-1]
        """
        log_prob = self.full_vocab_log_prob(input)                 # [... , V0+...Vc-1]
        return log_prob.argmax(dim=-1), log_prob                   # [...], [... V0+...Vc-1]
    
    @torch.no_grad()
    def predict_approximately(self, input):
        """ Instead of making predictions using true distribution over full vocab,
            here we first predict the cluster based only on router scores and then make the
            prediction within this cluster. This is inaccurate but fast if predicted clusters have small size.
            
            input: [... d]   out: [...]
        """
        assert input.ndim >= 2 and input.shape[-1] == self.dim  # [... d]
        out_shape = input.shape[:-1]                   # [...]
        
        input = input.view(-1, self.dim)               # [b d]
        
        vocab_sizes = self.vocab_sizes                 # [V0 ... Vc-1]
        cutoffs = [0] + list(accumulate(vocab_sizes))  # [0  V0  ...  V0+..Vc-1]
        
        used_rows = 0
        batch_size = input.size(0)
        vocab_inds = self.router(input).argmax(-1)      # [b]
        output = vocab_inds.new_empty(vocab_inds.shape) # [b]
        
        # intra vocab pred
        for i in range(len(cutoffs) - 1):
            
            low_idx = cutoffs[i]             # V0+..Vi-1
            high_idx = cutoffs[i + 1]        # V0+..Vi

            target_mask = vocab_inds == i    # [b] : is i'th cluster
            row_indices = target_mask.nonzero().squeeze()           # rows inds for i'th cluster

            if row_indices.numel() == 0:
                continue
            
            vocab_inds.index_fill_(0, row_indices, i)               # [b] : vocab id
            
            input_subset = input.index_select(0, row_indices)       # [bi d] subset of inputs for vocab i
            vocab_pred = self.vocabs(i, input_subset).argmax(-1)    # [bi]  intra vocab index
            
            output.index_copy_(0, row_indices, low_idx + vocab_pred) # [b]

            used_rows += row_indices.numel()
            
            if used_rows == batch_size:
                break

        return output.view(out_shape)   # [...]
    
#     @torch.no_grad()
#     def predict(self, input):
#         """input: [... d]   out: [...]
#         A bit faster than self.predict_with_full_vocab_log_prob
#         """        
#         cutoffs = [0] + list(accumulate(self.vocab_sizes))         # [0, V0, ... V0+..Vc-1]
        
#         cluster_pred_score = self.router(input)                    # [... num_vocabs]
#         cluster_pred = []                                          # [... 1]*num_vocabs
        
#         for i in range(len(cutoffs) - 1):
#             low_idx = cutoffs[i]             # V0+..Vi-1
            
#             vocab_logits = self.vocabs[i](input)                   # [... Vi]
#             vocab_max = vocab_logits.max(-1)                       # [...]
#             cluster_pred.append(low_idx + vocab_max.indices.unsqueeze(-1))
            
#             vocab_logsumexp = vocab_logits.logsumexp(-1)           # [...]
#             cluster_pred_score[..., i] += vocab_max.values - vocab_logsumexp # [...]
        
#         cluster_pred = torch.cat(cluster_pred, dim=-1)             # [... num_vocabs]
#         best_cluster = cluster_pred_score.argmax(-1, keepdim=True) # [... 1]
        
#         return cluster_pred.gather(-1, best_cluster).squeeze(-1)   #[...]
    
    
    

def main():
    d = 4
    vocab_sizes = [8,2]
    embeds = AdaptiveCrossEntropyLoss(d, vocab_sizes, ignore_index=-100)

    logits = torch.randn(10, d)
    labels = torch.randint(sum(vocab_sizes), size=logits.shape[:-1])
    labels[1] = -100

    print(labels, embeds(logits, labels))
    
if __name__ == '__main__':
    main()