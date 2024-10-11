"""
# full test sets
python test.py --per_device_batch_size=8 --limit_test_batches=1.0 --pr_prefix_dataset=0.0 --load_path=outputs/all-128-2e-4-no-mls_people-part2/checkpoint/epoch=6-step=1000000.ckpt --output_dir=./outputs/all-128-2e-4-no-mls_people-part2 --split=test

# python test.py --per_device_batch_size=8 --pr_prefix_dataset=0.0 --load_path=outputs/all-128-2e-4-part2/checkpoint/epoch=3-step=1000000.ckpt --only_ds=clean

# partial train sets
python test.py --per_device_batch_size=8 --limit_test_batches=3000 --pr_prefix_dataset=0.0 --load_path=outputs/all-128-2e-4-no-mls_people-part2/checkpoint/epoch=6-step=1000000.ckpt --output_dir=./outputs/all-128-2e-4-no-mls_people-part2 --split=train
"""

import os, sys, time, json, glob, shutil, socket, dac
from tqdm.auto import tqdm
from termcolor import colored
from copy import deepcopy
import numpy as np
import pandas as pd

from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer, seed_everything
from lightning.pytorch.utilities.combined_loader import CombinedLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from transformers import AutoTokenizer

from local_utils import read_file, dotdict, ram_GB, free_ram, save_df_to_tsv, read_file, write_file, dotdict, group_parameters_for_optimizer, get_cosine_schedule_with_warmup, override_globals_from_cl

from modeling import Transformer
from wave_dataloader import SpeechDataset, AudioDataset, AudioConcatDataset
DATASETS = SpeechDataset.DATASETS


from whisper.normalizers import EnglishTextNormalizer

torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

import warnings
warnings.filterwarnings("ignore")

# disable grad computation
torch.set_grad_enabled(False)
print('grad computation is disabled')


# -----------------------------------------------------------------------------
# I/O
output_dir = './outputs/temp'
load_path = ''                    # init weights 
seed = 42
# data
per_device_batch_size = 128  # should be manually set to largest micro-batch size allowed by single gpu
num_workers = 8              # for faster data loading - should be 16
max_audio_length = 16000*30  # 30s at 16kHz
max_text_length = 146        # max text sequence len
max_samples_per_dataset = -1 # -1 == all
upsample_hard_datasets = 1.  # upsample hard datasets by this factor
pack_multiple_samples = False # allow multiple samples per seq
pr_prefix_dataset = 0.0      # prob of prefixing dataset name to eos prompt
sound_effects = False        # data augmentation - can lead to mem leaks
only_ds = ''                 # only evaluate dataset1,dataset2
limit_test_batches = 3000    # max number of batches to test
split = 'test'               # test/train
# DDP
precision = 'bf16-mixed' # '16-mixed', '32'
# tokenizer
pretrained_model_name_or_path = 'bert-base-uncased'
cache_dir = './cache/'
# model
d_emb = 128
d_model = H = 768
n_layers = 16
gradient_checkpointing = False
ignore_index = -100
bidirec_prefix_len = 0       # attention is bidirec upto this prefix len
num_frames_to_stack = 4      # 3000 mel frames are stacked to 750 
quantize = False             # quantize to 6kbps

# read command line overrides
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
override_globals_from_cl(globals()) # overrides from command line or config file
config = dotdict({k: globals()[k] for k in config_keys}) # will be useful for logging
# -----------------------------------------------------------------------------
        

# -- various inits, derived attributes, I/O setup --
if 'bf' in precision and torch.cuda.get_device_capability(0) < (8,0):
    print(f"compute capability {torch.cuda.get_device_capability(0)} < (8,0) .. falling back to fp16")
    config.precision = '16-mixed'

if split == 'test':
    config.limit_test_batches = limit_test_batches = 1.0
    

train_tsvs, test_tsvs, val_tsvs = [], [], []

for x in DATASETS:
    DIR, DS = SpeechDataset.get_dir(f'/{x}/')
    files = glob.glob(DIR + '/*.tsv')
    assert len(files)
    for text_tsv in files:
        if only_ds.strip() and not any(x.strip() and x in text_tsv.lower() for x in only_ds.lower().split(',')):
            print(colored(f'- excluding {text_tsv}!!', 'red', attrs=['bold']))
            continue
        if 'train' in text_tsv:
            train_tsvs.append(text_tsv)
        elif 'test' in text_tsv:
            test_tsvs.append(text_tsv)
        else:
            val_tsvs.append(text_tsv)


def index(s, t):
    try:
        idx = s.index(t)
    except ValueError:
        idx = None
    return idx

            
class LightningModel(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, cache_dir=cache_dir,)
        args.vocab_size = self.tokenizer.vocab_size
        self.model = Transformer(args)        
        self.dac_model = None

        self.args = args
        self.save_hyperparameters(args)
        self.preds, self.targets, self.inputs, self.dataset_sample_index, self.tsvs = [], [], [], [], []
        
    def setup(self, stage):
        # called on every process
        tsvs = train_tsvs if split == 'train' else test_tsvs
        self.datasets = {tsv: AudioDataset(tsv, self.tokenizer, self.args) for tsv in tsvs}
        self.prompt_len = 1  # [CLS] tok
        os.environ["TOKENIZERS_PARALLELISM"] = "false"  # disable tokenizer warnings
        
    def forward(self, x):
        return self.model(x)
    
    def test_step(self, batch, batch_idx):
        batches, _, _ = batch
        wave, inputs, targets, dataset_sample_index, tsvs = [], [], [], [], []
        for tsv, batch in batches.items():
            if batch is None: 
                continue
            t = batch['wave'], batch['inputs'], batch['targets'], batch['dataset_sample_index']
            wave += [t[0]]; inputs += [t[1]]; targets += [t[2]]; dataset_sample_index += [t[3]]  
            tsvs += [tsv]*len(t[0])
        
        wave, inputs, targets, dataset_sample_index = map(torch.cat, (wave, inputs, targets, dataset_sample_index))
        if not len(wave): return
        
        if self.args.quantize:
            wave = self.quantize(wave)
        
        preds = self.model.inference(wave, inputs[:,:self.prompt_len], max_text_len=targets.shape[-1])
        self.preds.append(preds.cpu()); self.targets.append(targets.cpu()) 
        self.inputs.append(inputs.cpu()); self.dataset_sample_index.append(dataset_sample_index.cpu())
        self.tsvs += tsvs
        
    def on_test_epoch_end(self):
        t = tuple(map(torch.cat, (self.preds, self.targets, self.inputs, self.dataset_sample_index)))
        tsvs = self.tsvs
        d = {'preds':t[0], 'targets':t[1], 'inputs':t[2], 'dataset_sample_index':t[3], 'tsvs': tsvs}
        os.makedirs(self.args.output_dir, exist_ok=True)
        save_path = f'{self.args.output_dir}/preds_{split}_quant_{self.args.quantize}.pt'
        torch.save(d, save_path); print('saved to', save_path)
        
        tokenizer = self.tokenizer
        inputs, preds, dataset_sample_index = d['inputs'].numpy(), d['preds'].numpy(), d['dataset_sample_index'].numpy()
        inputs, preds = map(tokenizer.batch_decode, (inputs, preds))
        # remove part until first [SEP], remove remaining [SEP]s 
        inputs = [s[index(s, tokenizer.sep_token):].replace(tokenizer.sep_token, '').strip() for s in inputs]
        # remove part after first [SEP]
        preds = [s[:index(s, tokenizer.sep_token)].strip() for s in preds]
        df = pd.DataFrame({'inputs':inputs, 'preds':preds, 'dataset_idx': dataset_sample_index[:,0], 
                           'sample_idx': dataset_sample_index[:,1], 'tsv': tsvs})
        save_path = save_path.replace('.pt', '.tsv')
        save_df_to_tsv(df, save_path); print('saved to', save_path)
        # print('WER', evaluate.load("wer").compute(predictions=preds, references=inputs))
        
    def test_dataloader(self):
        args = self.args
        dataloaders = {tsv: torch.utils.data.DataLoader(ds, batch_size=args.per_device_batch_size, 
                          drop_last=False, shuffle=split=='train', num_workers=args.num_workers, pin_memory=True, 
                          collate_fn=AudioDataset.collate_fn) for tsv, ds in self.datasets.items()}
        return CombinedLoader(dataloaders, 'max_size')
        
    def quantize(self, x):
        if self.dac_model is None:
            model_path = dac.utils.download(model_type="16khz")  # 6kbps
            self.dac_model = dac.DAC.load(model_path).to(x.device)
        assert x.ndim == 2
        x = x[:, None, :]         # [b 1 l]
        scale = x.abs().amax(dim=(-2,-1), keepdim=True)
        x = x / scale.clip(min=1e-6)
        x = self.dac_model.preprocess(x, 16000)
        x_q = self.dac_model.decode(self.dac_model.encode(x)[0])
        return F.pad(x_q, (0,x.shape[-1]-x_q.shape[-1])).squeeze(1)
    
    def on_load_checkpoint(self, checkpoint):
        ckpt_state_dict = checkpoint["state_dict"]
        model_state_dict = self.state_dict()

        missing_keys = set(model_state_dict) - set(ckpt_state_dict)
        unexpected_keys = set(ckpt_state_dict) - set(model_state_dict)

        for key in unexpected_keys:
            del ckpt_state_dict[key]
        
        for key in missing_keys:
            ckpt_state_dict[key] = model_state_dict[key]
        
        print(f'keys missing from ckpt : {missing_keys}')
        print(f'extra keys in ckpt : {unexpected_keys}')
        # TODO: also remove extra keys from optimizer state


seed_everything(config.seed, workers=True)

assert config.load_path
model = LightningModel.load_from_checkpoint(config.load_path, args=config, map_location="cpu")

trainer = Trainer(
    precision=precision,
    accelerator='gpu',
    logger=None,
    devices=1,  # single gpu
    log_every_n_steps=1,
    limit_test_batches=limit_test_batches,
)

trainer.test(model)


"""
python test.py --per_device_batch_size=8 --limit_test_batches=1.0 --pr_prefix_dataset=1.0 --load_path=outputs/all-128-2e-4-part2/checkpoint/epoch=3-step=1000000.ckpt --output_dir=./outputs/temp --only_ds=multi --split=test 
"""
