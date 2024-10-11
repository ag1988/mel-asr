"""Transformer on padded logmels for ASR.

Author: Ankit Gupta
"""

import os, sys, time, json, glob, shutil, socket
from tqdm.auto import tqdm
from termcolor import colored
from copy import deepcopy
import numpy as np

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from transformers import AutoTokenizer

from local_utils import make_output_dir, read_file, dotdict, ram_GB, free_ram, save_df_to_tsv, read_file, write_file, dotdict, group_parameters_for_optimizer, get_cosine_schedule_with_warmup, override_globals_from_cl

from modeling import Transformer
from wave_dataloader import SpeechDataset, AudioDataset, AudioConcatDataset
DATASETS = SpeechDataset.DATASETS


torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

import warnings
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# I/O
output_dir = './outputs/temp'
val_check_interval = 10000        # val after these many steps
save_every_n_train_steps = 10000  # save checkpoint every these many steps
resume_path = ''                  # resume training
load_path = ''                    # init weights 
seed = 42
# logging
logger = ''  # 'wandb' disabled by default
# data
full_batch_size = 128        # num processes * per_device_batch_size * gradient_accumulation_steps
per_device_batch_size = 64   # should be manually set to largest micro-batch size allowed by single gpu
per_device_eval_batch_size = per_device_batch_size
num_workers = 16             # for faster data loading - should be 16
max_audio_length = 16000*30  # 30s at 16kHz
max_text_length = 146        # max text sequence len
max_samples_per_dataset = -1 # -1 == all
upsample_hard_datasets = 2.  # upsample hard datasets by this factor
pack_multiple_samples = True # allow multiple samples per seq
pr_prefix_dataset = 0.0      # prob of prefixing dataset name to eos prompt
sound_effects = True         # data augmentation - can lead to mem leaks
shuffle = True               # shuffle during training
exclude_ds = ''              # dataset1,dataset2 to exclude from training
only_ds = ''                 # only dataset1,dataset2 included in training
# adamw optimizer
learning_rate = 2e-4         # max lr
min_lr_scale = 0.0           # min lr == min_lr_scale * max lr
max_steps = int(1e6)         # total number of updates
weight_decay = 0.1
beta1, beta2 = 0.9, 0.99
gradient_clip_val = 1.0      # clip gradients at this value, or disable if == 0.0
fused_opt = False
num_warmup_steps = 1000      # how many steps to warm up for
# DDP
precision = 'bf16-mixed' # '16-mixed', '32'
num_nodes = 1
hostname = socket.getfqdn()
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

# world_size = os.environ['WORLD_SIZE']
devices = torch.cuda.device_count()
world_size = num_nodes * devices
assert full_batch_size % (world_size*per_device_batch_size) == 0, 'full_batch_size must be multiple of number of processes * per_device_batch_size'
config.accumulate_grad_batches = accumulate_grad_batches = full_batch_size // (world_size*per_device_batch_size)
config.checkpoint_dir = f"{output_dir}/checkpoint"
config.resume_path = config.resume_path if len(config.resume_path) else None


train_tsvs, test_tsvs, val_tsvs = [], [], []

for x in DATASETS:
    DIR, DS = SpeechDataset.get_dir(f'/{x}/')
    files = glob.glob(DIR + '/*.tsv')
    assert len(files)
    for text_tsv in files:
        if 'train' in text_tsv:
            if (any(x.strip() and x in text_tsv.lower() for x in exclude_ds.lower().split(',')) 
                or (only_ds.strip() and not any(x.strip() and x in text_tsv.lower() for x in only_ds.lower().split(',')))
               ): 
                print(colored(f'- excluding {text_tsv}!!', 'red', attrs=['bold']))
                continue
            train_tsvs.append(text_tsv)
        elif 'test' in text_tsv:
            test_tsvs.append(text_tsv)
        else:
            val_tsvs.append(text_tsv)


class LightningModel(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, cache_dir=cache_dir,)
        args.vocab_size = self.tokenizer.vocab_size
        self.model = Transformer(args)
        
        self.args = args
        self.save_hyperparameters(args)
    
    def setup(self, stage):
        # called on every process
        self.train_dataset = AudioConcatDataset(train_tsvs, self.tokenizer, self.args)
        self.val_dataset = AudioConcatDataset(test_tsvs, self.tokenizer, self.args)
        os.environ["TOKENIZERS_PARALLELISM"] = "false"  # disable tokenizer warnings
        
    def forward(self, x):
        return self.model(x)
    
    def shared_step(self, batch, batch_idx, phase='train'):
        loss, err, preds = self.model(**batch)
        if self.trainer.global_step % 5 == 0:
            for metric, x in [('loss', loss.cpu().item()), ('err', err.cpu().item()), ('ram', ram_GB(True))]:
                self.log(f"{phase}/{metric}", x, on_step=phase=='train', 
                         on_epoch=phase!='train', prog_bar=True, sync_dist=True)
        return loss, preds
    
    def training_step(self, batch, batch_idx):
        loss, preds = self.shared_step(batch, batch_idx, 'train')
        
        s = 1000
        if self.trainer.global_step % s >= s-50:
            free_ram()
        return loss
    
    def validation_step(self, batch, batch_idx):
        _, preds = self.shared_step(batch, batch_idx, 'val')
    
    def configure_optimizers(self):
        args = self.args
        param_groups = group_parameters_for_optimizer(self.model, weight_decay=args.weight_decay)
        self.optimizer = torch.optim.AdamW(param_groups, lr=args.learning_rate, betas=(args.beta1, args.beta2), 
                                           foreach=None, fused=args.fused_opt)        
        # scheduler should be called either every step (default) or every epoch
        scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, num_warmup_steps=args.num_warmup_steps, 
            num_training_steps=self.trainer.estimated_stepping_batches,
            min_lr_scale=args.min_lr_scale,
        )
        print('num train steps', self.trainer.estimated_stepping_batches)
        self.scheduler = scheduler
        return [self.optimizer], [{"scheduler": self.scheduler, "interval": "step", "frequency": 1}]
     
    def train_dataloader(self):
        args = self.args
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=args.per_device_batch_size, 
                          drop_last=True, shuffle=self.args.shuffle, num_workers=args.num_workers, pin_memory=True, 
                          collate_fn=AudioDataset.collate_fn)
    
    def val_dataloader(self):
        args = self.args
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=args.per_device_eval_batch_size, 
                          drop_last=False, shuffle=False, num_workers=min(16,args.num_workers), pin_memory=True, 
                          collate_fn=AudioDataset.collate_fn)
    
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

output_dir = config.output_dir

shutil.rmtree(output_dir, ignore_errors=True)
os.makedirs(output_dir, exist_ok=True)

if config.load_path:
    model = LightningModel.load_from_checkpoint(config.load_path, args=config, map_location="cpu")
else:
    model = LightningModel(config)

callback_list, logger = [LearningRateMonitor(logging_interval="step")], None

if config.logger == 'wandb':
    callback_list += [ModelCheckpoint(
        dirpath=config.checkpoint_dir,
        save_top_k=1,
        every_n_train_steps=config.save_every_n_train_steps,)
    ]
    logger = WandbLogger(project="dota")

trainer = Trainer(
    precision=config.precision,
    accelerator='gpu',
    max_steps=config.max_steps,
    accumulate_grad_batches=config.accumulate_grad_batches,
    logger=logger,
    callbacks=callback_list,
    gradient_clip_val=config.gradient_clip_val,
    devices=devices,
    log_every_n_steps=60,
    val_check_interval=config.val_check_interval,
)

trainer.fit(model, ckpt_path=config.resume_path)


"""
python train.py --num_workers=24  --per_device_batch_size=64  --full_batch_size=128 --learning_rate=2e-4  --max_samples_per_dataset=-1  --output_dir=./outputs/temp --logger='' --val_check_interval=5000
"""
