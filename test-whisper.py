"""
Author: Ankit Gupta

Evaluate Whisper.

# full test sets
python test-whisper.py --per_device_batch_size=8 --limit_test_batches=3000 --output_dir=outputs/whisper-v3 --split=test

# partial train sets
python test-whisper.py --per_device_batch_size=8 --limit_test_batches=3000 --output_dir=outputs/whisper-v3 --split=train
"""

import os, sys, time, json, glob, shutil, socket, dac, whisper
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

from local_utils import make_output_dir, dotdict, ram_GB, free_ram, save_df_to_tsv, read_file, write_file, dotdict, group_parameters_for_optimizer, get_cosine_schedule_with_warmup, override_globals_from_cl

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
sound_effects = False        # data augmentation - can lead to mem leaks
only_ds = ''                 # only evaluate dataset1,dataset2
limit_test_batches = 3000    # max number of batches to test
split = 'test'               # test/train
pr_prefix_dataset = 0.0
# model
whisper_version = "large-v3"
# DDP
precision = '16-mixed' # '32'
# tokenizer
pretrained_model_name_or_path = 'bert-base-uncased'
cache_dir = './cache/'
ignore_index = -100
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

            
def log_mel_spectrogram(audio, n_mels=80, device=None):
    assert torch.is_tensor(audio)
    if device is not None:
        audio = audio.to(device)
    window = torch.hann_window(N_FFT).to(audio.device)
    stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2

    filters = mel_filters(audio.device, n_mels)
    mel_spec = filters @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.amax(dim=(-2,-1), keepdim=True) - 8.0)  # fixes the bug when batching
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec
    
    
class LightningModel(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, cache_dir=cache_dir,)
        self.dac_model = None
        self.model = whisper.load_model(whisper_version, device='cpu', download_root='./cache/whisper')
        self.options = whisper.DecodingOptions(language='en', without_timestamps=True)
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
        
        audio = whisper.pad_or_trim(wave)
        mel = log_mel_spectrogram(audio, n_mels=128 if 'v3' in whisper_version else 80).to(wave.device)
        # decode the audio
        results = whisper.decode(self.model, mel, self.options)
        preds = [r.text for r in results]
        
        self.preds += preds; self.targets.append(targets.cpu()) 
        self.inputs.append(inputs.cpu()); self.dataset_sample_index.append(dataset_sample_index.cpu())
        self.tsvs += tsvs
        
    def on_test_epoch_end(self):
        t = tuple(map(torch.cat, (self.targets, self.inputs, self.dataset_sample_index)))
        tsvs = self.tsvs
        d = {'targets':t[0], 'inputs':t[1], 'dataset_sample_index':t[2], 'tsvs': tsvs}
        os.makedirs(self.args.output_dir, exist_ok=True)
        
        tokenizer = self.tokenizer
        inputs, preds, dataset_sample_index = d['inputs'].numpy(), self.preds, d['dataset_sample_index'].numpy()
        inputs = tokenizer.batch_decode(inputs)
        # remove part until first [SEP], remove remaining [SEP]s 
        inputs = [s[index(s, tokenizer.sep_token):].replace(tokenizer.sep_token, '').strip() for s in inputs]
        df = pd.DataFrame({'inputs':inputs, 'preds':preds, 'dataset_idx': dataset_sample_index[:,0], 
                           'sample_idx': dataset_sample_index[:,1], 'tsv': tsvs})
        save_path = f'{self.args.output_dir}/preds_{split}_whisper_{whisper_version}_quant_{self.args.quantize}.tsv'
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

model = LightningModel(config)

trainer = Trainer(
    precision=precision,
    accelerator='gpu',
    logger=None,
    devices=1,  # single gpu
    log_every_n_steps=1,
    limit_test_batches=limit_test_batches,
)

trainer.test(model)


