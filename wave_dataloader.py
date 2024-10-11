import os, random, h5py, argparse, time, re, string
from functools import partial 
from tqdm.auto import tqdm

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

os.environ['HF_DATASETS_CACHE'] = './cache'
os.environ['HF_DATASETS_OFFLINE'] = '1'
from datasets import Dataset

from local_utils import load_df_from_tsv, ram_GB, dotdict

from whisper.normalizers import EnglishTextNormalizer


normalize_ = EnglishTextNormalizer()
def normalize(s):
    e = normalize_(s).replace('\n','').strip()
    if not e: 
        return e
    # insert space after digits
    return ''.join(c+' ' if (c.isdigit() and i < len(e)-1 and len(e[i+1].strip())) else c for i, c in enumerate(e+' ')).strip()


class SoundEffects:
    def __init__(self, low=.9, high=1.1, prob_lowpass=0.001, 
                 prob_speed=0.001, prob_tempo=0.2, prob_reverb=0.001):
        """caution: keep probs low as some of these cause memory leaks"""
        # reverb is expensive
        self.low, self.high, self.prob_lowpass, self.prob_reverb = low, high, prob_lowpass, prob_reverb
        self.prob_speed, self.prob_tempo = prob_speed, prob_tempo
    
    def __call__(self, wave, sample_rate, out_sample_rate=16000, sound_effects=False):
        # wave [channels len]
        sox_effects = []
        
        if random.random() < self.prob_lowpass:
            sox_effects.append(["lowpass", "-1", "300"])          # apply single-pole lowpass filter
        
        if random.random() < self.prob_speed:  # leads to mem leak!
            # assert self.low <= 1, 'only slow down as speedup done on gpu'
            # speed_factor = self.low + random.random()*(1-self.low)
            speed_factor = self.low + random.random()*(self.high-self.low)
            sox_effects.append(["speed", f'{speed_factor:.2f}'])  # change speed - changes sample rate 
            
        if random.random() < self.prob_tempo:
            tempo_factor = self.low + random.random()*(self.high-self.low)
            sox_effects.append(["tempo", f'{tempo_factor:.2f}'])  # apply single-pole lowpass filter
            
        if random.random() < self.prob_reverb:
            sox_effects.append(["reverb", "-w"])                  # reverbration
            
        if sound_effects and sox_effects:
            wave, sample_rate = torchaudio.sox_effects.apply_effects_tensor(
                wave, sample_rate, sox_effects)
        
        if sample_rate != out_sample_rate:
            wave = torchaudio.functional.resample(wave, sample_rate, out_sample_rate)
        if wave.ndim > 1 and wave.shape[0] > 1:
            wave = wave.flatten(end_dim=-2).mean(0, keepdim=True)
        return wave


class SpeechDataset:    
    """this example class should contain the logic for reading various datasets and needs to be implemented by the user"""
    
    DATASETS = ['MULTILINGUAL_LIBRI_SPEECH_HDF']
    
    def __init__(self, text_tsv, max_samples_per_dataset=-1):
        self.text_tsv = text_tsv
        self.dir, self.ds_name = self.get_dir(text_tsv)
        self.ds_index = SpeechDataset.DATASETS.index(self.ds_name)
        
        usecols = ['audio_key', 'text', 'rate']
        self.hf_path = self.dir + 'audio.hdf5'
        
        df = load_df_from_tsv(text_tsv, usecols=usecols, nrows=max_samples_per_dataset)
        df = self.dataset_filter(df, text_tsv)
        
        self.df_hf = Dataset.from_pandas(df)  # using pandas dataframe directly leads to mem leaks
        del df
        
    def get_row(self, i):
        return dotdict(self.df_hf[i])
    
    def dataset_filter(self, df, path):
        df.reset_index(drop=True, inplace=True)
        return df
    
    @staticmethod
    def get_dir(path):
        DS = [x for x in SpeechDataset.DATASETS if f'/{x}/' in path]
        assert len(DS) == 1, f'{path} {DS}'
        DS = DS[0]
        DIR = f'/dataset/speechdata/{DS}/data/'
        return DIR, DS
    
    def __len__(self):
        return len(self.df_hf)
    
    def get_wav(self, i, max_secs=-1, sound_effects=False):
        DIR = self.dir
        hf_path, row = self.hf_path, self.get_row(i)
        try:
            sr, k = row.rate, row.audio_key
            l = int(max_secs*sr) if max_secs > 0 else None
            with h5py.File(hf_path) as hf:  # opened here to avoid mem leak
                wav = torch.from_numpy(hf[k][:l]).view(1,-1)
            if wav.dtype == torch.int16:
                wav = wav / 2**15
            else:
                assert False
            
            # resample, apply sounds effects
            return SoundEffects()(wav, sr, out_sample_rate=16000, sound_effects=sound_effects)
        except Exception as e:
            print(e)
            return None


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, text_tsv, tokenizer,
                 config=dotdict(max_audio_length = 16000*30,  # 30s at 16kHz
                                 max_text_length = 146,        # max input sequence len
                                 max_samples_per_dataset = -1, # -1 == all
                                 pack_multiple_samples = True, # allow multiple samples per seq
                                 pr_prefix_dataset = 0.0,
                                 sound_effects = True,         # speed, tempo, reverb
                                 upsample_hard_datasets = 1,   # upsample hard datasets
                               )
                ):
        """if pack_mutliple_samples, use num_workers >= 16 due to large number of reads
        """
        super().__init__()
        # copy attrbutes of args to self
        self.__dict__.update(config)
        self.config = config
        self.text_tsv = text_tsv
        
        self.ds = SpeechDataset(text_tsv, max_samples_per_dataset=self.max_samples_per_dataset)
        self.is_train = 'train' in text_tsv.lower()
        
        self.upsampling_factor = 1
        assert self.upsample_hard_datasets >= 1
        if self.is_train and not any(x.lower() in text_tsv.lower() for x in ['libri', 'people', 'giga', 'spgi']):
            self.upsampling_factor = self.upsample_hard_datasets
        
        print(f'{text_tsv} : {len(self.ds)}  samples ↑{self.upsampling_factor}x')
        print(f"current ram usage : {ram_GB(True)}")
        
        self.tokenizer = tokenizer    
        self.prompts = dotdict(eos = [tokenizer.sep_token_id],
                               dataset_name = tokenizer.encode(f"[{self.ds.ds_name.lower()}]", add_special_tokens=False),)
    
    def __len__(self):
        return int(len(self.ds) * self.upsampling_factor)
    
    def __getitem__(self, i):
        """ pad the audio to 30s, optionally append dataset name to text
           <eos> t1 t2 .. <eos> <pad> ..
           <dataset_name> <eos> t1 t2 .. <eos> <pad> ..
        """
        is_train = self.is_train
        if len(self) < len(self.ds) and is_train:
            # randomly cover entire ds if ds is downsampled
            i = i + len(self) * random.randint(0, len(self.ds)//len(self))
        
        i = i % len(self.ds)
        ds, config, prompts, orig_i = self.ds, self.config, self.prompts, i
        mal, mtl = config.max_audio_length, config.max_text_length
        waves, eos, text_ids, n = torch.zeros(0), prompts.eos, [], 0
        
        while len(waves) < mal and n < 6:
            row = ds.get_row(i)
            # apply sounds effects to train samples
            text = row.text.strip()
            wave = ds.get_wav(i, sound_effects=self.is_train and config.sound_effects)
            i = (i + 1) % len(ds)
            n += 1
            
            # keep if it fits in remaining len to avoid halucination
            if is_train and (wave is None or not (0 < wave.numel() <= mal-len(waves)) or not len(text)):
                continue
            
            assert wave.shape[0] == 1, f'{wave.shape} {ds.ds_name}'
            waves = torch.cat((waves, wave.view(-1)))
            text_ids += self.tokenizer.encode(normalize(text), add_special_tokens=False)
            
            if not (is_train and config.pack_multiple_samples and random.random() < 0.75): 
                break
             
        # 0-pad audio
        waves = waves[:mal]
        left_pad_len = int((mal-len(waves)) * random.random()) if (self.is_train and config.sound_effects) else 0
        waves = F.pad(waves, (left_pad_len, mal-len(waves) - left_pad_len))
        
        # pad text by <eos>
        prefix = (prompts.dataset_name if random.random() < config.pr_prefix_dataset else []) + eos
        text_ids += eos
        
        input_ids = (prefix + text_ids)[:mtl]
        labels = ([-100]*(len(prefix)-1) + text_ids)[:mtl]
        
        input_ids += eos*(mtl - len(input_ids))
        labels    += [-100]*(mtl - len(labels))
        
        return dict(
            wave = waves,
            inputs = torch.LongTensor(input_ids),
            targets = torch.LongTensor(labels),
            dataset_sample_index = torch.LongTensor([ds.ds_index, orig_i])
        )
    
    @staticmethod
    def collate_fn(features):
        wave, inputs, targets, dataset_sample_index = [], [], [], []
        for f in features:
            wave.append(f["wave"])
            inputs.append(f["inputs"])
            targets.append(f["targets"])
            dataset_sample_index.append(f["dataset_sample_index"])

        batch = dict(
            wave = torch.stack(wave).requires_grad_(False),
            inputs = torch.stack(inputs),
            targets = torch.stack(targets),
            dataset_sample_index = torch.stack(dataset_sample_index),
        )
        # if this returns a custom class it should have a pin_memory method
        return batch



class AudioConcatDataset(torch.utils.data.ConcatDataset):
    def __init__(self, tsvs, *args, **kwargs):
        super().__init__([AudioDataset(text_tsv, *args, **kwargs) 
                          for text_tsv in tsvs])
