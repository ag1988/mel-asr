"""
Author: Ankit Gupta
"""

from pathlib import Path
import os, sys, shutil, random, jsonlines, pickle, json, csv, glob, math, psutil, gc, ctypes, subprocess, contextlib, io
from itertools import islice
from ast import literal_eval
import queue, threading
import multiprocessing as mp

import numpy as np
import pandas as pd

import torch
from torch import nn


# ---- I/O utils ----

def read_file(file):
    file = str(file)
    
    if file.endswith('jsonl'):
        with jsonlines.open(file, 'r') as reader:
            return [d for d in reader.iter()]
    
    elif file.endswith('json'):
        with open(file, encoding='utf8') as f:
            return json.load(f)
    
    elif file.endswith('pt'):
        return torch.load(file, map_location='cpu')
    
    elif any([file.endswith(ext) for ext in ['pkl', 'pickle', 'pck', 'pcl']]):
        with open(file, 'rb') as f:
            return pickle.load(f)
        
    elif file.endswith('txt'):
        with open(file, encoding='utf8') as f:
            return f.read()


def write_file(data, file):
    file = str(file)
    
    if file.endswith('jsonl'):
        with jsonlines.open(file, mode='w') as writer:
            writer.write_all(data)

    elif file.endswith('json'):
        with open(file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
    
    elif file.endswith('pt'):
        torch.save(data, file)
        
    elif any([file.endswith(ext) for ext in ['pkl', 'pickle', 'pck', 'pcl']]):
        with open(file, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    elif file.endswith('txt'):
        with open(file, 'w', encoding='utf8') as f:
            f.write(data)


# ---- pandas utils ----


def csv_numlines(csv_path):
    return int(subprocess.check_output(f"wc -l {csv_path}", shell=True).split()[0]) - 1


def load_df_from_tsv(path, nrows=None, usecols=None, skiprows=None, single_row_index=None):
    """single_row_index loads only this row - slower than np memmap"""
    _path = path if isinstance(path, str) else path.as_posix()
    nrows = nrows if nrows and nrows > 0 else None
    
    if single_row_index is not None:
        o = subprocess.check_output(f"sed -n -e '{1}p' -e '{single_row_index+2}p' {path}", shell=True).decode("utf-8")
        _path = io.StringIO(o + '\n')
    
    return pd.read_csv(
        _path,
        sep="\t",
        header=0,
        encoding="utf-8",
        escapechar="\\",
        quoting=csv.QUOTE_NONE,
        na_filter=False,
        nrows=nrows,
        usecols=usecols,
        skiprows=skiprows,
    )


def save_df_to_tsv(dataframe, path):
    _path = path if isinstance(path, str) else path.as_posix()
    dataframe.to_csv(
        _path,
        sep="\t",
        header=True,
        index=False,
        encoding="utf-8",
        escapechar="\\",
        quoting=csv.QUOTE_NONE,
    )


def dataframe_apply(df, func, num_workers=8):
    dfs = np.array_split(df, num_workers)
    return pd.concat(mp.Pool(num_workers).map(func, dfs))


# ---- training utils ----

def make_output_dir(output_dir, scripts_to_save=glob.glob("*.py")):
    """output_dir: path to output dir to create
       scripts_to_save: relative paths of files to save
    """
    os.makedirs(output_dir, exist_ok=True)
    # remove prev log dir
    # shutil.rmtree(os.path.join(output_dir, 'log'), ignore_errors=True)
    code_dir = os.path.join(output_dir, 'scripts')
    os.makedirs(code_dir, exist_ok=True)
    for script in scripts_to_save:
        dst_file = os.path.join(code_dir, os.path.basename(script))
        shutil.copyfile(script, dst_file)
    

def cleanup_state_dict_(state_dict, unwanted_prefix='_orig_mod.'):
    for k in list(state_dict.keys()):
        new_k = k.replace(unwanted_prefix, '')
        if new_k != k:
            # print(f'correcting {k} --> {new_k}')
            state_dict[new_k] = state_dict.pop(k)
    return state_dict

    
def save_training(output_dir, model, tokenizer, training_args=None, training_state_dict=None):
    # Save a trained model, configuration and tokenizer
    model = model.module if hasattr(model, 'module') else model
    
    # If we save using the predefined names, we can load using `from_pretrained`
    output_config_file = os.path.join(output_dir, 'config.json')
    write_file(model.config.to_dict(), output_config_file)
    
    output_model_file = os.path.join(output_dir, 'pytorch_model.bin')
    state_dict = model.state_dict()
    state_dict = cleanup_state_dict_(state_dict, unwanted_prefix='_orig_mod.')
    torch.save(state_dict, output_model_file)
    
    if training_args is not None:
        output_args_file = os.path.join(output_dir, 'training_args.json')
        write_file(training_args, output_args_file)

    if training_state_dict is not None:
        training_state_file = os.path.join(output_dir, 'training_state.bin')
        torch.save(training_state_dict, training_state_file)
    
    tokenizer.save_pretrained(output_dir)


# def store_checkpoint(args):
#     # save a copy of the current contents of the output dir  
#     out_dir = args.output_dir
#     chkpt_idx = len([x for x in os.listdir(out_dir) if x.startswith('chkpt_')])
#     shutil.copytree(out_dir, os.path.join(out_dir, f'chkpt_{chkpt_idx}'), ignore=shutil.ignore_patterns('chkpt_*'))


def count_parameters(module, only_trainable=False):
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not only_trainable)



# ---- optimizer utils ----

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1, min_lr_scale=0.):
    """while decaying lr >= min_lr_scale * maxlr """
    assert num_warmup_steps <= num_training_steps
    def lr_lambda(current_step, *args):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        if progress >= 1:
            return min_lr_scale
        lr_scale = (1 + math.cos(math.pi * progress)) / 2  # [0,1] 
        return min_lr_scale + (1 - min_lr_scale) * lr_scale
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def group_parameters_for_optimizer(model, weight_decay=1e-2):
    """We are separating out all parameters of the model into two buckets: those that will experience
       weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
    """        
    blacklist_weight_modules = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm, 
                                nn.SyncBatchNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, 
                                nn.InstanceNorm3d, nn.LayerNorm, nn.LocalResponseNorm)
    # 0/1D params wont be weight decayed
    # i.e. weight tensors in matmuls + embeddings decay, biases and layernorms dont
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            if not p.requires_grad or hasattr(p, "_optim"): 
                continue
            if (p.dim() < 2 or getattr(p, '_no_weight_decay', False) 
                or isinstance(m, blacklist_weight_modules)):
                setattr(p, "_optim", {"weight_decay": 0.0})

    # weight decay the rest
    all_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_params = [p for p in all_dict.values() if not hasattr(p, "_optim")]
    optim_groups = [{'params': decay_params, 'weight_decay': weight_decay}]

    # Add params with special attr
    all_parameters = list(all_dict.values())

    hps = [getattr(p, "_optim") for p in all_parameters if hasattr(p, "_optim")]
    hps = [
        dict(s) for s in set(frozenset(hp.items()) for hp in hps)
    ]  # Unique dicts
    for hp in hps:
        params = [p for p in all_parameters if getattr(p, "_optim", None) == hp]
        optim_groups.append({"params": params, **hp})  

    return optim_groups


# ---- general utils ----

def override_globals_from_cl(globals):
    for arg in sys.argv[1:]:
        # assume it's a --key=value argument
        assert arg.startswith('--')
        key_val = arg.split('=')
        key, val = key_val[0], '='.join(key_val[1:])
        key = key[2:]
        if key in globals:
            try:
                # attempt to eval it (e.g. if bool, number, or etc)
                attempt = literal_eval(val)
            except (SyntaxError, ValueError):
                # if that goes wrong, just use the string
                attempt = val
            # ensure the types match ok
            assert type(attempt) == type(globals[key]), f"{attempt} != {key} {globals[key]}" 
            # cross fingers
            print(f"Overriding: {key} = {attempt}")
            globals[key] = attempt
        else:
            raise ValueError(f"Unknown config key: {key}")


class dotdict(dict):     
    """dict with dot.notation access to string attributes.
    Example:
        >>> d = dotdict({'foo': 1, 2: 1})
        >>> d.foo = {'bar': 'baz'}
        >>> d.foo.bar, d[2]
        ('baz', 1)
    """      
    def __getattr__(*args):
        val = dict.__getitem__(*args)
        return dotdict(val) if isinstance(val, dict) else val      
    __setattr__ = dict.__setitem__     
    __delattr__ = dict.__delitem__


def tensor_GB(t, ndigits=1):
    return round(t.element_size() * t.numel() / 2**30, ndigits)


def ram_GB(job=False):
    if not job:
        return psutil.Process().memory_info().rss / 2**30
    try:
        s = os.popen(f"bjobs -noheader -o mem {os.environ['LSB_JOBID']}").read().strip()
        f = float(s.split()[0])
        if 'Mbytes' in s: f /= 1024
        return f
    except:
        return None


def set_seed(seed=42):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch`.
    Args:
        seed (`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dict_diff(d, iterable):
    return {k:v for k, v in d.items() if k not in iterable}


def dict_prefix(d, prefix):
    return {prefix+k:v for k, v in d.items()}


def free_ram():
    gc.collect()
    ctypes.CDLL("libc.so.6").malloc_trim(0) # very helpful sometimes


class DummyFile(object):
    def write(self, *args, **kwargs): pass
    def flush(self, *args, **kwargs): pass

@contextlib.contextmanager
def nostdout():
    """supress printing
       with @nostdout():
           your code
       can be used as decorator @nostdout() over funcs
    """
    orig_stdout = sys.stdout
    sys.stdout = DummyFile()
    yield
    sys.stdout = orig_stdout

    
