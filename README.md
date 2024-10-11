This repository contains the accompanying code for the paper:

> **Exploring the limits of decoder-only models trained on public speech recognition corpora**\
> Ankit Gupta, George Saon, Brian Kingsbury\
> [[PDF]](https://www.isca-archive.org/interspeech_2024/gupta24_interspeech.pdf)

## Table of Contents
- [Setup](#setup)
- [Dataloader](#dataloader)
- [Experiments](#exp)

## Setup <a name="setup"></a>

### Requirements
This repository was tested on Python 3.10+ and [Pytorch 2.1+](https://pytorch.org/get-started/locally/).
You can create a new conda environment and install the required packages as follows. For using `bf16` make sure to create this env on a machine with A100s.

```bash
# create a new conda env (e.g. called "dota")
conda create -n dota python=3.10  &&  source activate dota

# latest torch nightly build
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121

pip install jsonlines tqdm einops opt_einsum more-itertools ffmpeg-python evaluate Cython matplotlib pandas h5py soundfile termcolor

pip install hydra-core hydra-colorlog hydra-optuna-sweeper pyrootutils rich lightning wandb timm torchmetrics pytest datasets git+https://github.com/huggingface/transformers

conda install -c pytorch ffmpeg 
conda install -c conda-forge sox   

# install whisper
pip install git+https://github.com/openai/whisper.git 

# install dac
pip install descript-audio-codec
```


## Dataloader <a name="dataloader"></a>

Logic for loading datasets and batching is in `wave_dataloader.py`. Currenly, the files contain hardcoded paths to `/dataset/speechdata/` which is an IBM internal storage. You'll need to modify `wave_dataloader.py` according to the dataset and the data format you're using.


## Experiments <a name="exp"></a>

Training a 634M sized prefix LM model on 8x A100s
```bash
python train.py --num_workers=15  --per_device_batch_size=16  --full_batch_size=128 --learning_rate=2e-4  --max_samples_per_dataset=-1  --output_dir=./outputs/dota-634M-8x-bidirec  --val_check_interval=10000 --sound_effects=True --pack_multiple_samples=True --d_model=1280 --H=1280 --n_layers=32 --num_frames_to_stack=8 --pr_prefix_dataset=0.0 --bidirec_prefix_len=375
```

#### Evaluation : in the provided code, the dataloader will truncate the waveforms to 30sec. In your datasets, you'll need to split longer waveforms into 30sec chunks, transcribe each and then concatenate the respective outputs during WER computation.

Evaluate trained model on test sets
```bash
python test.py --per_device_batch_size=2 --limit_test_batches=3000 --d_model=1280 --H=1280 --n_layers=32 --num_frames_to_stack=8 --pr_prefix_dataset=0.0 --bidirec_prefix_len=375 --load_path=./outputs/first_try/checkpoint/<checkpoint you want to evaluate>.ckpt --output_dir=./outputs/dota-634M-8x-bidirec/results/ --split=test --quantize=False
```

Evaluate Whisper on test sets
```bash
python test-whisper.py --per_device_batch_size=6 --limit_test_batches=3000 --output_dir=./outputs/whisper/ --split=test --whisper_version=large-v3 --quantize=False
```



## Citation
If you use this codebase, or otherwise found our work valuable, please cite:
```
@inproceedings{gupta2024dota,
  title = {Exploring the limits of decoder-only models trained on public speech recognition corpora},
  author = {Gupta, Ankit and Saon, George and Kingsbury, Brian},
  year = {2024},
  pages = {252-256},
  booktitle={INTERSPEECH},
}
```
