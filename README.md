Part of the 9th place solution for [the Bristol-Myers Squibb – Molecular Translation challenge](https://www.kaggle.com/c/bms-molecular-translation/overview) translating
images containing chemical structures into InChI (International Chemical Identifier) texts.

This repo is partially based on the following resources:
* [Y.Nakama's](https://www.kaggle.com/yasufuminakama) [tokenization](https://www.kaggle.com/yasufuminakama/inchi-preprocess-2)
* [Heng's](https://www.kaggle.com/hengck23) [transformer decoder](https://www.kaggle.com/c/bms-molecular-translation/discussion/231190)
* [Sam Stainsby's](https://www.kaggle.com/stainsby) [external images creation](https://www.kaggle.com/stainsby/improved-synthetic-data-for-bms-competition-v3) updated by [ZFTurbo](https://www.kaggle.com/zfturbo)


## Requirements
* install and activate [the conda environment](environment.yml)
* download and extract the data into `/data/bms/`
* extract and move [sample_submission_with_length.csv.gz](models/sample_submission_with_length.csv.gz) into `/data/bms/`
* tokenize training inputs: `python datasets/prepocess2.py`
* if you want to use pseudo labeling, execute: `python datasets/pseudo_prepocess2.py your_submission_file.csv`
* if you want to use external images, you can create with the following commands:
```
python r09_create_images_from_allowed_inchi.py
python datasets/extra_prepocess2.py 
```

## Training
This repo supports training any VIT/SWIN/CAIT transformer models from [timm](https://github.com/rwightman/pytorch-image-models/) as encoder together with the fairseq transformer decoder.


Here is an example configuration to train a SWIN `swin_base_patch4_window12_384` as encoder and 12 layer 16 head fairseq decoder:
```
python -m torch.distributed.launch --nproc_per_node=N train.py --logdir=logdir/ \
    --pipeline --train-batch-size=50 --valid-batch-size=128 --dataload-workers-nums=10 --mixed-precision --amp-level=O2  \
    --aug-rotate90-p=0.5 --aug-crop-p=0.5 --aug-noise-p=0.9 --label-smoothing=0.1 \
    --encoder-lr=1e-3 --decoder-lr=1e-3 --lr-step-ratio=0.3 --lr-policy=step --optim=adam --lr-warmup-steps=1000 --max-epochs=20 --weight-decay=0 --clip-grad-norm=1 \
    --verbose --image-size=384 --model=swin_base_patch4_window12_384 --loss=ce --embed-dim=1024 --num-head=16 --num-layer=12 \
    --fold=0 --train-dataset-size=0 --valid-dataset-size=65536 --valid-dataset-non-sorted
```

For pseudo labeling, use `--pseudo=pseudo.pkl`. If you want subsample the pseudo dataset, use: `--pseudo-dataset-size=448000`.
For using external images, use `--extra` (`--extra-dataset-size=448000`).

After training, you can also use Stochastic Weight Averaging (SWA) which gives a boost around 0.02:
```
python swa.py --image-size=384 --input logdir/epoch-17.pth,logdir/epoch-18.pth,logdir/epoch-19.pth,logdir/epoch-20.pth
```

## Inference

Evaluation:
```
python -m torch.distributed.launch --nproc_per_node=N eval.py --mixed-precision --batch-size=128 swa_model.pth
```

Inference:
```
python -m torch.distributed.launch --nproc_per_node=N inference.py --mixed-precision --batch-size=128 swa_model.pth
```

Normalization with RDKit:
```
./normalize_inchis.sh submission.csv
```
