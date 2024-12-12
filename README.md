<h1 align="center">CFDBench</h1>

<div align="center">
    <a href="https://huggingface.co/datasets/chen-yingfa/CFDBench">🤗Dataset</a> | <a href="https://www.preprints.org/manuscript/202309.1550/v1">Paper (preprints.org)</a> |
    <a href="https://arxiv.org/abs/2310.05963">Paper (arXiv)</a>
</div>

<div align="center">
    Yining Luo, Yingfa Chen, and Zhen Zhang</br>
    Tsinghua University</br>
    chenyingfa1999@qq.com
</div>

<div></br></div>

![flow-examples](figs/flow-examples.png)

This is the code for the paper: [CFDBench: A Large-Scale Benchmark for Machine Learning Methods in Fluid Dynamics](https://arxiv.org/abs/2310.05963).

CFDBench is the first large-scale benchmark for evaluating machine learning methods in fluid dynamics with varied boundary conditions (BCs), physical properties, and domain geometries. It consists of four classic problems in computational fluid dynamics (CFD), with many varying operating parameters, making it perfect for testing the inference-time generalization ability of surrogate models. Such generalizability is essential for avoiding expensive re-training when applying surrogate models to new problems.

## Quick Links

- [Data](#data)
- [Models](#models)
- [Main Results](#main-results)
- [How to Run?](#how-to-run)
- [How to Add New Models/Datasets?](#how-to-add-new-modelsdatasets)
- [Citation](#citation)

## Data

Main download links: 

Interpolated data (~13.4GB):

- [🤗HuggingFace](https://huggingface.co/datasets/chen-yingfa/CFDBench)

Raw data (~460GB):

- [🤗HuggingFace](https://huggingface.co/datasets/chen-yingfa/CFDBench-raw)
- [Baidu Drive (百度网盘)](https://pan.baidu.com/s/1p0q60cv2hFZ7UcIf3XKSaw?pwd=cfd4) (提取码: cfd4)

> You likely do not need the raw data if you are just trying to evaluate and/or train neural networks on our data.

### Data Generation

The directory `generation-code` contains the code for creating the mesh (ICEM code) and the schema code for batch generation in ANSYS Fluent.

> This part takes a lot of time, and you are better off using our generated data instead.


### Data Interpolation

After generating data with numerical algorithms, it is then interpolated to a grid of 64x64. The raw data before interpolation is very large; the link below is the interpolated data.

Main download link: [[click here]](https://cloud.tsinghua.edu.cn/d/435413b55dea434297d1/)

Contains 4 problems:

- `cavity`: Lid-driven cavity flow
- `tube`: Flow through a circular tube
- `dam`: Flow over a dam
- `cylinder`: Flow around a cylinder

> The cylinder flow is separated into three files because the file size exceeds the upload limit.

Each dataset includes 3 subsets, corresponding to changing BCs, domain geometries, and physical properties.

The directory tree for the datasets:

```
▼ cavity/
    ▼ bc/
        ▼ case0000/
            ▼ u.npy
            ▼ v.npy
        ► case0001/
    ► geo/
    ► prop/
► tube/
► dam/
► cylinder/
```

The actual data for each velocity field is stored in `u.npy` and `v.npy`.

## Models

![models](figs/input-output-overview.png)

The basic types of models are autoregressive and non-autoregressive:

- Autoregressive:
    - Auto-FFN
    - Auto-DeepONet
    - Auto-EDeepONet
    - Auto-DeepONetCNN
    - ResNet
    - U-Net
    - FNO

- Non-autoregressive
    - FFN
    - DeepONet

The implementation of the models is located in `src/models`

## Main Results

### Multi-Step Inference

![multi-step-results](figs/result-multistep-infer.png)

### Autoregressive Models

![autoregressive-results](figs/result-auto-summary.png)

### Nonautoregressive Models

![nonautoregressive-results](figs/result-nonauto.png)

## How to Run?

Tested on:

- PyTorch 1.13.3+cu117
- Python 3.9.0

### Installation

Make sure you have access to CUDA GPU, then setup the environment using

```shell
pip install -r requirements
```

### Prepare Data

Move the downloaded data into a `data` directory next to `src` directory, such that the directory
looks like:

```
▼ data/
    ▼ cavity/
        ▼ bc/
        ▼ geo/
        ▼ prop/
    ► tube/
    ► dam/
    ► cylinder/
► generation-code/
► src/
.gitignore
README.md
```

### Training

In the `src` directory, run `train.py` or `train_auto.py` to train non-autoregressive or autoregressive models respectively. Specify the model with `--model`, it must be one of the following:

Model | Value for `--model` | Script
--- | --- | ---
Non-autoregrssive FFN | `ffn` | `train.py`
Non-autoregressive DeepONet | `deeponet` | `train.py`
Autoregressive Auto-FFN | `auto_ffn` | `train_auto.py`
Autoregressive Auto-DeepONet | `auto_deeponet` | `train_auto.py`
Autoregressive Auto-EDeepONet | `auto_edeeponet` | `train_auto.py`
Autoregressive Auto-DeepONetCNN | `auto_deeponet_cnn` | `train_auto.py`
Autoregressive ResNet | `resnet` | `train_auto.py`
Autoregressive U-Net | `unet` | `train_auto.py`
Autoregressive FNO | `fno` | `train_auto.py`

#### Training Examples

For example, run FNO on the cavity flow subset with all cases:

```bash
python train_auto.py --model fno --data cavity_prop_bc_geo
```

or, run DeepONet on the dam flow PROP + GEO subset:

```bash
python train.py --model deeponet --data dam_prop_geo
```

Results will be saved to `result/` directory by default, but can be customized with the `--output_dir` argument.

For more options, such as model hyperparameters, run `python train.py -h` or `python train_auto.py -h`.

### Inference

Set `--mode test` when executing `train.py` or `train_auto.py`.

### Hardware Requirements

See the Results section in the paper. Reduce the batch size if you run out of VRAM.

## How to Add New Models/Datasets?

Our code is highly extensible and modular, and it is very easy to add new datasets or models.

### Models

To add a new model, simply create a class that inherits one of the following base models:

- `CfdModel`: If your model is nonautoregressive
- `AutoCfdModel`: If your model is autoregressive

Then depending on which base model, you have to implement just 2 or 3 methods in addition to the model architecture itself.

- Nonautoregressive: `forward`, `generate_one`.
- Autoregressive: `forward`, `generate_one`, and `generate_many`.

Then, if your model requires new hyperparameters, add them to the argument parsed in `args.py`.

Finally, you should create add a new `elif` in for the instantiation of your model. For autoregressive models, change `init_model` in `utils_auto.py` as follows.

```python
def init_model(args: Args) -> AutoCfdModel:
    # ...
    elif args.model == "your_model_name":
        model = YourModeClass(
            in_chan=args.in_chan,
            out_chan=args.out_chan,
            n_case_params=n_case_params,
            loss_fn=loss_fn,
            some_arg=args.some_arg,
            # ... more arguments for your model.  (or you can just pass `args`)
        ).cuda()
        return model
```

For nonautoregressive models, change `init_model` in `train.py` in a similar manner.

### Dataset

You just have to implement a new subclass for `CfdDataset` or `CfdAutoDataset` in `dataset/base.py`. Like any PyTorch Dataset, it needs to implement `__getitem__` and `__len__`. But for multi-step inference, it also has to load features into a member attribute named `all_features` which should be a list of Tensor or NumPy arrays.

Then, add a new `elif` in `get_dataset` in `dataset/__init__.py` for the instantiation of your data.

## Citation

If you find this code useful, please cite our paper:

```
@article{CFDBench,
  title={CFDBench: A Large-Scale Benchmark for Machine Learning Methods in Fluid Dynamics},
  author={Yining, Luo and Yingfa, Chen and Zhen, Zhang},
  url={https://arxiv.org/abs/2310.05963},
  year={2023}
}
```
