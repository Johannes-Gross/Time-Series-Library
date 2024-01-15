# Enhancing Temporal Causality in TimesNet with 2D TCN

The current repository is based on a forked version of the TSlib and contains the code base used in a project for Deep Learning class taught during the Fall semester of 2023 at ETH Zurich.

In the scope of this project, we attempted to add the element of causality in the seminal TimesNet work. In order to test our approach we deploy a plethora of datasets from various domains and train different baseline models to compare with our proposed method. 

TSlib is an open-source library for deep learning researchers, especially for deep time series analysis.

We provide a neat code base to evaluate advanced deep time series models or develop your model, which covers five mainstream tasks: **long- and short-term forecasting, imputation, anomaly detection, and classification.**

:triangular_flag_on_post:**News** (2023.10) We add an implementation to [iTransformer](https://arxiv.org/abs/2310.06625), which is the state-of-the-art model for long-term forecasting. The official code and complete scripts of iTransformer can be found [here](https://github.com/thuml/iTransformer).

:triangular_flag_on_post:**News** (2023.09) We added a detailed [tutorial](https://github.com/thuml/Time-Series-Library/blob/main/tutorial/TimesNet_tutorial.ipynb) for [TimesNet](https://openreview.net/pdf?id=ju_Uqw384Oq) and this library, which is quite friendly to beginners of deep time series analysis.

 
## Usage

1. Install Python 3.10.6. For convenience, execute the following command.

```
pip install -r requirements.txt
```

2. Prepare Data. You can obtain the well pre-processed datasets from [[Google Drive]](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2?usp=sharing) or [[Baidu Drive]](https://pan.baidu.com/s/1r3KhGd0Q9PJIUZdfEYoymg?pwd=i9iy), Then place the downloaded data in the folder`./dataset`. Here is a summary of supported datasets.

<p align="center">
<img src=".\pic\dataset.png" height = "200" alt="" align=center />
</p>

3. Train and evaluate model. We provide the experiment scripts for all benchmarks under the folder `./scripts/`. You can reproduce the experiment results as the following examples:

```
# long-term forecast
bash ./scripts/long_term_forecast/Weather_script/TimesNet.sh
# short-term forecast
bash ./scripts/short_term_forecast/TimesNet_M4.sh
# long-term forecast with causality mode
bash ./scripts/long_term_forecast/Weather_script/TimesNet.sh --mode causal
# short-term forecast with causality mode
bash ./scripts/short_term_forecast/TimesNet_M4.sh --mode causal
```

Here, and by using the mode option for our arguments we can force the TimesNet model to make use of the appropriate causality mode. We make a distinction between using a causal TimesNet model and one that make no use of causal structure.

4. Develop your own model.

- Add the model file to the folder `./models`. You can follow the `./models/Transformer.py`.
- Include the newly added model in the `Exp_Basic.model_dict` of  `./exp/exp_basic.py`.
- Create the corresponding scripts under the folder `./scripts`.
