# Enhancing Temporal Causality in TimesNet with 2D TCN

This is a forked repository of the TSlib and contains the code used in the deep learning project named "Enhancing Temporal Causality in TimesNet with 2D TCN". This project was conducted following the ETH Zürich Deep Learning class of HS2023.

In this study, we explore the integration of a 2D Temporal Convolutional Network (2D-TCN) within the TimesNet framework to enhance the model's ability to interpret temporal causality in time-series forecasting. Our research compares this novel approach against traditional machine learning methods like ARIMA and state-of-the-art deep learning models, including LSTM and ETSFormer, across various long-term and short-term forecasting tasks. Contrary to our initial hypothesis, the results reveal that the causal 2D-TCN integration does not significantly outperform the original TimesNet model and is sometimes surpassed by the traditional ARIMA method. This study highlights the complexities of embedding temporal causality in deep learning models and underscores the continuing relevance of traditional methods in certain forecasting scenarios.

## Usage

1. Install Python 3.10.6. For convenience, execute the following command.

```
pip install -r requirements.txt
```

2. Prepare Data. You can obtain the well-pre-processed datasets from [[Google Drive]](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2?usp=sharing) or [[Baidu Drive]](https://pan.baidu.com/s/1r3KhGd0Q9PJIUZdfEYoymg?pwd=i9iy), Then place the downloaded data in the folder`./dataset`. Here is a summary of supported datasets.

<p align="center">
<img src=".\pic\dataset.png" height = "200" alt="" align=center />
</p>

3. Train and evaluate model. We provide the experiment scripts for all benchmarks under the folder `./scripts/`. Specify the mode using the `--mode` flag (regular, causal, acausal). You can reproduce the experiment results as the following examples:

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

Here, and by using the mode option for our arguments, we can force the TimesNet model to use the appropriate causality mode. We distinguish between using a causal TimesNet model and one that makes no use of causal structure.


