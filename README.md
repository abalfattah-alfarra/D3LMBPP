# D3LMBPP: Deep Learning with Large Language Models for Bitcoin Price Prediction

This repository contains code, configuration and sample data to reproduce the results of the paper:

> **Deep Learning with Large Language Models for Bitcoin Price Prediction (D3LMBPP)**

## Quick start

```bash
# clone and install
git clone https://github.com/abalfattah-alfarra/Bitcoin.git
cd Bitcoin
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# download full raw dataset (prices, on‑chain, tweets)
python src/data_download.py

# preprocess → train → evaluate
python src/preprocess.py
python src/train_lstm.py --config configs/default.yaml
python src/eval.py --checkpoint saved_models/best_model_LSTM.h5
```

Figures (loss curve, residuals, attention heatmap, confusion matrix) are saved to `figures/`.

## Citation
Please cite our paper if you use this code:

```
@article{alfarra2025d3lmbpp,
  title={Deep Learning with Large Language Models for Bitcoin Price Prediction},
  author={Alfarra, Abdalfattah M. and El-Farra, Eyad J. and Firwana, Iyad N. and AbuSamra, Aiman A.},
  journal={International Journal of Information Technology and Computer Science},
  year={2025},
  volume={14},
  number={6},
  pages={1--4},
  doi={10.5815/ijitcs.2025.06.01}
}
```

## License
Released under the MIT License (see `LICENSE`).