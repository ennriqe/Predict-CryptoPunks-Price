# Predict CryptoPunks Price

Research project for estimating CryptoPunks fair value from on-chain sales history, market context, and image-derived attributes.

## TL;DR

- Dataset built from historical CryptoPunks sales, floor-price series, ETH price history, and punk metadata
- Models trained to predict ETH-denominated prices
- Main reported accuracy: **~87% (1 - MAPE)**
- Includes both a Giza-oriented ONNX/zk workflow prototype and a LightGBM notebook baseline

## Why This Project Matters

CryptoPunks are highly heterogeneous assets. A useful valuation model needs to combine:
- trait-level differences across punks
- market regime information (ETH/floor context)
- temporal effects
- reproducible feature pipelines

This repo was an early foundation for later PunkPredictor work.

## Repository Contents

- `generate_dataset.py` - builds the modeling dataset from multiple sources
- `label_skin.py` - image-processing helper for skin-label features
- `giza_build.ipynb` - ONNX/Giza-compatible modeling workflow
- `Predict CryptoPunks Prices LightGBM.ipynb` - alternative LightGBM model and analysis
- `utils/` - helper functions used by the notebooks/scripts
- Data snapshots (`sales.csv`, `cryptopunk_metadata.csv`, `cryptopunks_skin.csv`, etc.)

## Data Sources (Used in the Pipeline)

- Dune Analytics (CryptoPunks trades / floor-related queries)
- CoinGecko (ETH price history)
- OpenSea API (current floor price helper in the script workflow)
- Hugging Face CryptoPunks images dataset (for image-derived features)

## How To Run (High Level)

1. Create a Python environment and install dependencies:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
2. Prepare/download source data (Dune exports + image dataset)
3. Run `label_skin.py` (if regenerating image-derived labels)
4. Run `generate_dataset.py`
5. Open either notebook:
   - `giza_build.ipynb`
   - `Predict CryptoPunks Prices LightGBM.ipynb`

## Outputs / Artifacts in Repo

- ONNX model artifacts (`model.onnx`, `cryptopunks.onnx`)
- Notebook plots and model diagnostics
- Intermediate CSV datasets used for experimentation

## Limitations

- This repo is a research prototype, not the production PunkPredictor pipeline
- Some external APIs and Dune queries may require credentials or updated endpoints
- Large checked-in data files make cloning slower than a productionized repo layout
