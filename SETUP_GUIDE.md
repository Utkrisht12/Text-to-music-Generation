# CLAudioLM Setup and Running Guide

## Overview
This project implements **CLAudioLM: Text-Conditioned Music Generation** using Jupyter notebooks. The notebooks were originally designed for Google Colab but can be run locally.

## Prerequisites
- Python 3.8+ (Python 3.12.5 is installed)
- Virtual environment (already created)
- Jupyter Notebook (already installed)

## Step 1: Install Additional Dependencies

The notebooks require additional packages beyond `requirements.txt`. Most have been installed:

✅ **Already Installed:**
- `laion_clap` - CLAP model for audio-text embeddings
- `pydub` - Audio processing
- `webdataset` - Dataset handling
- `librosa` - Audio analysis
- `soundfile` - Audio file I/O

⚠️ **Note on audiolm-pytorch:**
The `audiolm-pytorch` package has a dependency on `fairseq` which may have build issues. You can try:

```bash
# Activate virtual environment
source venv/bin/activate

# Try installing audiolm-pytorch (may fail due to fairseq)
pip install audiolm-pytorch

# If it fails, you may need to install fairseq manually or use an alternative approach
# The notebooks may work without it depending on which features you use
```

**Alternative:** Install packages as needed when running specific notebooks. Some notebooks may work without `audiolm-pytorch` if you're only using CLAP components.

## Step 2: Access Jupyter Notebook

The Jupyter server is already running. Open your browser and go to:

```
http://localhost:8888/tree?token=f4504a1c2b99b2c47fa1ca05bb84b9c5ab460343dc735393
```

Or navigate to: `http://127.0.0.1:8888/tree?token=f4504a1c2b99b2c47fa1ca05bb84b9c5ab460343dc735393`

## Step 3: Notebook Workflow

The notebooks should be run in this order:

### 1. **Dataset Preparation** (if needed)
   - `dataset_prod_music_text.ipynb` - Generate music-text dataset
   - `CLAP_Dataset_gen.ipynb` - Prepare CLAP dataset format

### 2. **Training**
   - `Clap_train.ipynb` - Train CLAP model
   - `AudioLM+CLAP.ipynb` - Main model training and inference

### 3. **Evaluation** (optional)
   - `FAD_TRILL.ipynb` - FAD evaluation
   - `FADvgg.ipynb` - FAD evaluation with VGG
   - `Text_quality_comparison.ipynb` - Text quality comparison
   - `Data_display.ipynb` - Display dataset samples

## Step 4: Adapting Colab Code for Local Use

The notebooks contain Google Colab-specific code. You'll need to:

1. **Remove/Modify Google Drive mounting:**
   ```python
   # Remove or comment out:
   # from google.colab import drive
   # drive.mount('/content/gdrive')
   ```

2. **Update file paths:**
   - Change Colab paths like `/content/drive/MyDrive/...` to local paths
   - Example: `/content/drive/MyDrive/Project/` → `./data/` or your local path

3. **Install packages:**
   - Replace `!pip install` commands with direct pip installs (already done)
   - Or run them in notebook cells if needed

## Step 5: Running a Notebook

1. Open Jupyter in your browser (URL above)
2. Navigate to `source/notebooks/`
3. Click on a notebook (e.g., `AudioLM+CLAP.ipynb`)
4. Run cells sequentially (Shift+Enter) or run all (Cell → Run All)

## Important Notes

- **Data Requirements:** You'll need music datasets and pre-trained models
- **GPU Recommended:** Training requires significant computational resources
- **Model Checkpoints:** Pre-trained models will be downloaded automatically from HuggingFace
- **Storage:** Ensure sufficient disk space for datasets and model checkpoints

## Troubleshooting

### If Jupyter is not running:
```bash
cd /path/to/claudiolm
source venv/bin/activate
jupyter notebook --no-browser --port=8888
```

### If you get import errors:
```bash
source venv/bin/activate
pip install <missing-package>
```

### For CLAP repository:
The notebooks clone the CLAP repository. You may need to:
```bash
git clone https://github.com/LAION-AI/CLAP.git
# Then install it:
cd CLAP
pip install -e .
```

## Quick Start Example

1. Open `AudioLM+CLAP.ipynb`
2. Modify the first cell to remove Colab-specific code
3. Update file paths to your local directories
4. Run cells sequentially

## Additional Resources

- [AudioLM PyTorch](https://github.com/lucidrains/audiolm-pytorch)
- [CLAP Repository](https://github.com/LAION-AI/CLAP)
- [MusicLM PyTorch](https://github.com/lucidrains/musiclm-pytorch)
