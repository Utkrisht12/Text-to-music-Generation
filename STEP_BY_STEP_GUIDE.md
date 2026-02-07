# Step-by-Step Guide: Running CLAudioLM in Jupyter

## 🎯 Quick Start - Follow These Steps

### STEP 1: Navigate to Notebooks Folder
1. In Jupyter (http://localhost:8888/tree), click on **`source`** folder
2. Click on **`notebooks`** folder
3. You'll see 8 notebooks

---

## 🧪 STEP 2: Test Your Environment (Start Here!)

**Open: `Data_display.ipynb`** (This is the simplest notebook to test)

### What to do:
1. **Click on `Data_display.ipynb`** to open it
2. **Modify Cell 0** - Remove Google Colab code:
   - Find this code:
     ```python
     from google.colab import drive
     drive.mount('/content/gdrive')
     ```
   - **DELETE or COMMENT OUT** these lines (add `#` at the start)
   - Change paths from `/content/gdrive/MyDrive/IDL_Project/data` to `./data` (or your local path)
   - Modified Cell 0 should look like:
     ```python
     # from google.colab import drive
     # drive.mount('/content/gdrive')
     
     import numpy as np 
     import os
     import json 
     import pandas as pd
     
     # Update these paths to your local directories
     ROOT_DATA = "./data"  # Change this to your data path
     AUDIOSET = "./data/audioset"  # Change this to your data path
     audioset_paths = {
         "eval": os.path.join(AUDIOSET,"csvs","eval_segments.csv"),
         "eval_tagged": os.path.join(AUDIOSET,"tags","eval_tagged.csv"),
         "eval_caps": os.path.join(AUDIOSET,"captions","eval_caps.csv"),
         "music": os.path.join(AUDIOSET,"sounds","Music")
     }
     ```
3. **Run Cell 0**: Click on the cell, then press **Shift + Enter**
4. **Skip cells that download data** (Cells 5-9) - these are for downloading datasets
5. **Run remaining cells** one by one with **Shift + Enter**

**✅ If this works, your environment is set up correctly!**

---

## 🚀 STEP 3: Run Main Application - AudioLM+CLAP.ipynb

**Open: `AudioLM+CLAP.ipynb`** (Main notebook)

### Cell-by-Cell Instructions:

#### **Cell 1: Setup (MODIFY THIS!)**
- **FIND**: `from google.colab import drive` and `drive.mount('/content/gdrive')`
- **DELETE or COMMENT OUT** these lines
- **MODIFIED CODE** should be:
  ```python
  # from google.colab import drive
  # drive.mount('/content/gdrive')
  ```
- **RUN**: Shift + Enter

#### **Cell 2: Install audiolm-pytorch**
- **OPTION A**: If you want to try installing:
  - Keep the code: `!pip install audiolm-pytorch -q`
  - **RUN**: Shift + Enter
  - ⚠️ **May fail** due to fairseq dependency - that's OK, continue anyway

- **OPTION B**: Skip this cell if installation fails
  - Comment it out: `# !pip install audiolm-pytorch -q`

#### **Cell 3: Install laion-clap**
- **MODIFY**: Change `!pip install laion-clap -q` to `!pip install laion_clap -q` (underscore, not hyphen)
- **RUN**: Shift + Enter
- ✅ Should work (already installed)

#### **Cell 4: Check GPU (SKIP THIS)**
- **SKIP** or **COMMENT OUT**: `!nvidia-smi`
- This is for checking GPU in Colab, not needed locally

#### **Cell 5: Install wandb**
- **RUN**: Shift + Enter
- ✅ Should work (already installed)

#### **Cell 6: Set release path (MODIFY THIS!)**
- **FIND**: `release_path = "/content/gdrive/MyDrive/IDL_Project/data/releases/v2_1.zip"` (or similar paths)
- **CHANGE TO**: `release_path = "./data/releases/v2_1.zip"` (or your local path)
- **RUN**: Shift + Enter

#### **Continue Running Cells:**
- **Run each cell sequentially** with **Shift + Enter**
- **When you see paths like `/content/gdrive/...`**, change them to local paths like `./data/...`
- **Skip or modify** any Google Colab-specific commands

---

## 📋 STEP 4: What Each Notebook Does

### **Recommended Order:**

1. **`Data_display.ipynb`** ✅ **START HERE**
   - Tests your environment
   - Displays sample audio data
   - Good for verifying setup

2. **`AudioLM+CLAP.ipynb`** 🎯 **MAIN APPLICATION**
   - Main music generation model
   - Requires pre-trained models
   - Can generate music from text

3. **`Clap_train.ipynb`** (Optional)
   - Trains CLAP model
   - Requires dataset preparation
   - Time-consuming

4. **`CLAP_Dataset_gen.ipynb`** (Optional)
   - Prepares dataset for CLAP training
   - Converts audio formats

5. **Other notebooks** (Evaluation/Comparison)
   - `FAD_TRILL.ipynb` - Evaluation metrics
   - `FADvgg.ipynb` - Evaluation metrics
   - `Text_quality_comparison.ipynb` - Compare text quality
   - `dataset_prod_music_text.ipynb` - Dataset production

---

## 🔧 Common Modifications Needed

### **Pattern 1: Google Drive Mounting**
**FIND:**
```python
from google.colab import drive
drive.mount('/content/gdrive')
```

**REPLACE WITH:**
```python
# from google.colab import drive
# drive.mount('/content/gdrive')
```

### **Pattern 2: File Paths**
**FIND:**
```python
"/content/gdrive/MyDrive/IDL_Project/data/..." (or your local path)
```

**REPLACE WITH:**
```python
"./data/..."  # or your actual local path
```

### **Pattern 3: Shell Commands**
**FIND:**
```python
!ls "/content/gdrive/..."
```

**REPLACE WITH:**
```python
!ls "./data/..."  # or remove ! and use Python os.listdir()
```

---

## ⚠️ Important Notes

1. **Data Required**: You'll need music datasets. The notebooks expect:
   - Audio files (.wav, .flac, .mp3)
   - Text captions/descriptions
   - Pre-trained model checkpoints (downloaded automatically)

2. **GPU Recommended**: Training/inference is slow on CPU. For testing, CPU works but will be slow.

3. **Model Downloads**: Pre-trained models download automatically from HuggingFace when needed.

4. **Errors are OK**: Some cells may fail if:
   - Data files are missing
   - Models aren't downloaded yet
   - Dependencies aren't installed
   - **Just continue to next cell!**

---

## 🎯 Quick Test Checklist

- [ ] Opened Jupyter at http://localhost:8888/tree
- [ ] Navigated to `source/notebooks/`
- [ ] Opened `Data_display.ipynb`
- [ ] Removed Google Colab code from Cell 0
- [ ] Updated file paths
- [ ] Ran Cell 0 successfully
- [ ] Ready to run main notebook!

---

## 🆘 Troubleshooting

### **Import Error?**
```python
# In a new cell, try:
!pip install <package-name>
```

### **Path Not Found?**
- Check if data directory exists
- Create directories: `mkdir -p ./data/audioset`
- Update paths in notebook

### **Jupyter Not Running?**
```bash
cd /Users/petrof/Downloads/idl-project-ClaudioLM-main
source venv/bin/activate
jupyter notebook --no-browser --port=8888
```

---

## 📞 Next Steps After Setup

Once you can run `Data_display.ipynb` successfully:
1. Prepare your music dataset (or use sample data)
2. Run `AudioLM+CLAP.ipynb` for music generation
3. Modify text prompts to generate different music!

**Good luck! 🎵**
