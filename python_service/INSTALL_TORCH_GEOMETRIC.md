# Quick Installation Guide for torch-geometric

## The Issue
`torch-geometric` requires special dependencies that must be installed first.

## Solution

Since you have **PyTorch 2.2.1**, run these commands:

### Step 1: Install torch-geometric dependencies
```bash
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.2.1+cpu.html
```

### Step 2: Install torch-geometric
```bash
pip install torch-geometric
```

### Step 3: Install remaining packages
```bash
pip install librosa scikit-learn flask flask-cors python-dotenv
```

### Step 4: Verify installation
```bash
python -c "from torch_geometric.nn import GCNConv; print('Success!')"
```

---

## Alternative: Install All at Once

```bash
# Install torch-geometric with dependencies
pip install torch-scatter torch-sparse torch-geometric -f https://data.pyg.org/whl/torch-2.2.1+cpu.html

# Install other packages
pip install librosa==0.10.1 scikit-learn==1.3.2 flask==3.0.0 flask-cors==4.0.0 python-dotenv==1.0.0
```

---

## If Installation Fails

Try installing without version constraints:
```bash
pip install torch-geometric librosa scikit-learn flask flask-cors python-dotenv
```

---

## After Installation

Run the ML service:
```bash
python predict.py
```

You should see:
```
✅ Model loaded successfully
📊 Classes: Asthma, Bronchial, COPD, Healthy, Pneumonia
🖥️  Device: cpu
```
