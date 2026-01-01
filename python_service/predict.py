import os
import sys
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# --- Model Architecture ---
class FeatureToGraphConverter:
    def __init__(self, k_neighbors=5):
        self.k_neighbors = k_neighbors

    def features_to_graph(self, features, label=None):
        features = features.T
        edge_index = self._create_knn_graph(features)
        x = torch.FloatTensor(features)
        edge_index = torch.LongTensor(edge_index)
        if label is not None:
            data = Data(x=x, edge_index=edge_index, y=torch.LongTensor([label]))
        else:
            data = Data(x=x, edge_index=edge_index)
        return data

    def _create_knn_graph(self, features):
        n_nodes = features.shape[0]
        if n_nodes <= 1:
            return np.array([[0], [0]])
        distances = np.linalg.norm(features[:, np.newaxis] - features[np.newaxis, :], axis=2)
        edges = []
        for i in range(n_nodes):
            k_actual = min(self.k_neighbors, n_nodes - 1)
            neighbors = np.argsort(distances[i])[1:k_actual + 1]
            for neighbor in neighbors:
                edges.append([i, neighbor])
                edges.append([neighbor, i])
        if edges:
            edges = list(set([tuple(edge) for edge in edges]))
            edges = np.array(edges).T
        else:
            edges = np.array([[0], [0]])
        return edges

class MultiClassRespiratoryGNN(nn.Module):
    def __init__(self, input_dim=76, hidden_dim=128, num_classes=5, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.conv4 = GCNConv(hidden_dim, hidden_dim // 2)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.bn4 = nn.BatchNorm1d(hidden_dim // 2)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout * 0.7),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(dropout * 0.5),
            nn.Linear(32, num_classes)
        )

    def forward(self, x, edge_index, batch):
        x1 = F.relu(self.bn1(self.conv1(x, edge_index)))
        x1 = self.dropout(x1)
        x2 = F.relu(self.bn2(self.conv2(x1, edge_index)))
        x2 = self.dropout(x2) + x1
        x3 = F.relu(self.bn3(self.conv3(x2, edge_index)))
        x3 = self.dropout(x3) + x2
        x4 = F.relu(self.bn4(self.conv4(x3, edge_index)))
        x = global_mean_pool(x4, batch)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)

# --- Global State ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None
label_encoder = None
graph_converter = None

def load_model():
    global model, label_encoder, graph_converter
    
    try:
        model_folder = Path(r"C:\Users\admin\Desktop\pro\saved_models\respiratory_model_20250904_002134")
        
        if not model_folder.exists():
            print(f"❌ Model folder not found at: {model_folder}")
            return False
            
        # Load metadata
        metadata_path = model_folder / 'metadata.json'
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        # Load label encoder
        label_path = model_folder / 'label_encoder.pkl'
        with open(label_path, 'rb') as f:
            label_encoder = pickle.load(f)
            
        # Initialize converter
        graph_converter = FeatureToGraphConverter()
        
        # Initialize model
        model_config = metadata['model_config']
        model = MultiClassRespiratoryGNN(**model_config)
        
        # Load weights
        model_path_file = model_folder / 'model.pth'
        checkpoint = torch.load(model_path_file, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        print(f"✅ Model loaded successfully")
        print(f"📊 Classes: {label_encoder.classes_}")
        print(f"🖥️  Device: {device}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def extract_features(audio, sr):
    try:
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)
        rms = librosa.feature.rms(y=audio)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=13)
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        features = np.vstack([
            mfcc,
            mfcc_delta,
            mfcc_delta2,
            spectral_centroid,
            spectral_rolloff,
            spectral_contrast,
            spectral_bandwidth,
            zero_crossing_rate,
            chroma,
            mel_spec,
            rms
        ])
        
        if features.shape[0] != 76:
            print(f"Warning: Expected 76 features, got {features.shape[0]}")
            
        return features
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'device': str(device)
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400
        
        # Save temporarily
        temp_path = os.path.join('/tmp', file.filename)
        # Ensure /tmp exists (on Windows usually C:\tmp or similar, but let's use current dir for safety if /tmp is issue)
        # Better to use tempfile module but sticking to simple path for now, adjusting for windows
        if os.name == 'nt':
            temp_path = os.path.join(os.environ.get('TEMP', '.'), file.filename)
            
        file.save(temp_path)
        
        try:
            # Load and process audio
            target_sr = 16000
            duration = 5.0
            target_samples = int(duration * target_sr)
            
            audio, sr = librosa.load(temp_path, sr=target_sr)
            
            if len(audio) > target_samples:
                start = (len(audio) - target_samples) // 2
                audio = audio[start:start + target_samples]
            elif len(audio) < target_samples:
                padding = target_samples - len(audio)
                audio = np.pad(audio, (0, padding), mode='constant')
                
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))
            
            # Extract features
            features = extract_features(audio, sr)
            if features is None:
                return jsonify({'error': 'Feature extraction failed'}), 500
            
            # Convert to graph
            graph = graph_converter.features_to_graph(features)
            graph = graph.to(device)
            
            # Predict
            with torch.no_grad():
                batch = torch.zeros(graph.x.shape[0], dtype=torch.long, device=device)
                output = model(graph.x, graph.edge_index, batch)
                probabilities = torch.exp(output[0])
                predicted_idx = output.argmax(dim=1).item()
                confidence = probabilities.max().item()
                predicted_class = label_encoder.inverse_transform([predicted_idx])[0]
            
            # Prepare response
            all_probabilities = {
                str(cls): float(probabilities[i])
                for i, cls in enumerate(label_encoder.classes_)
            }
            
            result = {
                'predicted_class': str(predicted_class),
                'confidence': float(confidence),
                'all_probabilities': all_probabilities
            }
            
            return jsonify(result)
            
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("="*60)
    print("🚀 Starting Python ML Service")
    print("="*60)
    
    if load_model():
        app.run(host='0.0.0.0', port=5001, debug=False)
    else:
        print("❌ Failed to start service - model not loaded")
        sys.exit(1)
