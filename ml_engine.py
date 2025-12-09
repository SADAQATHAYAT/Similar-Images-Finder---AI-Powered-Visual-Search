import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image as kimage
from tensorflow.keras.models import Model
import faiss

# --- Model and Feature Extraction ---

class FeatureExtractor:
    def __init__(self):
        print("Loading ResNet50 model...")
        # Load ResNet50 pretrained on ImageNet, without the classification layer (include_top=False)
        # pooling='avg' means the output is a single 2048-dimensional vector (the feature vector)
        base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        self.model = Model(inputs=base_model.input, outputs=base_model.output)
        # Warmup the model to improve the speed of the very first prediction
        self.model.predict(np.zeros((1, 224, 224, 3)))
        print("Model loaded successfully.")

    def extract_features(self, img_path):
        """Returns a normalized 2048-d feature vector for an image."""
        try:
            # Load and resize image to 224x224 (ResNet's required input size)
            img = kimage.load_img(img_path, target_size=(224, 224))
            x = kimage.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            
            features = self.model.predict(x)
            
            # Normalize for Cosine Similarity: L2 normalization is necessary for IndexFlatIP (Inner Product)
            # to behave like Cosine Similarity (A . B / ||A|| ||B||) -> A . B if ||A||=||B||=1
            features = features / np.linalg.norm(features, axis=1, keepdims=True)
            return features[0].astype('float32')
        except Exception as e:
            print(f"Error extracting features from {img_path}: {e}")
            return None

# --- FAISS Search Index ---

class SearchIndex:
    def __init__(self, index_file="faiss_index.bin", mapping_file="file_mapping.pkl"):
        self.dimension = 2048
        self.index_file = index_file
        self.mapping_file = mapping_file
        # file_mapping links the FAISS internal ID to the actual filename
        self.file_mapping = {} 
        
        # Check if index files exist to load existing data or create new one
        if os.path.exists(index_file) and os.path.exists(mapping_file):
            print("Loading existing FAISS index...")
            self.index = faiss.read_index(index_file)
            with open(mapping_file, 'rb') as f:
                self.file_mapping = pickle.load(f)
        else:
            print("Creating new FAISS index (IndexFlatIP for Cosine Similarity)...")
            # IndexFlatIP = Index Flat Inner Product
            self.index = faiss.IndexFlatIP(self.dimension)

    def add_image(self, filename, vector):
        """Adds a feature vector to the FAISS index and saves the index."""
        if vector is None: return
        
        # FAISS expects an input as a matrix (N, D), so expand dimensions
        vector_batch = np.expand_dims(vector, axis=0)
        self.index.add(vector_batch)
        
        # Store the mapping
        internal_id = self.index.ntotal - 1
        self.file_mapping[internal_id] = filename
        
        self.save()

    def search(self, vector, k=5):
        """Searches the FAISS index for the k nearest neighbors to the input vector."""
        if self.index.ntotal == 0:
            return []
        
        vector_batch = np.expand_dims(vector, axis=0)
        # D = distances/scores (similarity score, 1.0 = perfect match), I = Indices (internal IDs)
        D, I = self.index.search(vector_batch, k)
        
        results = []
        for score, idx in zip(D[0], I[0]):
            # Check if index is valid (-1 means no result, or result is outside mapping)
            if idx != -1 and idx in self.file_mapping:
                results.append({
                    "filename": self.file_mapping[idx],
                    "score": float(score)
                })
        return results

    def save(self):
        """Writes the current FAISS index and mapping dictionary to disk."""
        faiss.write_index(self.index, self.index_file)
        with open(self.mapping_file, 'wb') as f:
            pickle.dump(self.file_mapping, f)
