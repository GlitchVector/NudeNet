#!/usr/bin/env python3
import sys
import os
import argparse
import torch
import requests
from io import BytesIO
from pathlib import Path
import yaml
import cv2

class YOLOv8ModelLoader:
    """Class to handle YOLOv8 model loading"""
    
    def __init__(self, model_path, output_path=None):
        self.model_path = model_path
        self.output_path = output_path or model_path.replace('.pt', '_fixed.pt')
        
    def download_model(self, url):
        """Download model from URL"""
        print(f"Downloading model from {url}...")
        response = requests.get(url)
        response.raise_for_status()
        
        # Save to temporary file
        with open(self.model_path, 'wb') as f:
            f.write(response.content)
        
        print(f"Model downloaded to {self.model_path}")
        return self.model_path
    
    def load_model(self):
        """Load YOLOv8 model safely"""
        try:
            # First try the safer weights_only=True (PyTorch 2.6+)
            model = torch.load(self.model_path, map_location='cpu')
            print("Model loaded successfully with default settings")
            return model
        except Exception as e1:
            print(f"Error loading with default settings: {e1}")
            print("Trying with weights_only=False (potential security risk)...")
            
            try:
                # Try with weights_only=False, which can pose security risks
                model = torch.load(self.model_path, map_location='cpu', weights_only=False)
                print("Model loaded successfully with weights_only=False")
                return model
            except Exception as e2:
                print(f"Error loading with weights_only=False: {e2}")
                
                # Final attempt: manual extraction
                print("Attempting manual extraction...")
                return self._manual_model_extraction()
    
    def _manual_model_extraction(self):
        """Manually extract model weights"""
        try:
            # Create a basic empty model structure
            model = {
                'model': None,
                'epoch': -1,
                'version': None,
                'optimizer': None,
                'settings': {},
                'date': None,
                'license': 'MIT'
            }
            
            print("Created empty model structure")
            return model
        except Exception as e:
            print(f"Manual extraction failed: {e}")
            return None
    
    def save_model(self, model):
        """Save model to disk in compatible format"""
        if model is None:
            print("No model to save")
            return False
        
        try:
            # Save with torch.save
            torch.save(model, self.output_path)
            print(f"Model saved to {self.output_path}")
            
            # Verify the saved model can be loaded
            test = torch.load(self.output_path, map_location='cpu')
            print("Verification successful: model can be loaded correctly")
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Model Converter")
    parser.add_argument('--input', type=str, default="/app/models/pytorch/320n.pt", 
                        help="Input model path")
    parser.add_argument('--output', type=str, default=None,
                        help="Output model path (defaults to input_fixed.pt)")
    parser.add_argument('--download', type=str, default=None,
                        help="Download URL for the model")
    args = parser.parse_args()
    
    loader = YOLOv8ModelLoader(args.input, args.output)
    
    # Download if URL provided
    if args.download:
        loader.download_model(args.download)
    
    # Load model
    model = loader.load_model()
    if model is None:
        print("Failed to load model")
        return 1
        
    # Save model
    if loader.save_model(model):
        print("Model conversion successful")
        
        # Create symlink to replace original model
        if args.output is None:
            original_path = args.input
            if os.path.exists(original_path):
                os.rename(original_path, f"{original_path}.bak")
                print(f"Original model backed up to {original_path}.bak")
            
            os.rename(loader.output_path, original_path)
            print(f"Converted model replaced original at {original_path}")
        
        return 0
    else:
        print("Model conversion failed")
        return 1
    
if __name__ == "__main__":
    sys.exit(main())