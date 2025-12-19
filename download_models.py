"""
Model Downloader for Age and Gender Detector

This script downloads the required FairFace model from a public source.
Run this once before using the main application.
"""

import os
import sys
import urllib.request
import hashlib

MODEL_INFO = {
    'filename': 'res34_fair_align_multi_7_20190809.pt',
    'url': 'https://drive.google.com/uc?export=download&id=11y0Wi3YQf21a_VcspUV4FwqzhMcfaVAB',
    'size_mb': 81.3,
    'description': 'FairFace ResNet34 model for age, gender, and race detection'
}

def download_with_progress(url, filepath, expected_size_mb=None):
    """Download a file with a progress indicator."""
    print(f"Downloading: {os.path.basename(filepath)}")
    if expected_size_mb:
        print(f"Expected size: ~{expected_size_mb} MB")
    print(f"Source: {url}")
    print()
    
    try:
        # For Google Drive, we need to handle the download differently
        # Using gdown if available, otherwise provide manual instructions
        try:
            import gdown
            gdown.download(url, filepath, quiet=False)
            return True
        except ImportError:
            print("=" * 60)
            print("MANUAL DOWNLOAD REQUIRED")
            print("=" * 60)
            print()
            print("The automatic download requires 'gdown'. You can either:")
            print()
            print("Option 1: Install gdown and re-run this script:")
            print("    pip install gdown")
            print("    python download_models.py")
            print()
            print("Option 2: Download manually:")
            print("    1. Open this URL in your browser:")
            print(f"       https://drive.google.com/file/d/11y0Wi3YQf21a_VcspUV4FwqzhMcfaVAB/view")
            print()
            print("    2. Click 'Download' button")
            print()
            print("    3. Save the file as:")
            print(f"       {os.path.abspath(filepath)}")
            print()
            print("=" * 60)
            return False
            
    except Exception as e:
        print(f"Error downloading: {e}")
        return False


def verify_model(filepath):
    """Verify the downloaded model exists and has reasonable size."""
    if not os.path.exists(filepath):
        return False
    
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    if size_mb < 50:  # Model should be at least 50MB
        print(f"Warning: Downloaded file seems too small ({size_mb:.1f} MB)")
        return False
    
    print(f"Model verified: {filepath} ({size_mb:.1f} MB)")
    return True


def main():
    print("=" * 60)
    print("Age and Gender Detector - Model Downloader")
    print("=" * 60)
    print()
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, MODEL_INFO['filename'])
    
    # Check if model already exists
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"Model already exists: {model_path}")
        print(f"Size: {size_mb:.1f} MB")
        print()
        
        response = input("Re-download? (y/N): ").strip().lower()
        if response != 'y':
            print("Skipping download. Model is ready to use!")
            return
        print()
    
    # Download the model
    success = download_with_progress(
        MODEL_INFO['url'],
        model_path,
        MODEL_INFO['size_mb']
    )
    
    if success and verify_model(model_path):
        print()
        print("=" * 60)
        print("SUCCESS! Model downloaded and ready.")
        print("You can now run: python main.py")
        print("=" * 60)
    elif not success:
        print()
        print("Download incomplete. Please follow the manual instructions above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
