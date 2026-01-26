"""
Download audio samples from VITS demo page for training
"""
import os
import requests
from bs4 import BeautifulSoup
from pathlib import Path
import time
from urllib.parse import urljoin

def download_audio_samples(output_dir="data/raw", test_dir="data/test", test_split=0.2):
    """
    Download audio samples from VITS demo page
    
    Args:
        output_dir: Directory to save training audio files
        test_dir: Directory to save test audio files
        test_split: Fraction of samples to use for testing (default 0.2 = 20%)
    """
    url = "https://jaywalnut310.github.io/vits-demo/index.html"
    
    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    print(f"Fetching audio samples from {url}...")
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all audio tags
        audio_tags = soup.find_all('audio')
        audio_urls = []
        
        for audio in audio_tags:
            src = audio.get('src')
            if src:
                # Make absolute URL
                absolute_url = urljoin(url, src)
                audio_urls.append(absolute_url)
        
        print(f"Found {len(audio_urls)} audio samples")
        
        if not audio_urls:
            print("No audio samples found. Let me try finding source tags...")
            source_tags = soup.find_all('source')
            for source in source_tags:
                src = source.get('src')
                if src and src.endswith('.wav'):
                    absolute_url = urljoin(url, src)
                    audio_urls.append(absolute_url)
        
        print(f"Total audio URLs: {len(audio_urls)}")
        
        # Calculate split
        num_test = int(len(audio_urls) * test_split)
        num_train = len(audio_urls) - num_test
        
        print(f"\nDownloading {num_train} training samples and {num_test} test samples...")
        
        downloaded_train = 0
        downloaded_test = 0
        
        for idx, audio_url in enumerate(audio_urls):
            try:
                # Determine if this goes to train or test
                is_test = idx < num_test
                target_dir = test_dir if is_test else output_dir
                
                # Generate filename
                filename = f"sample_{idx:04d}.wav"
                filepath = os.path.join(target_dir, filename)
                
                # Skip if already exists
                if os.path.exists(filepath):
                    print(f"  [{idx+1}/{len(audio_urls)}] Already exists: {filename}")
                    if is_test:
                        downloaded_test += 1
                    else:
                        downloaded_train += 1
                    continue
                
                # Download
                print(f"  [{idx+1}/{len(audio_urls)}] Downloading: {audio_url}")
                audio_response = requests.get(audio_url, timeout=30)
                audio_response.raise_for_status()
                
                # Save
                with open(filepath, 'wb') as f:
                    f.write(audio_response.content)
                
                if is_test:
                    downloaded_test += 1
                else:
                    downloaded_train += 1
                
                # Be polite - don't hammer the server
                time.sleep(0.5)
                
            except Exception as e:
                print(f"  Error downloading {audio_url}: {e}")
                continue
        
        print(f"\n✅ Download complete!")
        print(f"   Training samples: {downloaded_train} in {output_dir}")
        print(f"   Test samples: {downloaded_test} in {test_dir}")
        
        return downloaded_train, downloaded_test
        
    except Exception as e:
        print(f"❌ Error fetching page: {e}")
        print("\n💡 Alternative: Manual download")
        print("   1. Visit https://jaywalnut310.github.io/vits-demo/index.html")
        print("   2. Right-click on audio players → Save audio as...")
        print(f"   3. Save ~15-20 files to {output_dir}")
        print(f"   4. Save ~3-5 files to {test_dir}")
        return 0, 0


if __name__ == "__main__":
    print("=" * 60)
    print("VITS Audio Sample Downloader")
    print("=" * 60)
    print()
    
    train_count, test_count = download_audio_samples()
    
    if train_count == 0:
        print("\n" + "=" * 60)
        print("⚠️  AUTOMATED DOWNLOAD FAILED")
        print("=" * 60)
        print("\nPLAN B - Quick Manual Download:")
        print()
        print("1. Open: https://jaywalnut310.github.io/vits-demo/index.html")
        print()
        print("2. In each section (Single Speaker, Multi-Speaker):")
        print("   - Right-click each audio player")
        print("   - Select 'Save audio as...' or 'Download'")
        print("   - Save as .wav files")
        print()
        print("3. Organize files:")
        print("   - Put 15-20 files in: data/raw/")
        print("   - Put 3-5 files in: data/test/")
        print()
        print("💡 You need at least 10 training samples to start!")
        print()
        print("Alternatively, use any clean speech dataset you have!")
