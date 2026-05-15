import os
import time
import requests
from tqdm import tqdm
import argparse

def download_faces(output_dir, num_faces=1500, delay=1.0):
    """
    Downloads random synthetic faces from thispersondoesnotexist.com.
    
    Args:
        output_dir (str): Directory to save the downloaded faces.
        num_faces (int): Number of faces to download.
        delay (float): Delay in seconds between requests to avoid rate limiting.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the current number of files to avoid overwriting existing faces
    existing_files = [f for f in os.listdir(output_dir) if f.endswith('.jpg')]
    start_idx = len(existing_files)
    
    print(f"Found {start_idx} existing faces in {output_dir}.")
    print(f"Starting download of {num_faces} new faces...")
    
    url = "https://thispersondoesnotexist.com/"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    successful_downloads = 0
    attempts = 0
    max_attempts = num_faces * 2
    
    pbar = tqdm(total=num_faces, desc="Downloading Faces")
    
    while successful_downloads < num_faces and attempts < max_attempts:
        attempts += 1
        try:
            # We add a timestamp to the URL to bypass caching
            req_url = f"{url}?time={time.time()}"
            response = requests.get(req_url, headers=headers, timeout=10)
            
            if response.status_code == 200 and 'image/jpeg' in response.headers.get('Content-Type', ''):
                file_idx = start_idx + successful_downloads
                filename = f"tpdne_{file_idx:05d}.jpg"
                filepath = os.path.join(output_dir, filename)
                
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                
                successful_downloads += 1
                pbar.update(1)
                
                # Small delay to be polite to the server
                time.sleep(delay)
            else:
                # If we didn't get a 200 or an image, back off a bit
                time.sleep(delay * 2)
                
        except Exception as e:
            # Network error, back off
            time.sleep(delay * 3)
            
    pbar.close()
    print(f"\nDownload complete! Successfully downloaded {successful_downloads} faces.")
    print(f"Faces are saved in: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download faces from thispersondoesnotexist.com")
    parser.add_argument("--count", type=int, default=1500, help="Number of faces to download")
    parser.add_argument("--delay", type=float, default=1.5, help="Delay between requests (seconds)")
    args = parser.parse_args()
    
    # Path to the faces directory based on project structure
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    faces_dir = os.path.join(base_dir, "data", "faces")
    
    download_faces(faces_dir, num_faces=args.count, delay=args.delay)
