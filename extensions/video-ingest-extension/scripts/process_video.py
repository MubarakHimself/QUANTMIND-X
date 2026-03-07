import os
import sys
import subprocess
import argparse
import json
import shutil
from pathlib import Path

def check_dependencies():
    """Verify yt-dlp and ffmpeg are installed."""
    deps = ['yt-dlp', 'ffmpeg']
    missing = []
    for dep in deps:
        if shutil.which(dep) is None:
            missing.append(dep)
    
    if missing:
        print(f"Error: Missing dependencies: {', '.join(missing)}")
        print("Please install them via pip (yt-dlp) or system package manager (ffmpeg).")
        sys.exit(1)

def run_command(cmd, cwd=None):
    """Run a shell command and handle errors."""
    try:
        subprocess.run(cmd, check=True, shell=True, cwd=cwd, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {cmd}")
        print(f"Stderr: {e.stderr.decode()}")
        sys.exit(1)

def preprocess_video(url, output_dir, is_test=False):
    """Download video and extract artifacts."""
    
    # 1. Setup Directories
    video_dir = Path(output_dir)
    video_dir.mkdir(parents=True, exist_ok=True)
    
    # Clear previous run if exists to avoid stale data mixing
    for item in video_dir.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()

    print(f"Processing URL: {url}")
    print(f"Output Directory: {video_dir}")

    # 2. Download Video (yt-dlp)
    # Filename template: video.mp4
    video_path = video_dir / "video.mp4"
    audio_path = video_dir / "audio.mp3"
    frames_dir = video_dir / "frames"
    frames_dir.mkdir(exist_ok=True)

    print("Step 1: Downloading Video...")
    dl_cmd = f"yt-dlp -f 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best' -o '{video_path}' --force-overwrites"
    
    if is_test:
        # Download only first 1 minute for testing
        dl_cmd += " --download-sections '*00:00-01:00'"
    
    dl_cmd += f" \"{url}\""
    run_command(dl_cmd)

    if not video_path.exists():
        print("Error: Video download failed.")
        sys.exit(1)

    # 3. Extract Audio (ffmpeg)
    print("Step 2: Extracting Audio...")
    # Extract audio to mp3 (fast)
    audio_cmd = f"ffmpeg -i '{video_path}' -q:a 0 -map a '{audio_path}' -y"
    run_command(audio_cmd)

    # 4. Extract Frames (ffmpeg)
    print("Step 3: Extracting Frames (1 frame every 30s)...")
    # Extract 1 frame every 30 seconds
    # filename pattern: frame_001.jpg
    frame_cmd = f"ffmpeg -i '{video_path}' -vf fps=1/30 -q:v 2 '{frames_dir}/frame_%03d.jpg' -y"
    run_command(frame_cmd)

    # 5. Generate Manifest
    # Create a JSON list of all generated files for the consumer to read
    manifest = {
        "video_path": str(video_path.absolute()),
        "audio_path": str(audio_path.absolute()),
        "frames": [str(f.absolute()) for f in sorted(frames_dir.glob("*.jpg"))]
    }
    
    output_json = video_dir / "manifest.json"
    with open(output_json, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Pre-processing complete. Manifest saved to {output_json}")
    # Print manifest path to stdout for the caller (CLI/Server) to pick up
    print(f"MANIFEST_PATH={output_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NPRD Video Pre-processor")
    parser.add_argument("--url", required=True, help="YouTube URL to process")
    parser.add_argument("--dir", default="./tmp/nprd_data", help="Directory to save outputs")
    parser.add_argument("--test", action="store_true", help="Run in test mode (download partial video)")
    
    args = parser.parse_args()
    
    check_dependencies()
    preprocess_video(args.url, args.dir, args.test)
