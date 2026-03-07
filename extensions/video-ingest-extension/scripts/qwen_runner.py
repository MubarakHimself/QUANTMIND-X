import os
import sys
import json
import argparse
import base64
from openai import OpenAI

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def analyze_frames(manifest_path, prompt_text):
    # 1. Load Manifest
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    frames = manifest.get('frames', [])
    if not frames:
        print("Error: No frames found in manifest.")
        sys.exit(1)
        
    # 2. Select Frames (Limit to 8 for context/cost)
    # Uniformly sample if many frames
    step = max(1, len(frames) // 8)
    selected_frames = frames[::step][:8]
    
    print(f"Selected {len(selected_frames)} frames for analysis.")

    # 3. Prepare Message for Qwen-VL
    # OpenAI Compatible format for Vision
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text}
            ]
        }
    ]
    
    for frame_path in selected_frames:
        if os.path.exists(frame_path):
            base64_img = encode_image(frame_path)
            messages[0]["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_img}"
                }
            })
    
    # 4. Call API
    api_key = os.environ.get("QWEN_API_KEY")
    base_url = os.environ.get("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    
    if not api_key:
        print("Error: QWEN_API_KEY not found in environment.")
        sys.exit(1)

    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )
    
    try:
        completion = client.chat.completions.create(
            model="qwen-vl-max", # Or qwen-vl-plus depending on tier
            messages=messages,
            temperature=0.1
        )
        
        # Output Result (JSON) to stdout
        print(completion.choices[0].message.content)
        
    except Exception as e:
        print(f"API Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--prompt", required=True)
    args = parser.parse_args()
    
    analyze_frames(args.manifest, args.prompt)
