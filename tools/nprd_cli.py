#!/usr/bin/env python3
"""
NPRD CLI - General Purpose Multimodal Indexer

Extracts unbiased, structured data from trading education videos.
Supports: Gemini 2.0 Flash, ZhipuAI GLM-4.5V

Usage:
    python nprd_cli.py                    # Interactive mode
    python nprd_cli.py VIDEO_URL          # Direct mode (Gemini)
    python nprd_cli.py VIDEO_URL --model zai  # Use ZAI
"""

import click
import json
import os
import sys
import getpass
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from nprd.video_downloader import download_video, get_video_info
from nprd.video_splitter import split_video, get_video_duration, format_duration
from nprd.playlist_handler import is_playlist as is_playlist_url, get_playlist_videos
from nprd.prompts import LIBRARIAN_PROMPT
from nprd.retry_handler import retry_with_backoff, ChunkRecoveryManager
from nprd.markdown_generator import generate_markdown, combine_chunk_results

# ASCII Art Banner
BANNER = r"""
███╗   ██╗██████╗ ██████╗ ██████╗ 
████╗  ██║██╔══██╗██╔══██╗██╔══██╗
██╔██╗ ██║██████╔╝██████╔╝██║  ██║
██║╚██╗██║██╔═══╝ ██╔══██╗██║  ██║
██║ ╚████║██║     ██║  ██║██████╔╝
╚═╝  ╚═══╝╚═╝     ╚═╝  ╚═╝╚═════╝ 

╔═══════════════════════════════════════════════╗
║  General Purpose Multimodal Video Indexer     ║
║  Unbiased extraction • No strategy analysis  ║
║  v1.3 • Gemini 2.0 • ZAI GLM-4.5V/4.6V      ║
╚═══════════════════════════════════════════════╝
"""

CONFIG_FILE = Path(".nprd_config.json")

def load_config():
    """Load configuration from JSON file."""
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text())
        except:
            pass
    return {}

def save_config(config):
    """Save configuration to JSON file."""
    try:
        current = load_config()
        current.update(config)
        CONFIG_FILE.write_text(json.dumps(current, indent=2))
    except:
        pass

def get_analyzer(model_provider: str, api_key: str, model_name_detail: str = None):
    """Get the appropriate video analyzer based on model provider."""
    if model_provider == 'zai':
        from nprd.zai_client import ZaiVideoAnalyzer
        model = model_name_detail if model_name_detail else 'glm-4.5v'
        return ZaiVideoAnalyzer(api_key, model_name=model), model
    elif model_provider == 'openrouter':
        from nprd.openrouter_client import OpenRouterVideoAnalyzer
        model = model_name_detail if model_name_detail else 'google/gemini-2.0-flash-001'
        return OpenRouterVideoAnalyzer(api_key, model_id=model), model
    else:
        from nprd.gemini_client import GeminiVideoAnalyzer
        return GeminiVideoAnalyzer(api_key), 'gemini-2.0-flash'


def sanitize_filename(name: str) -> str:
    """Create safe filename from video title."""
    safe = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in name)
    return safe.replace(' ', '_')[:80]


def interactive_mode():
    """Run in interactive mode - prompt for model, API key and video URL."""
    print(BANNER)
    print("="*60)
    print("  INTERACTIVE MODE")
    print("="*60)
    
    config = load_config()
    saved_model = config.get('model_choice', '1')
    saved_api_key = config.get('api_key', '')
    
    # Select model provider
    print("\n🤖 Select model provider:")
    print("   1. Gemini 2.0 Flash (Google AI Studio)")
    print("   2. ZAI - GLM-4.5V")
    print("   3. ZAI - GLM-4.6V")
    print("   4. ZAI - Coding Plan")
    print("   5. OpenRouter (Multiple Models) ⭐")
    
    model_choice = input(f"   Choice [{saved_model}]: ").strip() or saved_model
    
    model_provider = 'gemini'
    model_detail = None
    env_key = 'GEMINI_API_KEY'
    provider_name = 'Gemini'
    
    if model_choice == '2':
        model_provider = 'zai'
        model_detail = 'glm-4.5v'
        env_key = 'Z_AI_API_KEY'
        provider_name = 'ZAI'
    elif model_choice == '3':
        model_provider = 'zai'
        model_detail = 'glm-4.6v'
        env_key = 'Z_AI_API_KEY'
        provider_name = 'ZAI'
    elif model_choice == '4':
        model_provider = 'zai'
        model_detail = 'glm-4.5v'
        env_key = 'Z_AI_API_KEY'
        provider_name = 'ZAI (Coding Plan)'
    elif model_choice == '5':
        model_provider = 'openrouter'
        env_key = 'OPENROUTER_API_KEY'
        provider_name = 'OpenRouter'
        
        # Show OpenRouter model selection
        from nprd.openrouter_client import OpenRouterVideoAnalyzer, VIDEO_MODELS
        OpenRouterVideoAnalyzer.list_models()
        
        print("\n   Enter model number (1-7) or paste full model ID:")
        model_input = input("   Model: ").strip() or "4"  # Default to Gemini Flash
        
        # Resolve model ID
        model_detail = OpenRouterVideoAnalyzer.get_model_id(model_input)
        if not model_detail:
            print(f"⚠️  Invalid model. Using default: google/gemini-2.0-flash-001")
            model_detail = "google/gemini-2.0-flash-001"
        else:
            # Find name for display
            for m in VIDEO_MODELS.values():
                if m['id'] == model_detail:
                    print(f"   ✅ Selected: {m['name']}")
                    break
            else:
                print(f"   ✅ Selected: {model_detail}")
    
    print(f"\n   Provider: {provider_name}")
    
    # API Key logic
    api_key = ''
    
    # 1. Try saved config
    if saved_api_key and config.get('provider') == model_provider:
        print(f"\n🔑 Found saved API key: {'*'*10}{saved_api_key[-4:]}")
        use_saved = input("   Use this key? (Y/n): ").strip().lower()
        if use_saved != 'n':
            api_key = saved_api_key
            
    # 2. Try environment
    if not api_key:
        env_val = os.environ.get(env_key, '')
        if env_val:
            print(f"\n🔑 Found {env_key} in environment: {'*'*10}{env_val[-4:]}")
            use_env = input("   Use this key? (Y/n): ").strip().lower()
            if use_env != 'n':
                api_key = env_val

    # 3. Prompt user
    if not api_key:
        print(f"\n🔑 Enter your {provider_name} API key:")
        api_key = getpass.getpass("   API Key: ").strip()
        
        # Save explicitly if entered manually
        if api_key:
            save = input("   Save key to config? (y/N): ").strip().lower()
            if save == 'y':
                save_config({'api_key': api_key, 'provider': model_provider, 'model_choice': model_choice})
    
    if not api_key:
        print("❌ No API key provided. Exiting.")
        return
    
    # Save selection preference anyway
    save_config({'model_choice': model_choice, 'provider': model_provider})
    
    # Get video URL
    print("\n📹 Enter YouTube video URL (or playlist URL):")
    video_url = input("   URL: ").strip()
    
    if not video_url:
        print("❌ No URL provided. Exiting.")
        return
    
    # Ask if playlist
    is_playlist_mode = False
    if "playlist" in video_url or "list=" in video_url:
        is_playlist_mode = True
        print("\n📚 Detected playlist URL")
    else:
        # Smart check - don't ask if obviously not a playlist
        if "&list=" not in video_url:
            pass 
        else:
             check_playlist = input("\n📚 Is this a playlist? (y/N): ").strip().lower()
             is_playlist_mode = check_playlist == 'y'
    
    # Output directory
    output_dir = Path("outputs/videos")
    print(f"\n📂 Output directory: {output_dir.absolute()}")
    
    # Process
    return process(
        video_url=video_url,
        api_key=api_key,
        output_dir=output_dir,
        is_playlist=is_playlist_mode,
        model_provider=model_provider,  # Pass provider string
        model_detail=model_detail,      # Pass specific model name
        max_videos=None,
        max_chunks=None,
        resume=True
    )


def process_single_video(
    url: str,
    output_dir: Path,
    analyzer,
    model_name: str,
    max_chunks: int = None,
    resume: bool = False
) -> dict:
    """Process a single video through the full pipeline."""
    
    # 1. Get video info
    print("\n📋 Getting video info...")
    try:
        info = get_video_info(url)
    except Exception as e:
        print(f"❌ Failed to get video info: {e}")
        return None
        
    title = info['title']
    duration = info['duration']
    
    print(f"   Title: {title}")
    print(f"   Duration: {format_duration(duration)}")
    
    # 2. Create output directory
    safe_title = sanitize_filename(title)
    video_dir = output_dir / safe_title
    video_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir = video_dir / "chunks"
    
    # 3. Download video
    video_file = video_dir / f"{safe_title}.mp4"
    if video_file.exists():
        print(f"⏭️  Video already downloaded: {video_file.name}")
    else:
        print("\n📥 Downloading video...")
        with tqdm(total=100, desc="   Progress", unit="%") as pbar:
            last_progress = [0]
            def update_progress(p):
                delta = int(p) - last_progress[0]
                if delta > 0:
                    pbar.update(delta)
                    last_progress[0] = int(p)
            
            try:
                downloaded = download_video(url, video_dir, progress_callback=update_progress)
                pbar.update(100 - last_progress[0])
                
                # Rename if different
                if downloaded.name != video_file.name:
                    if video_file.exists():
                        video_file.unlink()
                    downloaded.rename(video_file)
            except Exception as e:
                print(f"❌ Download failed: {e}")
                return None
    
    # 4. Split if >45min
    actual_duration = get_video_duration(video_file)
    chunk_limit_sec = 15 * 60  # 15 minutes per chunk (smaller for API limits)
    
    if actual_duration > chunk_limit_sec:
        print(f"\n🔪 Splitting video ({format_duration(actual_duration)} > 15min)...")
        chunks = split_video(video_file, chunk_duration_minutes=15, output_dir=chunks_dir)
        print(f"   Created {len(chunks)} chunks")
    else:
        chunks = [video_file]
        print(f"\n✅ Video is under 45min, no splitting needed")
    
    if max_chunks:
        chunks = chunks[:max_chunks]
        print(f"   Limited to first {max_chunks} chunks")
    
    # 5. Process each chunk
    recovery = ChunkRecoveryManager(chunks_dir if len(chunks) > 1 else video_dir)
    results = []
    
    print(f"\n🤖 Analyzing {len(chunks)} chunk(s) with {model_name}...")
    
    for i, chunk in enumerate(chunks, 1):
        print(f"\n   Chunk {i}/{len(chunks)}: {chunk.name}")
        
        # Check if already processed
        if resume and recovery.is_processed(chunk):
            print(f"   ⏭️  Already processed, loading cached result")
            results.append(recovery.load_result(chunk))
            continue
        
        try:
            # Analyze based on model type
            def analyze():
                if hasattr(analyzer, 'upload_video'):
                    # Gemini style - upload then analyze
                    video_file_obj = analyzer.upload_video(chunk, display_name=f"{safe_title}_chunk_{i}")
                    return analyzer.analyze_video(video_file_obj, LIBRARIAN_PROMPT)
                else:
                    # ZAI style - direct analysis
                    return analyzer.analyze_video(chunk, LIBRARIAN_PROMPT)
            
            result = retry_with_backoff(analyze, max_retries=3)
            
            # Add metadata
            if not isinstance(result, dict):
                result = {"raw_response": str(result), "meta": {}}
            
            result['video_id'] = info['id']
            result['source_url'] = url
            if 'meta' not in result:
                result['meta'] = {}
            result['meta']['title'] = title
            result['meta']['model_used'] = model_name
            result['meta']['chunk_index'] = i
            result['meta']['total_chunks'] = len(chunks)
            result['meta']['processed_at'] = datetime.now().isoformat()
            
            # Save immediately
            recovery.save_result(chunk, result)
            results.append(result)
            print(f"   ✅ Chunk {i} complete")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            recovery.log_failure(chunk, str(e))
    
    # 6. Combine results
    if results:
        print("\n📊 Combining results...")
        chunk_results_dir = chunks_dir if len(chunks) > 1 else video_dir
        chunk_files = list(chunk_results_dir.glob("*.json"))
        
        if chunk_files:
            final_json = video_dir / f"{safe_title}.json"
            combined = combine_chunk_results(chunk_files, final_json)
            # Ensure meta is preserved in combined
            if 'meta' not in combined: combined['meta'] = {}
            combined['meta']['model_used'] = model_name
            combined['meta']['processed_at'] = datetime.now().isoformat()
            
            final_json.write_text(json.dumps(combined, indent=2))
            print(f"   Saved: {final_json.name}")
            
            # 7. Generate markdown
            final_md = video_dir / f"{safe_title}.md"
            try:
                generate_markdown(combined, final_md)
                print(f"   Saved: {final_md.name}")
            except Exception as e:
                print(f"   ⚠️ Markdown generation failed: {e}")
            
            return combined
    
    return None


def process_playlist(
    url: str,
    output_dir: Path,
    analyzer,
    model_name: str,
    max_videos: int = None,
    max_chunks: int = None,
    resume: bool = False
) -> dict:
    """Process a YouTube playlist."""
    
    print("\n📚 Extracting playlist info...")
    try:
        playlist = get_playlist_videos(url)
    except Exception as e:
        print(f"❌ Failed to get playlist: {e}")
        return None
    
    print(f"   Playlist: {playlist['playlist_title']}")
    print(f"   Videos: {playlist['total_videos']}")
    
    videos = playlist['videos']
    if max_videos:
        videos = videos[:max_videos]
        print(f"   Limited to first {max_videos} videos")
    
    # Create playlist directory
    safe_title = sanitize_filename(playlist['playlist_title'])
    playlist_dir = output_dir / safe_title
    playlist_dir.mkdir(parents=True, exist_ok=True)
    
    # Create manifest
    manifest = {
        'playlist_id': playlist['playlist_id'],
        'playlist_title': playlist['playlist_title'],
        'total_videos': len(videos),
        'processed_at': datetime.now().isoformat(),
        'model_used': model_name,
        'videos': []
    }
    
    # Process each video
    for i, video in enumerate(videos, 1):
        print(f"\n{'='*60}")
        print(f"📹 Video {i}/{len(videos)}: {video['title'][:50]}...")
        print(f"{'='*60}")
        
        try:
            video_output = playlist_dir / f"video_{i:02d}"
            result = process_single_video(
                video['url'],
                video_output.parent,
                analyzer,
                model_name,
                max_chunks=max_chunks,
                resume=resume
            )
            
            status = 'completed' if result else 'failed'
            manifest['videos'].append({
                'video_id': video['id'],
                'title': video['title'],
                'url': video['url'],
                'status': status,
                'output_path': str(video_output.relative_to(playlist_dir)) if result else None
            })
            
        except Exception as e:
            print(f"❌ Failed: {e}")
            manifest['videos'].append({
                'video_id': video['id'],
                'title': video['title'],
                'url': video['url'],
                'status': 'failed',
                'error': str(e)
            })
    
    # Save manifest
    manifest_file = playlist_dir / "playlist.json"
    manifest_file.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"\n📁 Playlist manifest saved: {manifest_file}")
    
    return manifest


def process(video_url, api_key, output_dir, model_provider='gemini', model_detail=None, is_playlist=False, max_videos=None, max_chunks=None, resume=True):
    """Main processing function."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🔑 API Key: {'*' * 20}{api_key[-4:]}")
    print(f"📂 Output: {output_dir.absolute()}")
    
    try:
        analyzer, model_name = get_analyzer(model_provider, api_key, model_detail)
        print(f"🤖 Model: {model_name}")
        
        if is_playlist:
            result = process_playlist(
                video_url, output_dir, analyzer, model_name,
                max_videos=max_videos,
                max_chunks=max_chunks,
                resume=resume
            )
        else:
            result = process_single_video(
                video_url, output_dir, analyzer, model_name,
                max_chunks=max_chunks,
                resume=resume
            )
        
        print("\n" + "="*60)
        print("✨ PROCESSING COMPLETE")
        print("="*60)
        
        return result
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted! Use --resume to continue later.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


@click.command()
@click.argument('video_input', required=False)
@click.option('--output-dir', '-o', default='outputs/videos', help='Output directory')
@click.option('--api-key', envvar='GEMINI_API_KEY', help='API key (or set GEMINI_API_KEY / Z_AI_API_KEY)')
@click.option('--model', '-m', type=click.Choice(['gemini', 'zai']), default='gemini', help='Model provider')
@click.option('--playlist', is_flag=True, help='Treat input as playlist')
@click.option('--max-videos', type=int, help='Max videos from playlist')
@click.option('--max-chunks', type=int, help='Max chunks per video')
@click.option('--resume', is_flag=True, default=True, help='Resume interrupted processing')
@click.option('--interactive', '-i', is_flag=True, help='Run in interactive mode')
def main(video_input, output_dir, api_key, model, playlist, max_videos, max_chunks, resume, interactive):
    """
    NPRD - General Purpose Multimodal Video Indexer
    
    Extracts unbiased, structured data from trading education videos.
    
    \b
    Examples:
        nprd                                    # Interactive mode
        nprd "https://youtube.com/watch?v=xxx"  # Gemini (default)
        nprd "https://youtube.com/watch?v=xxx" --model zai  # ZAI
        nprd "https://youtube.com/playlist?list=xxx" --playlist
    """
    # If no video input, run interactive mode
    if not video_input or interactive:
        return interactive_mode()
    
    # Print banner
    print(BANNER)
    
    # Get API key based on model
    if model == 'zai':
        api_key = api_key or os.environ.get('Z_AI_API_KEY', '')
    
    # Validate API key
    if not api_key:
        provider_name = 'ZAI' if model == 'zai' else 'Gemini'
        print(f"🔑 Enter your {provider_name} API key:")
        api_key = getpass.getpass("   API Key: ").strip()
    
    if not api_key:
        print("❌ Error: No API key provided")
        sys.exit(1)
    
    return process(
        video_url=video_input,
        api_key=api_key,
        output_dir=Path(output_dir),
        model_provider=model,
        is_playlist=playlist,
        max_videos=max_videos,
        max_chunks=max_chunks,
        resume=resume
    )


if __name__ == "__main__":
    main()
