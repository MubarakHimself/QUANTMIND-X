"""
Markdown Generator - Creates human-readable summary from timeline
"""
import json
from pathlib import Path
from typing import Dict, Any
from datetime import datetime


def generate_markdown(data: Dict[str, Any], output_path: Path) -> None:
    """
    Create detailed markdown summary from timeline data.
    """
    output_path = Path(output_path)
    
    meta = data.get('meta', {})
    timeline = data.get('timeline', [])
    stats = data.get('summary_stats', {})
    
    md = []
    
    # Header
    md.append(f"# Video Analysis: {meta.get('title', 'Unknown')}\n")
    md.append(f"**URL:** {data.get('source_url', 'N/A')}  ")
    md.append(f"**Duration:** {meta.get('total_duration', 'N/A')}  ")
    md.append(f"**Processed:** {datetime.now().strftime('%Y-%m-%d')}  ")
    md.append(f"**Model:** {meta.get('model_used', 'gemini-1.5-pro')}\n")
    
    # Quick Stats
    md.append("## Quick Stats\n")
    md.append("| Metric | Value |")
    md.append("|--------|-------|")
    md.append(f"| Total Clips | {stats.get('total_clips', len(timeline))} |")
    md.append(f"| Chart Analysis | {stats.get('chart_clips', 0)} clips |")
    md.append(f"| Slides | {stats.get('slide_clips', 0)} clips |")
    md.append("")
    
    # Speakers
    speakers = meta.get('speakers', {})
    if speakers:
        md.append("## Speakers\n")
        for speaker_id, info in speakers.items():
            if isinstance(info, dict):
                role = info.get('role', 'Unknown')
                time_str = info.get('speaking_time', '')
                md.append(f"- **{speaker_id}:** {role} ({time_str})")
            else:
                md.append(f"- **{speaker_id}:** {info}")
        md.append("")
    
    md.append("---\n")
    md.append("## Timeline\n")
    
    # Timeline entries
    for clip in timeline:
        clip_id = clip.get('clip_id', '?')
        start = clip.get('timestamp_start', '00:00')
        end = clip.get('timestamp_end', '00:00')
        speaker = clip.get('speaker', 'UNKNOWN')
        
        visual = clip.get('visual', {})
        scene_type = visual.get('scene_type', 'discussion')
        scene_emoji = {
            'intro': '🎬',
            'chart_analysis': '📊',
            'slide': '📋',
            'discussion': '💬',
            'outro': '🎬'
        }.get(scene_type, '💬')
        
        md.append(f"### [{start} - {end}] {speaker} | {scene_emoji} {scene_type.replace('_', ' ').title()}\n")
        
        # Transcript
        transcript = clip.get('transcript', '')
        if transcript:
            md.append(f"> \"{transcript[:300]}{'...' if len(transcript) > 300 else ''}\"\n")
        
        # Visual description
        if isinstance(visual, dict):
            desc = visual.get('description', '')
            if desc:
                md.append(f"**Visual:** {desc}\n")
        
        # OCR content
        ocr = clip.get('ocr_content', [])
        if ocr:
            md.append("**On Screen:**")
            for text in ocr[:5]:
                md.append(f"- {text}")
            md.append("")
        
        # Keywords
        keywords = clip.get('keywords', [])
        if keywords:
            md.append(f"**Keywords:** {', '.join(keywords[:10])}\n")
        
        md.append("---\n")
    
    output_path.write_text('\n'.join(md), encoding='utf-8')


def combine_chunk_results(chunk_files: list, output_path: Path) -> Dict[str, Any]:
    """Combine multiple chunk JSONs into one timeline."""
    combined = {
        'video_id': '',
        'source_url': '',
        'meta': {},
        'timeline': [],
        'summary_stats': {'total_clips': 0, 'chart_clips': 0, 'slide_clips': 0}
    }
    
    for chunk_file in sorted(chunk_files):
        try:
            raw = json.loads(Path(chunk_file).read_text())
            
            # Handle case where data is a list instead of dict
            if isinstance(raw, list):
                # Wrap it in a timeline structure
                data = {'timeline': raw, 'meta': {}, 'summary_stats': {}}
            elif not isinstance(raw, dict):
                # Skip malformed data
                print(f"⚠️ Skipping malformed chunk: {chunk_file}")
                continue
            else:
                data = raw
            
            if not combined['video_id']:
                combined['video_id'] = data.get('video_id', '')
                combined['source_url'] = data.get('source_url', '')
                combined['meta'] = data.get('meta', {})
            
            # Safely get timeline
            timeline = data.get('timeline', [])
            if isinstance(timeline, list):
                combined['timeline'].extend(timeline)
            
            stats = data.get('summary_stats', {})
            if isinstance(stats, dict):
                combined['summary_stats']['total_clips'] += stats.get('total_clips', 0)
                combined['summary_stats']['chart_clips'] += stats.get('chart_clips', 0)
                combined['summary_stats']['slide_clips'] += stats.get('slide_clips', 0)
                
        except Exception as e:
            print(f"⚠️ Error reading chunk {chunk_file}: {e}")
            continue
    
    # Re-number clips
    for i, clip in enumerate(combined['timeline'], 1):
        if isinstance(clip, dict):
            clip['clip_id'] = i
    
    output_path.write_text(json.dumps(combined, indent=2, ensure_ascii=False))
    return combined

