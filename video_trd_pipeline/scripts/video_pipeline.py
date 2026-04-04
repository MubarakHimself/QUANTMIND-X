#!/usr/bin/env python3
"""
video_trd_pipeline - Autonomous Video → TRD Pipeline

Workflow:
  YouTube URL / Playlist / batch file
    → Download (yt-dlp)
    → Extract: frames + audio
    → Gemini CLI → timeline_gemini.json
    → Qwen CLI  → timeline_qwen.json
    → OpenCode  → TRD.md
    → STOP (no compile, no backtest)

Author: Orchestrated by Hermes
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# ─── Paths ───────────────────────────────────────────────────────────────────

BASE_DIR      = Path(__file__).parent.parent.resolve()
VIDEO_IN      = BASE_DIR / "video_in"
OUTPUT_BASE   = BASE_DIR / "video_out"
PROMPTS_DIR   = BASE_DIR / "prompts"
CONFIG_DIR    = BASE_DIR / "config"
LOG_DIR       = BASE_DIR / "logs"

for d in [VIDEO_IN, OUTPUT_BASE, PROMPTS_DIR, CONFIG_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("video_trd_pipeline")


# ─── Config ──────────────────────────────────────────────────────────────────

class Config:
    FRAME_INTERVAL   = 30          # seconds between frame extracts
    AUDIO_BITRATE    = "128k"
    AUDIO_CHANNELS   = 1            # mono
    GEMINI_MODEL     = "gemini-2.0-flash-exp"
    QWEN_MODEL       = "qwen-vl-plus"
    MAX_CONCURRENT   = 3            # parallel video processing
    OPENCODE_MODEL   = "opencode"    # model override for opencode


config = Config()


# ─── Shell helpers ────────────────────────────────────────────────────────────

def run(cmd: list[str], cwd: Optional[Path] = None, timeout: int = 600,
        env: Optional[dict] = None) -> subprocess.CompletedProcess:
    """Run a command, log it, and return the CompletedProcess."""
    log.info(f"CMD: {' '.join(str(x) for x in cmd)}")
    merged_env = dict(os.environ)
    if env:
        merged_env.update(env)
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
            env=merged_env,
        )
        if result.stdout:
            log.debug(f"STDOUT: {result.stdout[:500]}")
        if result.stderr:
            log.warning(f"STDERR: {result.stderr[:500]}")
        result.check_returncode()
        return result
    except subprocess.CalledProcessError as e:
        log.error(f"Command failed with exit {e.returncode}: {' '.join(str(x) for x in cmd)}")
        log.error(f"STDERR: {e.stderr[:1000]}")
        raise
    except subprocess.TimeoutExpired as e:
        log.error(f"Command timed out after {timeout}s: {' '.join(str(x) for x in cmd)}")
        raise


# ─── Stage 1: Download ──────────────────────────────────────────────────────

def get_video_id(url: str) -> str:
    """Derive a safe folder name from a YouTube URL."""
    import re
    m = re.search(r'[?&]v=([^&]+)', url)
    if m:
        return m.group(1)
    m = re.search(r'youtu\.be/([^?]+)', url)
    if m:
        return m.group(1)
    # playlist or other
    safe = re.sub(r'[^a-zA-Z0-9_-]', '_', url)
    return safe[:80]


def get_playlist_urls(playlist_url: str) -> list[str]:
    """Extract all video URLs from a YouTube playlist."""
    log.info(f"Extracting playlist URLs: {playlist_url}")
    result = run(
        ["yt-dlp", "--flat-playlist", "--print", "url", playlist_url],
        timeout=120,
    )
    urls = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    log.info(f"Found {len(urls)} videos in playlist")
    return urls


def get_batch_urls(file_path: Path) -> list[str]:
    """Read URLs from a text file (one per line, # = comment)."""
    urls = []
    for line in file_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            urls.append(line)
    return urls


def download_video(url: str, output_folder: Path) -> Path:
    """Download a single video + audio using yt-dlp."""
    video_dir = output_folder / "video"
    video_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Downloading: {url}")

    # Download video (best video + best audio merged)
    run([
        "yt-dlp",
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "-o", str(video_dir / "%(id)s.%(ext)s"),
        url,
    ], timeout=600)

    # Find the downloaded file
    files = list(video_dir.glob("*.mp4"))
    if not files:
        raise FileNotFoundError(f"No mp4 found after download in {video_dir}")
    video_path = files[0]
    log.info(f"Video downloaded: {video_path} ({video_path.stat().st_size / 1024 / 1024:.1f} MB)")
    return video_path


# ─── Stage 2: Extract frames + audio ─────────────────────────────────────────

def extract_audio(video_path: Path, output_folder: Path) -> Path:
    """Extract mono audio from video as MP3."""
    audio_path = output_folder / "audio.mp3"
    if audio_path.exists():
        log.info(f"Audio already exists: {audio_path}")
        return audio_path

    run([
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",
        "-acodec", "libmp3lame",
        "-ab", config.AUDIO_BITRATE,
        "-ac", str(config.AUDIO_CHANNELS),
        str(audio_path),
    ], timeout=300)
    log.info(f"Audio extracted: {audio_path}")
    return audio_path


def extract_frames(video_path: Path, output_folder: Path) -> list[Path]:
    """Extract frames at FRAME_INTERVAL seconds."""
    frames_dir = output_folder / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Get video duration
    result = run([
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format", str(video_path),
    ], timeout=30)
    data = json.loads(result.stdout)
    duration = float(data["format"].get("duration", 0))

    log.info(f"Video duration: {duration:.0f}s — extracting frames every {config.FRAME_INTERVAL}s")

    frame_paths = []
    for t in range(0, int(duration), config.FRAME_INTERVAL):
        output_file = frames_dir / f"frame_{t:06d}.jpg"
        if output_file.exists():
            frame_paths.append(output_file)
            continue
        run([
            "ffmpeg", "-y",
            "-ss", str(t),
            "-i", str(video_path),
            "-vframes", "1",
            "-q:v", "2",
            str(output_file),
        ], timeout=30)
        frame_paths.append(output_file)

    log.info(f"Extracted {len(frame_paths)} frames")
    return frame_paths


# ─── Stage 3: Gemini CLI ─────────────────────────────────────────────────────

PROMPT_GEMINI = """\
You are an expert trading strategy analyst. Your job is to produce a {strategy_type} **Timeline Analysis Report** from this trading video.

For each segment, provide:

1. **Timestamp** (seconds)
2. **What is shown** — objective visual description (charts, indicators, price action)
3. **What is said** — key verbatim statements or concepts from narration
4. **Trading signal detected** — BUY / SELL / NEUTRAL (if any)
5. **Indicator / pattern mentioned** — e.g. RSI, MACD, support/resistance, candlestick patterns
6. **Timeframe(s) discussed** — M1, M5, M15, H1, H4, D1, etc.
7. **Asset class(es)** — Forex, Crypto, Stocks, Indices, Commodities
8. **Risk / money management notes** — lot size, SL/TP, risk:reward

Be factual and exact. Do NOT add personal commentary or interpretation.
Return your analysis as a single JSON object:
{{
  "video_url": "<url>",
  "title": "<video title>",
  "duration_seconds": <duration>,
  "segments": [
    {{
      "timestamp_start": <seconds>,
      "timestamp_end": <seconds>,
      "visual_description": "<what is shown>",
      "transcript_excerpt": "<key things said>",
      "signal": "BUY|SELL|NEUTRAL",
      "indicators": ["<indicator names>"],
      "timeframes": ["<timeframes>"],
      "asset_classes": ["<asset class>"],
      "risk_notes": "<notes on position sizing / SL / TP>"
    }}
  ],
  "summary": {{
    "primary_strategy_type": "<trend|mean_reversion|breakout|scalping|swing>",
    "best_timeframes": ["<timeframes>"],
    "key_indicators": ["<indicators used>"],
    "overall_bias": "bullish|bearish|neutral",
    "symbols_mentioned": ["<currency pairs, stocks, etc.>"],
    "risk_management_summary": "<summary of MM approach described>"
  }}
}}
"""


def run_gemini(video_path: Path, audio_path: Path, frames: list[Path],
               output_folder: Path, video_url: str) -> Path:
    """Run Gemini CLI to produce timeline_gemini.json."""
    out_path = output_folder / "timeline_gemini.json"
    if out_path.exists():
        log.info(f"Gemini output already exists: {out_path}")
        return out_path

    prompt_file = PROMPTS_DIR / "gemini_prompt.txt"
    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    if not prompt_file.exists():
        prompt_file.write_text(PROMPT_GEMINI)

    log.info("Running Gemini CLI analysis...")

    file_refs = "\n".join([f"Frame: {f}" for f in frames])
    file_refs += f"\nAudio: {audio_path}"
    full_prompt = PROMPT_GEMINI + "\n\n" + file_refs

    # Write combined prompt to temp file to avoid shell escaping issues
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as pf:
        pf.write(full_prompt)
        prompt_file_path = pf.name

    try:
        result = run([
            "gemini",
            "--approval-mode", "yolo",
            "--output-format", "json",
            "--model", config.GEMINI_MODEL,
            "--prompt-interactive",
            full_prompt,
        ], timeout=600)

        raw = result.stdout.strip()

        # Try to extract JSON from output (may be wrapped in markdown code blocks)
        import re
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            raw = m.group(0)

        data = json.loads(raw)
        out_path.write_text(json.dumps(data, indent=2))
        log.info(f"Gemini output written: {out_path}")
        return out_path

    finally:
        os.unlink(prompt_file_path)


# ─── Stage 4: Qwen CLI ───────────────────────────────────────────────────────

PROMPT_QWEN = """\
You are an expert trading strategy analyst. Your job is to produce a {strategy_type} **Timeline Analysis Report** from this trading video.

For each segment, provide:

1. **Timestamp** (seconds)
2. **What is shown** — objective visual description (charts, indicators, price action)
3. **What is said** — key verbatim statements or concepts from narration
4. **Trading signal detected** — BUY / SELL / NEUTRAL (if any)
5. **Indicator / pattern mentioned** — e.g. RSI, MACD, support/resistance, candlestick patterns
6. **Timeframe(s) discussed** — M1, M5, M15, H1, H4, D1, etc.
7. **Asset class(es)** — Forex, Crypto, Stocks, Indices, Commodities
8. **Risk / money management notes** — lot size, SL/TP, risk:reward

Be factual and exact. Do NOT add personal commentary or interpretation.
Return your analysis as a single JSON object:
{{
  "video_url": "<url>",
  "title": "<video title>",
  "duration_seconds": <duration>,
  "segments": [
    {{
      "timestamp_start": <seconds>,
      "timestamp_end": <seconds>,
      "visual_description": "<what is shown>",
      "transcript_excerpt": "<key things said>",
      "signal": "BUY|SELL|NEUTRAL",
      "indicators": ["<indicator names>"],
      "timeframes": ["<timeframes>"],
      "asset_classes": ["<asset class>"],
      "risk_notes": "<notes on position sizing / SL / TP>"
    }}
  ],
  "summary": {{
    "primary_strategy_type": "<trend|mean_reversion|breakout|scalping|swing>",
    "best_timeframes": ["<timeframes>"],
    "key_indicators": ["<indicators used>"],
    "overall_bias": "bullish|bearish|neutral",
    "symbols_mentioned": ["<currency pairs, stocks, etc.>"],
    "risk_management_summary": "<summary of MM approach described>"
  }}
}}
"""


def run_qwen(video_path: Path, audio_path: Path, frames: list[Path],
             output_folder: Path, video_url: str) -> Path:
    """Run Qwen CLI to produce timeline_qwen.json."""
    out_path = output_folder / "timeline_qwen.json"
    if out_path.exists():
        log.info(f"Qwen output already exists: {out_path}")
        return out_path

    log.info("Running Qwen CLI analysis...")

    file_refs = "\n".join([f"Frame: {f}" for f in frames])
    file_refs += f"\nAudio: {audio_path}"
    full_prompt = PROMPT_QWEN + "\n\n" + file_refs

    try:
        result = run([
            "qwen",
            "--approval-mode", "yolo",
            "--model", config.QWEN_MODEL,
            "-p", full_prompt,
        ], timeout=600)

        raw = result.stdout.strip()

        import re
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            raw = m.group(0)

        data = json.loads(raw)
        out_path.write_text(json.dumps(data, indent=2))
        log.info(f"Qwen output written: {out_path}")
        return out_path

    except subprocess.CalledProcessError as e:
        log.warning(f"Qwen CLI failed: {e.stderr[:300]}")
        # Create empty marker so pipeline can continue
        out_path.write_text(json.dumps({"error": "Qwen CLI failed", "details": e.stderr[:500]}))
        return out_path


# ─── Stage 5: OpenCode → TRD ─────────────────────────────────────────────────

PROMPT_OPENCODE = """\
You are a professional MQL5 expert and trading strategy engineer.

Your task: Write a complete, production-ready **Trading Requirements Document (TRD)** from this video analysis.

## Input Data

The video has been independently analyzed by TWO AI models (Gemini CLI and Qwen VL) to extract unbiased timeline data.
- Gemini timeline: {gemini_path}
- Qwen timeline:  {qwen_path}

Read both files carefully. Note agreements and discrepancies between the two analyses.

## TRD Structure

Write the TRD as a Markdown file (`TRD.md`) with ALL of the following sections:

```markdown
# Trading Requirements Document (TRD)

## 1. Strategy Overview
- Strategy Name: (give it a descriptive name)
- Strategy Type: scalping | breakout | trend | mean_reversion | swing
- Core Concept: (2-3 sentence summary of the strategy as described in the video)
- Timeframe(s): (best timeframe(s) from video)
- Asset Class: (Forex / Crypto / etc.)
- Symbols: (specific pairs mentioned)

## 2. Entry Rules

### Long (BUY)
- Exact conditions for opening a long position
- Include specific indicator values, price action conditions, candlestick patterns
- Minimum confluence factors required

### Short (SELL)
- Exact conditions for opening a short position
- Same level of specificity as long

## 3. Exit Rules
- Stop Loss: (specific value or calculation method)
- Take Profit: (specific value or calculation method)
- Trailing Stop: (if mentioned)
- Time-based exits (if mentioned)

## 4. Position Sizing & Risk Management
- Risk per trade (% of account)
- Fixed lot size or dynamic calculation
- Maximum lot size
- Maximum risk per day (if mentioned)

## 5. Indicators & Parameters

| Indicator | Parameter | Value | Rationale |
|-----------|-----------|-------|-----------|
| (e.g. RSI) | Period | 14 | (reason) |
| (e.g. EMA) | Fast Period | 20 | (reason) |

List ALL parameters with exact values or ranges if a parameter scan is suggested.

## 6. Time Sessions / Market Hours
- Which trading sessions are preferred (London, NY, Tokyo, etc.)
- News/event considerations (if any)

## 7. Trade Management
- How to add to positions (if scalping / pyramid)
- Breakeven adjustments (if mentioned)
- Partial closes (if mentioned)

## 8. Filters / Confluence Rules
- Market regime filters (trending vs ranging)
-Volatility filters (ATR thresholds, etc.)
- Any other filters mentioned

## 9. Edge Cases
- What to do during low liquidity
- Weekend gaps
- Major news events
- Indicator conflicts

## 10. Quality Checklist
- [ ] Entry rules are unambiguous and testable
- [ ] Exit rules have specific values or formulas
- [ ] Position sizing is quantified
- [ ] All mentioned indicators have parameter values
- [ ] Edge cases are addressed
- [ ] This strategy can be coded from this TRD alone

## 11. Open Code Implementation Notes
Anything the developer needs to know that is not yet captured above.
```

## Important Rules

1. **Read BOTH gemini and qwen timeline files** before writing the TRD
2. **Prioritise what BOTH models agree on** — consensus = higher confidence
3. **Flag discrepancies** between the two models in the TRD under "Open Code Implementation Notes"
4. **Do NOT fabricate** indicator values or parameters — only use what was explicitly stated or can be directly inferred
5. **Be conservative** — if something is unclear, mark it as "needs verification" rather than guessing
6. **The TRD must be self-contained** — a developer reading only the TRD should be able to code the strategy without watching the video

Start now. Read the two timeline files, then write the TRD.md to the current directory.
"""


def run_opencode_trd(gemini_path: Path, qwen_path: Path, output_folder: Path) -> Path:
    """Launch OpenCode to produce TRD.md from both timeline files."""
    trd_path = output_folder / "TRD.md"
    if trd_path.exists():
        log.info(f"TRD already exists: {trd_path}")
        return trd_path

    log.info("Launching OpenCode for TRD generation...")

    prompt = PROMPT_OPENCODE.format(
        gemini_path=str(gemini_path),
        qwen_path=str(qwen_path),
    )

    # Write prompt to a file to avoid shell escaping
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as pf:
        pf.write(prompt)
        prompt_file = pf.name

    try:
        # Use OpenCode in one-shot mode with yolo approval
        result = run([
            "opencode",
            "--approval-mode", "yolo",
            "-p", prompt,
        ], cwd=output_folder, timeout=1800)  # 30 min for TRD writing

        log.info(f"OpenCode stdout: {result.stdout[:500]}")

        # Check if TRD was created
        if not trd_path.exists():
            # OpenCode may have written it somewhere else — scan
            candidates = list(output_folder.glob("*.md")) + list(output_folder.glob("TRD*"))
            if candidates:
                # Rename to canonical TRD.md
                src = candidates[0]
                import shutil
                shutil.move(str(src), str(trd_path))
                log.info(f"TRD moved to: {trd_path}")
            else:
                # Write stdout as fallback
                trd_path.write_text(f"# TRD Generation Failed\n\nOpenCode output:\n{result.stdout[:2000]}")
                log.warning("TRD.md not created — wrote fallback")

        log.info(f"TRD written: {trd_path}")
        return trd_path

    finally:
        os.unlink(prompt_file)


# ─── Stage 6: Validate outputs ───────────────────────────────────────────────

def validate_video_folder(folder: Path) -> dict:
    """Return dict of what exists in the video output folder."""
    return {
        "video":    list(folder.glob("video/*.mp4")),
        "audio":    list(folder.glob("audio.mp3")),
        "frames":  list(folder.glob("frames/*.jpg")),
        "gemini":   list(folder.glob("timeline_gemini.json")),
        "qwen":     list(folder.glob("timeline_qwen.json")),
        "trd":      list(folder.glob("TRD.md")),
    }


# ─── Per-video pipeline ──────────────────────────────────────────────────────

def process_video(video_url: str, output_folder: Path) -> dict:
    """Run the full pipeline for one video. Returns result dict."""
    log.info(f"=== Processing video: {video_url} ===")
    log.info(f"Output folder: {output_folder}")
    output_folder.mkdir(parents=True, exist_ok=True)

    stage_results = {}

    # ── Download ──────────────────────────────────────────────────────────
    try:
        video_path = download_video(video_url, output_folder)
        stage_results["download"] = {"status": "ok", "path": str(video_path)}
    except Exception as e:
        log.error(f"Download failed: {e}")
        stage_results["download"] = {"status": "failed", "error": str(e)}
        return stage_results

    # ── Extract frames + audio ──────────────────────────────────────────────
    try:
        audio_path = extract_audio(video_path, output_folder)
        frames = extract_frames(video_path, output_folder)
        stage_results["extract"] = {
            "status": "ok",
            "audio": str(audio_path),
            "frames": len(frames),
        }
    except Exception as e:
        log.error(f"Extract failed: {e}")
        stage_results["extract"] = {"status": "failed", "error": str(e)}
        return stage_results

    # ── Gemini ──────────────────────────────────────────────────────────────
    try:
        gemini_path = run_gemini(video_path, audio_path, frames, output_folder, video_url)
        stage_results["gemini"] = {"status": "ok", "path": str(gemini_path)}
    except Exception as e:
        log.error(f"Gemini failed: {e}")
        stage_results["gemini"] = {"status": "failed", "error": str(e)}

    # ── Qwen ────────────────────────────────────────────────────────────────
    try:
        qwen_path = run_qwen(video_path, audio_path, frames, output_folder, video_url)
        stage_results["qwen"] = {"status": "ok", "path": str(qwen_path)}
    except Exception as e:
        log.error(f"Qwen failed: {e}")
        stage_results["qwen"] = {"status": "failed", "error": str(e)}

    # ── OpenCode TRD ───────────────────────────────────────────────────────
    gemini_out = output_folder / "timeline_gemini.json"
    qwen_out   = output_folder / "timeline_qwen.json"

    if gemini_out.exists() and qwen_out.exists():
        try:
            trd_path = run_opencode_trd(gemini_out, qwen_out, output_folder)
            stage_results["opencode_trd"] = {"status": "ok", "path": str(trd_path)}
        except Exception as e:
            log.error(f"OpenCode TRD failed: {e}")
            stage_results["opencode_trd"] = {"status": "failed", "error": str(e)}
    else:
        log.warning("Skipping OpenCode TRD — missing Gemini and/or Qwen timeline")
        stage_results["opencode_trd"] = {
            "status": "skipped",
            "reason": "missing_timeline_files"
        }

    log.info(f"=== Video complete: {video_url} ===")
    log.info(f"Stage results: {json.dumps(stage_results, indent=2)}")
    return stage_results


# ─── Main dispatcher ─────────────────────────────────────────────────────────

def process_urls(urls: list[str], base_output: Path) -> dict:
    """Process a list of URLs, one folder per video."""
    summary = {}
    for i, url in enumerate(urls, 1):
        video_id = get_video_id(url)
        folder = base_output / video_id
        log.info(f"\n{'='*60}")
        log.info(f"Video {i}/{len(urls)}: {url}")
        log.info(f"Folder: {folder}")
        log.info(f"{'='*60}")
        try:
            result = process_video(url, folder)
            summary[url] = {"status": "ok", "stages": result}
        except Exception as e:
            log.error(f"Video {url} catastrophically failed: {e}")
            summary[url] = {"status": "failed", "error": str(e)}
        # Save progress after each video
        progress_path = base_output / "_pipeline_progress.json"
        progress_path.write_text(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description="video_trd_pipeline — Autonomous Video → TRD")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--url", help="Single YouTube URL")
    group.add_argument("--playlist", help="YouTube playlist URL")
    group.add_argument("--batch", type=Path, help="Text file with one URL per line")
    parser.add_argument("--output", type=Path, default=OUTPUT_BASE,
                        help=f"Output base dir (default: {OUTPUT_BASE})")
    args = parser.parse_args()

    # Resolve URLs
    if args.url:
        urls = [args.url]
    elif args.playlist:
        urls = get_playlist_urls(args.playlist)
    else:
        urls = get_batch_urls(args.batch)

    log.info(f"Pipeline starting — {len(urls)} video(s) to process")
    log.info(f"Output directory: {args.output}")

    results = process_urls(urls, args.output)

    # Final summary
    log.info("\n" + "="*60)
    log.info("PIPELINE COMPLETE — SUMMARY")
    log.info("="*60)
    for url, res in results.items():
        status = res.get("status", "?")
        log.info(f"  [{status}] {url}")
    summary_path = args.output / "_pipeline_summary.json"
    summary_path.write_text(json.dumps(results, indent=2))
    log.info(f"Full summary written to: {summary_path}")
    log.info(f"Log file: {LOG_FILE}")


if __name__ == "__main__":
    main()
