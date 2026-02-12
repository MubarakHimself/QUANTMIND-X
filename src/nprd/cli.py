"""
NPRD Command Line Interface.

This module provides a Click-based CLI for the NPRD video processing system.

Commands:
- process: Process a single video URL
- batch: Process multiple videos from a file
- playlist: Process all videos in a playlist
- status: Check job status
- jobs: List all jobs
- config: Show/update configuration

Exit Codes:
- 0: Success
- 1: General error
- 2: Validation error (invalid input)
- 3: Authentication error (API key issues)
"""

import sys
import json
import logging
from pathlib import Path
from typing import Optional, List
from datetime import datetime

import click

from .models import NPRDConfig, JobOptions, JobState, TimelineOutput
from .processor import NPRDProcessor
from .job_queue import JobQueueManager
from .logger import NPRDLogger
from .exceptions import (
    NPRDError,
    DownloadError,
    ValidationError,
    ProviderError,
    AuthenticationError,
)


# Exit codes
EXIT_SUCCESS = 0
EXIT_ERROR = 1
EXIT_VALIDATION_ERROR = 2
EXIT_AUTH_ERROR = 3


def setup_logging(verbose: bool, log_file: Optional[Path] = None) -> None:
    """Configure logging based on verbosity level."""
    level = logging.DEBUG if verbose else logging.INFO
    
    handlers = [logging.StreamHandler()]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers
    )


def get_config() -> NPRDConfig:
    """Load configuration from environment."""
    return NPRDConfig.from_env()


def create_processor(config: NPRDConfig) -> NPRDProcessor:
    """Create NPRD processor with configuration."""
    return NPRDProcessor(config=config)


def create_job_queue(config: NPRDConfig) -> JobQueueManager:
    """Create job queue manager with configuration."""
    return JobQueueManager(config=config)


def format_job_status(job_status) -> str:
    """Format job status for display."""
    status_colors = {
        JobState.PENDING: "yellow",
        JobState.DOWNLOADING: "cyan",
        JobState.PROCESSING: "cyan",
        JobState.ANALYZING: "cyan",
        JobState.COMPLETED: "green",
        JobState.FAILED: "red",
    }
    
    status = job_status.status
    color = status_colors.get(status, "white")
    
    result = [
        click.style(f"Job ID: {job_status.job_id}", bold=True),
        f"Status: {click.style(status.value, fg=color)}",
        f"Progress: {job_status.progress}%",
        f"Video URL: {job_status.video_url}",
        f"Created: {job_status.created_at.isoformat()}",
        f"Updated: {job_status.updated_at.isoformat()}",
    ]
    
    if job_status.error:
        result.append(f"Error: {click.style(job_status.error, fg='red')}")
    
    if job_status.result_path:
        result.append(f"Result: {job_status.result_path}")
    
    return "\n".join(result)


def format_timeline(timeline: TimelineOutput, format_type: str = "json") -> str:
    """Format timeline output based on format type."""
    if format_type == "json":
        return timeline.to_json()
    elif format_type == "summary":
        lines = [
            f"Title: {timeline.title}",
            f"Duration: {timeline.duration_seconds}s",
            f"Clips: {len(timeline.timeline)}",
            f"Provider: {timeline.model_provider}",
            f"Processed: {timeline.processed_at}",
            "",
            "Timeline:",
        ]
        for clip in timeline.timeline:
            lines.append(f"  [{clip.timestamp_start} - {clip.timestamp_end}]")
            lines.append(f"    Transcript: {clip.transcript[:100]}..." if len(clip.transcript) > 100 else f"    Transcript: {clip.transcript}")
        return "\n".join(lines)
    else:
        return timeline.to_json()


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option("--config-file", "-c", type=click.Path(exists=True), help="Path to config file")
@click.version_option(version="1.0.0", prog_name="nprd")
@click.pass_context
def cli(ctx, verbose: bool, config_file: Optional[str]):
    """
    NPRD - Video Processing Pipeline
    
    Extract transcripts and visual descriptions from videos using AI.
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["config_file"] = config_file
    
    # Setup logging
    setup_logging(verbose)


@cli.command()
@click.argument("url")
@click.option("--output", "-o", type=click.Path(), help="Output file path")
@click.option("--output-dir", "-d", type=click.Path(), help="Output directory")
@click.option("--format", "-f", "output_format", type=click.Choice(["json", "summary"]), default="json", help="Output format")
@click.option("--provider", "-p", type=click.Choice(["gemini", "qwen"]), help="Model provider to use")
@click.option("--frame-interval", "-i", type=int, default=30, help="Frame extraction interval in seconds")
@click.option("--no-cache", is_flag=True, help="Disable caching")
@click.option("--async", "async_mode", is_flag=True, help="Run asynchronously (submit job and return immediately)")
@click.pass_context
def process(
    ctx,
    url: str,
    output: Optional[str],
    output_dir: Optional[str],
    output_format: str,
    provider: Optional[str],
    frame_interval: int,
    no_cache: bool,
    async_mode: bool
):
    """
    Process a single video URL.
    
    Examples:
        nprd process https://youtube.com/watch?v=xxx
        nprd process https://youtube.com/watch?v=xxx -o output.json
        nprd process https://youtube.com/watch?v=xxx --provider gemini
    """
    try:
        config = get_config()
        
        # Validate URL
        if not url.startswith(("http://", "https://")):
            click.echo(click.style("Error: Invalid URL. Must start with http:// or https://", fg="red"), err=True)
            sys.exit(EXIT_VALIDATION_ERROR)
        
        # Create job options
        options = JobOptions(
            output_dir=Path(output_dir) if output_dir else None,
            model_provider=provider,
            frame_interval=frame_interval,
            cache_enabled=not no_cache,
        )
        
        if async_mode:
            # Submit job to queue and return immediately
            queue = create_job_queue(config)
            job_id = queue.submit_job(url, options)
            click.echo(f"Job submitted: {job_id}")
            click.echo(f"Check status with: nprd status {job_id}")
            sys.exit(EXIT_SUCCESS)
        
        # Process synchronously
        click.echo(f"Processing: {url}")
        processor = create_processor(config)
        
        # Progress callback for verbose mode
        if ctx.obj.get("verbose"):
            def progress_callback(job_id, state, progress, message):
                click.echo(f"  [{progress}%] {state.value}: {message}")
            processor.set_status_callback(progress_callback)
        
        result = processor.process(url, options=options)
        
        # Format output
        output_text = format_timeline(result.timeline, output_format)
        
        # Write to file or stdout
        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(output_text)
            click.echo(f"Output written to: {output_path}")
        else:
            click.echo(output_text)
        
        click.echo(click.style(f"\nProcessing complete in {result.processing_time_seconds:.2f}s", fg="green"))
        sys.exit(EXIT_SUCCESS)
        
    except AuthenticationError as e:
        click.echo(click.style(f"Authentication error: {e}", fg="red"), err=True)
        sys.exit(EXIT_AUTH_ERROR)
    except ValidationError as e:
        click.echo(click.style(f"Validation error: {e}", fg="red"), err=True)
        sys.exit(EXIT_VALIDATION_ERROR)
    except NPRDError as e:
        click.echo(click.style(f"Error: {e}", fg="red"), err=True)
        sys.exit(EXIT_ERROR)
    except Exception as e:
        click.echo(click.style(f"Unexpected error: {e}", fg="red"), err=True)
        if ctx.obj.get("verbose"):
            import traceback
            traceback.print_exc()
        sys.exit(EXIT_ERROR)


@cli.command()
@click.argument("file", type=click.Path(exists=True))
@click.option("--output-dir", "-d", type=click.Path(), required=True, help="Output directory for results")
@click.option("--provider", "-p", type=click.Choice(["gemini", "qwen"]), help="Model provider to use")
@click.option("--frame-interval", "-i", type=int, default=30, help="Frame extraction interval in seconds")
@click.option("--concurrent", "-j", type=int, default=3, help="Number of concurrent jobs")
@click.option("--no-cache", is_flag=True, help="Disable caching")
@click.option("--continue-on-error", is_flag=True, help="Continue processing if a video fails")
@click.pass_context
def batch(
    ctx,
    file: str,
    output_dir: str,
    provider: Optional[str],
    frame_interval: int,
    concurrent: int,
    no_cache: bool,
    continue_on_error: bool
):
    """
    Process multiple videos from a file.
    
    The input file should contain one URL per line.
    
    Examples:
        nprd batch urls.txt -d output/
        nprd batch urls.txt -d output/ --concurrent 5
    """
    try:
        config = get_config()
        config.max_concurrent_jobs = concurrent
        
        # Read URLs from file
        input_path = Path(file)
        with open(input_path, "r") as f:
            urls = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        
        if not urls:
            click.echo(click.style("Error: No URLs found in input file", fg="red"), err=True)
            sys.exit(EXIT_VALIDATION_ERROR)
        
        click.echo(f"Found {len(urls)} URLs to process")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create job options
        options = JobOptions(
            output_dir=output_path,
            model_provider=provider,
            frame_interval=frame_interval,
            cache_enabled=not no_cache,
        )
        
        # Submit jobs
        queue = create_job_queue(config)
        
        job_ids = []
        for i, url in enumerate(urls):
            try:
                job_id = queue.submit_job(url, options)
                job_ids.append(job_id)
                click.echo(f"[{i+1}/{len(urls)}] Submitted: {url[:60]}... -> {job_id}")
            except Exception as e:
                click.echo(click.style(f"[{i+1}/{len(urls)}] Failed to submit {url}: {e}", fg="red"), err=True)
                if not continue_on_error:
                    sys.exit(EXIT_ERROR)
        
        click.echo(f"\nSubmitted {len(job_ids)} jobs")
        click.echo(f"Check status with: nprd jobs")
        click.echo(f"Results will be written to: {output_path}")
        
        sys.exit(EXIT_SUCCESS)
        
    except ValidationError as e:
        click.echo(click.style(f"Validation error: {e}", fg="red"), err=True)
        sys.exit(EXIT_VALIDATION_ERROR)
    except Exception as e:
        click.echo(click.style(f"Error: {e}", fg="red"), err=True)
        sys.exit(EXIT_ERROR)


@cli.command()
@click.argument("url")
@click.option("--output-dir", "-d", type=click.Path(), required=True, help="Output directory for results")
@click.option("--provider", "-p", type=click.Choice(["gemini", "qwen"]), help="Model provider to use")
@click.option("--frame-interval", "-i", type=int, default=30, help="Frame extraction interval in seconds")
@click.option("--no-cache", is_flag=True, help="Disable caching")
@click.option("--extract-only", is_flag=True, help="Only extract video URLs, don't process")
@click.pass_context
def playlist(
    ctx,
    url: str,
    output_dir: str,
    provider: Optional[str],
    frame_interval: int,
    no_cache: bool,
    extract_only: bool
):
    """
    Process all videos in a playlist.
    
    Examples:
        nprd playlist https://youtube.com/playlist?list=xxx -d output/
        nprd playlist https://youtube.com/playlist?list=xxx --extract-only
    """
    try:
        config = get_config()
        processor = create_processor(config)
        
        # Extract playlist info
        click.echo(f"Extracting playlist: {url}")
        playlist_info = processor.extract_playlist(url)
        
        click.echo(f"Playlist: {playlist_info.title}")
        click.echo(f"Videos: {playlist_info.video_count}")
        
        if extract_only:
            # Just print video URLs
            click.echo("\nVideo URLs:")
            for i, video_url in enumerate(playlist_info.video_urls):
                click.echo(f"  {i+1}. {video_url}")
            sys.exit(EXIT_SUCCESS)
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create job options
        options = JobOptions(
            output_dir=output_path,
            model_provider=provider,
            frame_interval=frame_interval,
            cache_enabled=not no_cache,
        )
        
        # Submit jobs for each video
        queue = create_job_queue(config)
        job_ids = processor.create_playlist_jobs(url, queue, options)
        
        click.echo(f"\nSubmitted {len([j for j in job_ids if j])} jobs")
        click.echo(f"Check status with: nprd jobs")
        click.echo(f"Results will be written to: {output_path}")
        
        sys.exit(EXIT_SUCCESS)
        
    except ValidationError as e:
        click.echo(click.style(f"Validation error: {e}", fg="red"), err=True)
        sys.exit(EXIT_VALIDATION_ERROR)
    except DownloadError as e:
        click.echo(click.style(f"Download error: {e}", fg="red"), err=True)
        sys.exit(EXIT_ERROR)
    except Exception as e:
        click.echo(click.style(f"Error: {e}", fg="red"), err=True)
        sys.exit(EXIT_ERROR)


@cli.command()
@click.argument("job_id")
@click.option("--watch", "-w", is_flag=True, help="Watch job status until completion")
@click.option("--interval", "-i", type=int, default=5, help="Watch poll interval in seconds")
@click.pass_context
def status(ctx, job_id: str, watch: bool, interval: int):
    """
    Check the status of a job.
    
    Examples:
        nprd status job_abc123
        nprd status job_abc123 --watch
    """
    import time
    
    try:
        config = get_config()
        queue = create_job_queue(config)
        
        while True:
            job_status = queue.get_job_status(job_id)
            
            if watch:
                click.clear()
            
            click.echo(format_job_status(job_status))
            
            if not watch:
                break
            
            # Check if job is in terminal state
            if job_status.status in [JobState.COMPLETED, JobState.FAILED]:
                click.echo(f"\nJob reached terminal state: {job_status.status.value}")
                break
            
            click.echo(f"\nRefreshing in {interval}s... (Ctrl+C to stop)")
            time.sleep(interval)
        
        sys.exit(EXIT_SUCCESS)
        
    except Exception as e:
        click.echo(click.style(f"Error: {e}", fg="red"), err=True)
        sys.exit(EXIT_ERROR)


@cli.command()
@click.option("--status", "-s", "filter_status", type=click.Choice(["pending", "downloading", "processing", "analyzing", "completed", "failed"]), help="Filter by status")
@click.option("--limit", "-n", type=int, default=20, help="Maximum number of jobs to show")
@click.option("--format", "-f", "output_format", type=click.Choice(["table", "json"]), default="table", help="Output format")
@click.pass_context
def jobs(ctx, filter_status: Optional[str], limit: int, output_format: str):
    """
    List all jobs.
    
    Examples:
        nprd jobs
        nprd jobs --status pending
        nprd jobs --format json
    """
    try:
        config = get_config()
        queue = create_job_queue(config)
        
        # Map filter status to JobState
        status_filter = None
        if filter_status:
            status_filter = JobState(filter_status.upper())
        
        job_list = queue.list_jobs(status=status_filter, limit=limit)
        
        if not job_list:
            click.echo("No jobs found")
            sys.exit(EXIT_SUCCESS)
        
        if output_format == "json":
            output = json.dumps([j.to_dict() for j in job_list], indent=2)
            click.echo(output)
        else:
            # Table format
            click.echo(f"{'JOB ID':<20} {'STATUS':<12} {'PROGRESS':<10} {'URL':<50}")
            click.echo("-" * 92)
            
            for job in job_list:
                status_color = {
                    JobState.PENDING: "yellow",
                    JobState.DOWNLOADING: "cyan",
                    JobState.PROCESSING: "cyan",
                    JobState.ANALYZING: "cyan",
                    JobState.COMPLETED: "green",
                    JobState.FAILED: "red",
                }.get(job.status, "white")
                
                status_text = click.style(f"{job.status.value:<12}", fg=status_color)
                url_truncated = job.video_url[:50] + "..." if len(job.video_url) > 50 else job.video_url
                
                click.echo(f"{job.job_id:<20} {status_text} {job.progress:<10}% {url_truncated}")
        
        sys.exit(EXIT_SUCCESS)
        
    except Exception as e:
        click.echo(click.style(f"Error: {e}", fg="red"), err=True)
        sys.exit(EXIT_ERROR)


@cli.command()
@click.option("--show", is_flag=True, help="Show current configuration")
@click.option("--set", "set_value", type=(str, str), multiple=True, help="Set configuration value (--set KEY VALUE)")
@click.option("--format", "-f", "output_format", type=click.Choice(["table", "json"]), default="table", help="Output format")
@click.pass_context
def config(ctx, show: bool, set_value: tuple, output_format: str):
    """
    Show or update NPRD configuration.
    
    Examples:
        nprd config --show
        nprd config --show --format json
        nprd config --set NPRD_MAX_CONCURRENT_JOBS 5
    """
    try:
        nprd_config = get_config()
        
        if set_value:
            click.echo(click.style("Note: Setting configuration values via CLI is not persisted.", fg="yellow"))
            click.echo("Use environment variables or a config file instead.")
            click.echo("\nEnvironment variables to set:")
            for key, value in set_value:
                click.echo(f"  export {key}={value}")
            sys.exit(EXIT_SUCCESS)
        
        # Default behavior: show configuration
        config_dict = nprd_config.to_dict()
        
        if output_format == "json":
            click.echo(json.dumps(config_dict, indent=2))
        else:
            click.echo("NPRD Configuration")
            click.echo("=" * 50)
            
            # Group by category
            categories = {
                "Model Providers": ["gemini_api_key", "gemini_yolo_mode", "qwen_api_key", "qwen_headless", "qwen_requests_per_day"],
                "Cache": ["cache_dir", "cache_max_size_gb", "cache_max_age_days"],
                "Job Queue": ["max_concurrent_jobs", "job_db_path"],
                "Output": ["output_dir"],
                "Processing": ["default_frame_interval", "default_audio_bitrate", "default_audio_channels"],
                "Retry": ["max_retry_attempts", "base_retry_delay"],
                "Logging": ["log_level", "log_file"],
            }
            
            for category, keys in categories.items():
                click.echo(f"\n{category}:")
                for key in keys:
                    value = config_dict.get(key, "N/A")
                    click.echo(f"  {key}: {value}")
        
        sys.exit(EXIT_SUCCESS)
        
    except Exception as e:
        click.echo(click.style(f"Error: {e}", fg="red"), err=True)
        sys.exit(EXIT_ERROR)


@cli.command()
@click.argument("job_id")
@click.option("--output", "-o", type=click.Path(), help="Output file path")
@click.option("--format", "-f", "output_format", type=click.Choice(["json", "summary"]), default="json", help="Output format")
@click.pass_context
def result(ctx, job_id: str, output: Optional[str], output_format: str):
    """
    Get the result of a completed job.
    
    Examples:
        nprd result job_abc123
        nprd result job_abc123 -o output.json
    """
    try:
        config = get_config()
        queue = create_job_queue(config)
        
        # Get job status first
        job_status = queue.get_job_status(job_id)
        
        if job_status.status != JobState.COMPLETED:
            click.echo(click.style(f"Job not completed. Current status: {job_status.status.value}", fg="yellow"), err=True)
            sys.exit(EXIT_VALIDATION_ERROR)
        
        # Get result
        timeline = queue.get_job_result(job_id)
        
        if timeline is None:
            click.echo(click.style("Result not found", fg="red"), err=True)
            sys.exit(EXIT_ERROR)
        
        # Format output
        output_text = format_timeline(timeline, output_format)
        
        # Write to file or stdout
        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(output_text)
            click.echo(f"Result written to: {output_path}")
        else:
            click.echo(output_text)
        
        sys.exit(EXIT_SUCCESS)
        
    except Exception as e:
        click.echo(click.style(f"Error: {e}", fg="red"), err=True)
        sys.exit(EXIT_ERROR)


def main():
    """Main entry point for NPRD CLI."""
    cli(obj={})


if __name__ == "__main__":
    main()
