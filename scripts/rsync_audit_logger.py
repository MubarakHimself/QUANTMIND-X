#!/usr/bin/env python3
"""
Audit Logger for Rsync Scripts

Logs rsync operations to the QuantMindX audit trail.
Can be called from bash scripts or used directly.

Usage:
    python -m scripts.rsync_audit_logger --action rsync_start --details "Starting sync from Cloudzy"
    python -m scripts.rsync_audit_logger --action rsync_success --details "Synced 150MB"
    python -m scripts.rsync_audit_logger --action rsync_failure --reason "SSH connection timeout"
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def get_default_log_path():
    """Get default path for rsync audit log."""
    log_dir = os.environ.get('LOG_DIR', '/var/log/quantmindx')
    return Path(log_dir) / 'rsync_audit.json'


def log_to_file(action: str, details: str = None, reason: str = None, metadata: dict = None):
    """Log rsync operation to JSON audit file."""
    log_path = get_default_log_path()

    # Ensure log directory exists
    log_path.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'action': action,
        'layer': 'system_health',
        'event_type': f'rsync_{action}',
        'actor': 'rsync_cron',
        'details': details,
        'reason': reason,
        'metadata': metadata or {}
    }

    # Append to JSON log (one JSON object per line - JSON Lines format)
    with open(log_path, 'a') as f:
        f.write(json.dumps(entry) + '\n')

    return entry


def log_rsync_start(details: str = None, metadata: dict = None):
    """Log rsync operation started."""
    return log_to_file('start', details=details, metadata=metadata)


def log_rsync_success(files_synced: int = None, bytes_transferred: int = None, metadata: dict = None):
    """Log rsync operation succeeded."""
    details = f"Synced {files_synced} files, {bytes_transferred} bytes" if files_synced else "Sync completed successfully"
    return log_to_file('success', details=details, metadata=metadata)


def log_rsync_failure(reason: str, metadata: dict = None):
    """Log rsync operation failed."""
    return log_to_file('failure', reason=reason, metadata=metadata)


def main():
    parser = argparse.ArgumentParser(description='Audit logger for rsync operations')
    parser.add_argument('--action', required=True,
                        choices=['start', 'success', 'failure', 'verify_start', 'verify_success', 'verify_failure'],
                        help='Rsync action type')
    parser.add_argument('--details', help='Details about the operation')
    parser.add_argument('--reason', help='Reason for failure')
    parser.add_argument('--files', type=int, help='Number of files synced')
    parser.add_argument('--bytes', type=int, help='Bytes transferred')
    parser.add_argument('--metadata', type=str, help='JSON metadata')

    args = parser.parse_args()

    metadata = {}
    if args.metadata:
        try:
            metadata = json.loads(args.metadata)
        except json.JSONDecodeError:
            metadata = {'raw': args.metadata}

    if args.files is not None:
        metadata['files_synced'] = args.files
    if args.bytes is not None:
        metadata['bytes_transferred'] = args.bytes

    if args.action == 'start' or args.action == 'verify_start':
        log_rsync_start(details=args.details, metadata=metadata)
    elif args.action == 'success' or args.action == 'verify_success':
        log_rsync_success(files_synced=args.files, bytes_transferred=args.bytes, metadata=metadata)
    elif args.action == 'failure' or args.action == 'verify_failure':
        log_rsync_failure(reason=args.reason or 'Unknown failure', metadata=metadata)

    print(f"Logged: {args.action}")


if __name__ == '__main__':
    main()