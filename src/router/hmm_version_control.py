"""
HMM Version Control System
===========================

Manages model version synchronization between Contabo (training server)
and Cloudzy (trading server). Handles model sync, checksum verification,
and version mismatch detection.

Features:
- Version tracking on both servers
- Model sync via SSH/SFTP or HTTP API
- Checksum verification for integrity
- Real-time sync progress streaming via WebSocket

Reference: docs/architecture/components.md
"""

import os
import sys
import json
import logging
import hashlib
import pickle
import paramiko
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.database.models import HMMModel, HMMSyncStatus
from src.database.engine import engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import desc

logger = logging.getLogger(__name__)


class SyncStatus(Enum):  # noqa: F821
    """Sync status enumeration."""
    IDLE = "idle"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    VERSION_MISMATCH = "version_mismatch"


@dataclass
class SyncProgress:
    """Sync progress information."""
    status: SyncStatus
    progress: float  # 0.0 - 100.0
    message: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    error: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'status': self.status.value,
            'progress': self.progress,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'error': self.error
        }


class HMMVersionControl:
    """
    HMM Model Version Control System.
    
    Manages model synchronization between training (Contabo) and
    trading (Cloudzy) servers with integrity verification.
    
    Usage:
        ```python
        vc = HMMVersionControl()
        
        # Check for version mismatch
        if vc.check_version_mismatch():
            # Sync model
            vc.sync_model(progress_callback=update_ui)
        ```
    """
    
    def __init__(self, config_path: str = "config/hmm_config.json"):
        """Initialize version control with configuration."""
        self.config = self._load_config(config_path)
        self.sync_config = self.config.get('sync', {})
        
        # Server paths
        self.contabo_model_path = self.sync_config.get('contabo_model_path', '/data/hmm/models')
        self.contabo_metadata_path = self.sync_config.get('contabo_metadata_path', '/data/hmm/metadata')
        self.cloudzy_model_path = self.sync_config.get('cloudzy_model_path', '/data/hmm/models')
        
        # SSH connection settings
        self.contabo_host = os.environ.get('CONTABO_HOST', self.sync_config.get('contabo_host', ''))
        self.contabo_port = int(os.environ.get('CONTABO_PORT', self.sync_config.get('contabo_port', 22)))
        self.contabo_user = os.environ.get('CONTABO_USER', self.sync_config.get('contabo_user', ''))
        
        # Database session
        self.Session = sessionmaker(bind=engine)
        
        # Sync state
        self._sync_progress = SyncProgress(SyncStatus.IDLE, 0.0, "Ready")
        self._progress_callbacks: List[Callable] = []
        
        # Ensure local directories exist
        Path(self.cloudzy_model_path).mkdir(parents=True, exist_ok=True)
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load config file: {e}")
            return {}
    
    def _update_progress(self, status: SyncStatus, progress: float, 
                         message: str, error: Optional[str] = None) -> None:
        """Update sync progress and notify callbacks."""
        self._sync_progress = SyncProgress(status, progress, message, error=error)
        
        for callback in self._progress_callbacks:
            try:
                callback(self._sync_progress.to_dict())
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")
    
    def add_progress_callback(self, callback: Callable) -> None:
        """Add a callback for sync progress updates."""
        self._progress_callbacks.append(callback)
    
    def remove_progress_callback(self, callback: Callable) -> None:
        """Remove a progress callback."""
        if callback in self._progress_callbacks:
            self._progress_callbacks.remove(callback)
    
    def get_sync_progress(self) -> Dict:
        """Get current sync progress."""
        return self._sync_progress.to_dict()
    
    def get_contabo_version(self) -> Optional[Dict]:
        """
        Get current model version on Contabo.
        
        Returns:
            Dictionary with version info or None if unavailable
        """
        try:
            # Try SSH connection to get version
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            ssh.connect(
                hostname=self.contabo_host,
                port=self.contabo_port,
                username=self.contabo_user,
                timeout=10
            )
            
            # Get latest model metadata
            stdin, stdout, stderr = ssh.exec_command(
                f"ls -t {self.contabo_metadata_path}/*.json 2>/dev/null | head -1"
            )
            
            metadata_file = stdout.read().decode().strip()
            
            if not metadata_file:
                ssh.close()
                return None
            
            # Read metadata content
            stdin, stdout, stderr = ssh.exec_command(f"cat {metadata_file}")
            metadata_content = stdout.read().decode()
            
            ssh.close()
            
            metadata = json.loads(metadata_content)
            
            return {
                'version': metadata.get('version'),
                'training_date': metadata.get('training_date'),
                'checksum': metadata.get('checksum'),
                'model_type': metadata.get('model_type'),
                'symbol': metadata.get('symbol'),
                'timeframe': metadata.get('timeframe')
            }
            
        except Exception as e:
            logger.error(f"Failed to get Contabo version: {e}")
            return None
    
    def get_cloudzy_version(self) -> Optional[Dict]:
        """
        Get current model version on Cloudzy.
        
        Returns:
            Dictionary with version info or None if unavailable
        """
        try:
            # Check local model directory
            model_path = Path(self.cloudzy_model_path)
            metadata_files = list(model_path.glob("*.pkl.metadata.json"))
            
            if not metadata_files:
                return None
            
            # Get latest by modification time
            latest_metadata = max(metadata_files, key=lambda x: x.stat().st_mtime)
            
            with open(latest_metadata, 'r') as f:
                metadata = json.load(f)
            
            return {
                'version': metadata.get('version'),
                'training_date': metadata.get('training_date'),
                'checksum': metadata.get('checksum'),
                'model_type': metadata.get('model_type'),
                'symbol': metadata.get('symbol'),
                'timeframe': metadata.get('timeframe'),
                'deployed_date': datetime.fromtimestamp(
                    latest_metadata.stat().st_mtime, tz=timezone.utc
                ).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get Cloudzy version: {e}")
            return None
    
    def check_version_mismatch(self) -> bool:
        """
        Check if Contabo and Cloudzy versions are different.
        
        Returns:
            True if versions mismatch, False if same or unavailable
        """
        contabo = self.get_contabo_version()
        cloudzy = self.get_cloudzy_version()
        
        if not contabo or not cloudzy:
            return False
        
        # Compare versions
        if contabo.get('version') != cloudzy.get('version'):
            logger.info(f"Version mismatch: Contabo={contabo.get('version')}, Cloudzy={cloudzy.get('version')}")
            return True
        
        # Compare checksums
        if contabo.get('checksum') != cloudzy.get('checksum'):
            logger.warning("Same version but different checksums!")
            return True
        
        return False
    
    def sync_model(self, version: Optional[str] = None,
                   model_type: str = "universal",
                   verify_checksum: bool = True) -> bool:
        """
        Sync model from Contabo to Cloudzy.
        
        Args:
            version: Specific version to sync (None = latest)
            model_type: Model type to sync ('universal', 'per_symbol', 'per_symbol_timeframe')
            verify_checksum: Whether to verify checksum after transfer (default: True)
            
        Returns:
            True if sync succeeded
        """
        self._update_progress(SyncStatus.IN_PROGRESS, 0.0, "Starting sync...")
        
        try:
            # Connect to Contabo via SSH
            self._update_progress(SyncStatus.IN_PROGRESS, 10.0, "Connecting to Contabo...")
            
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            ssh.connect(
                hostname=self.contabo_host,
                port=self.contabo_port,
                username=self.contabo_user,
                timeout=30
            )
            
            sftp = ssh.open_sftp()
            
            # Find model file to sync
            self._update_progress(SyncStatus.IN_PROGRESS, 20.0, "Finding model file...")
            
            if version:
                model_pattern = f"hmm_{model_type}_{version}.pkl"
            else:
                model_pattern = f"hmm_{model_type}_*.pkl"
            
            # List matching files
            stdin, stdout, stderr = ssh.exec_command(
                f"ls -t {self.contabo_model_path}/{model_pattern} 2>/dev/null | head -1"
            )
            
            remote_model_path = stdout.read().decode().strip()
            
            if not remote_model_path:
                raise ValueError(f"No model found matching pattern: {model_pattern}")
            
            # Get metadata file
            remote_metadata_path = remote_model_path + ".metadata.json"
            
            self._update_progress(SyncStatus.IN_PROGRESS, 30.0, "Downloading model...")
            
            # Download model file
            local_model_path = Path(self.cloudzy_model_path) / Path(remote_model_path).name
            sftp.get(remote_model_path, str(local_model_path))
            
            self._update_progress(SyncStatus.IN_PROGRESS, 60.0, "Downloading metadata...")
            
            # Download metadata file
            local_metadata_path = Path(self.cloudzy_model_path) / Path(remote_metadata_path).name
            sftp.get(remote_metadata_path, str(local_metadata_path))
            
            sftp.close()
            ssh.close()
            
            # Load metadata
            with open(local_metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Verify checksum if requested
            if verify_checksum:
                self._update_progress(SyncStatus.IN_PROGRESS, 70.0, "Verifying checksum...")
                
                expected_checksum = metadata.get('checksum')
                actual_checksum = self._calculate_checksum(local_model_path)
                
                if expected_checksum != actual_checksum:
                    raise ValueError(f"Checksum mismatch: expected {expected_checksum[:16]}..., got {actual_checksum[:16]}...")
            else:
                self._update_progress(SyncStatus.IN_PROGRESS, 70.0, "Skipping checksum verification...")
            
            self._update_progress(SyncStatus.IN_PROGRESS, 80.0, "Updating database...")
            
            # Update database
            self._update_sync_status(
                contabo_version=metadata.get('version'),
                cloudzy_version=metadata.get('version'),
                status='success'
            )
            
            self._update_progress(SyncStatus.SUCCESS, 100.0, 
                                  f"Model synced successfully: {metadata.get('version')}")
            
            logger.info(f"Model synced: {local_model_path} (v{metadata.get('version')})")
            
            return True
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Sync failed: {error_msg}")
            
            self._update_progress(SyncStatus.FAILED, 0.0, "Sync failed", error=error_msg)
            self._update_sync_status(status='failed', message=error_msg)
            
            return False
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of file."""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def _update_sync_status(self, contabo_version: Optional[str] = None,
                            cloudzy_version: Optional[str] = None,
                            status: str = 'success',
                            message: Optional[str] = None) -> None:
        """Update sync status in database."""
        session = self.Session()
        
        try:
            # Get or create sync status record
            sync_status = session.query(HMMSyncStatus).first()
            
            if not sync_status:
                sync_status = HMMSyncStatus()
                session.add(sync_status)
            
            if contabo_version:
                sync_status.contabo_version = contabo_version
                sync_status.contabo_last_trained = datetime.now(timezone.utc)
            
            if cloudzy_version:
                sync_status.cloudzy_version = cloudzy_version
                sync_status.cloudzy_last_deployed = datetime.now(timezone.utc)
            
            sync_status.last_sync_attempt = datetime.now(timezone.utc)
            sync_status.last_sync_status = status
            sync_status.sync_progress = 100.0 if status == 'success' else 0.0
            sync_status.sync_message = message or f"Sync {status}"
            sync_status.version_mismatch = False
            
            session.commit()
            
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to update sync status: {e}")
        finally:
            session.close()
    
    def get_version_info(self) -> Dict:
        """
        Get comprehensive version information.
        
        Returns:
            Dictionary with version info for both servers
        """
        contabo = self.get_contabo_version()
        cloudzy = self.get_cloudzy_version()
        
        mismatch = False
        if contabo and cloudzy:
            mismatch = contabo.get('version') != cloudzy.get('version')
        
        return {
            'contabo': contabo,
            'cloudzy': cloudzy,
            'version_mismatch': mismatch,
            'sync_progress': self.get_sync_progress()
        }
    
    def list_available_models(self) -> List[Dict]:
        """
        List available models on Contabo.
        
        Returns:
            List of model metadata dictionaries
        """
        try:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            ssh.connect(
                hostname=self.contabo_host,
                port=self.contabo_port,
                username=self.contabo_user,
                timeout=10
            )
            
            # List all metadata files
            stdin, stdout, stderr = ssh.exec_command(
                f"ls {self.contabo_metadata_path}/*.json 2>/dev/null"
            )
            
            metadata_files = stdout.read().decode().strip().split('\n')
            
            models = []
            for metadata_file in metadata_files:
                if not metadata_file:
                    continue
                
                stdin, stdout, stderr = ssh.exec_command(f"cat {metadata_file}")
                content = stdout.read().decode()
                
                try:
                    metadata = json.loads(content)
                    models.append({
                        'version': metadata.get('version'),
                        'model_type': metadata.get('model_type'),
                        'symbol': metadata.get('symbol'),
                        'timeframe': metadata.get('timeframe'),
                        'training_date': metadata.get('training_date'),
                        'training_samples': metadata.get('metrics', {}).get('n_samples', 0),
                        'log_likelihood': metadata.get('metrics', {}).get('train_log_likelihood', 0)
                    })
                except json.JSONDecodeError:
                    continue
            
            ssh.close()
            
            return models
            
        except Exception as e:
            logger.error(f"Failed to list available models: {e}")
            return []


# Global instance
_version_control: Optional[HMMVersionControl] = None


def get_version_control() -> HMMVersionControl:
    """Get or create global version control instance."""
    global _version_control
    if _version_control is None:
        _version_control = HMMVersionControl()
    return _version_control


# Example usage
if __name__ == "__main__":
    vc = HMMVersionControl()
    
    # Get version info
    info = vc.get_version_info()
    print("Version Info:")
    print(json.dumps(info, indent=2, default=str))
    
    # Check for mismatch
    if info.get('version_mismatch'):
        print("\nVersion mismatch detected!")
        
        # Sync model
        def progress_callback(progress):
            print(f"Progress: {progress['progress']:.0f}% - {progress['message']}")
        
        vc.add_progress_callback(progress_callback)
        vc.sync_model()