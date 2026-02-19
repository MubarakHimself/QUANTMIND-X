"""
GitHub EA Sync Service

Synchronizes Expert Advisor (EA) files from GitHub repositories.
Parses MQL5 files and generates BotManifests for the QuantMind system.

**Validates: Property 18: GitHub EA Sync**
"""

import re
import hashlib
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

from sqlalchemy.orm import Session

from src.agents.integrations.git_client import GitClient
from src.database.models import ImportedEA, BotManifest, get_db_session

logger = logging.getLogger(__name__)


class EAParser:
    """
    Parser for MQL5 Expert Advisor files.
    
    Extracts metadata, input parameters, and strategy information.
    """
    
    # Regex patterns for MQL5 parsing
    INPUT_PATTERN = re.compile(
        r'input\s+(\w+)\s+(\w+)\s*=\s*([^;]+);(?:\s*//\s*(.+))?',
        re.MULTILINE
    )
    
    PROPERTY_PATTERN = re.compile(
        r'#property\s+(\w+)\s+"?([^"]+)"?',
        re.MULTILINE
    )
    
    STRATEGY_COMMENT_PATTERN = re.compile(
        r'//\s*Strategy:\s*(.+)',
        re.IGNORECASE
    )
    
    TIMEFRAME_PATTERN = re.compile(
        r'PERIOD_(\w+)',
        re.MULTILINE
    )
    
    SYMBOL_PATTERN = re.compile(
        r'Symbol\s*\(\s*\)|"[A-Z]{6}"',
        re.MULTILINE
    )

    def parse_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Parse an MQL5 file and extract metadata.
        
        Args:
            file_path: Path to the .mq5 file
            
        Returns:
            Dictionary with parsed EA data
        """
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            return self.parse_content(content, str(file_path))
        except Exception as e:
            logger.error(f"Failed to parse file {file_path}: {e}")
            return {'error': str(e)}

    def parse_content(self, content: str, file_path: str = "unknown") -> Dict[str, Any]:
        """
        Parse MQL5 content and extract metadata.
        
        Args:
            content: MQL5 source code
            file_path: File path for reference
            
        Returns:
            Dictionary with parsed EA data
        """
        result = {
            'file_path': file_path,
            'filename': Path(file_path).name,
            'lines_of_code': len(content.splitlines()),
            'checksum': self.calculate_checksum(content),
            'parsed_at': datetime.utcnow().isoformat()
        }
        
        # Extract #property directives
        properties = {}
        for match in self.PROPERTY_PATTERN.finditer(content):
            prop_name = match.group(1)
            prop_value = match.group(2).strip('"')
            properties[prop_name] = prop_value
        
        result['properties'] = properties
        result['description'] = properties.get('description', '')
        result['version'] = properties.get('version', '1.00')
        result['author'] = properties.get('copyright', 'Unknown')
        result['strict'] = 'strict' in properties
        
        # Extract input parameters
        inputs = []
        for match in self.INPUT_PATTERN.finditer(content):
            param_type = match.group(1)
            param_name = match.group(2)
            default_value = match.group(3).strip()
            description = match.group(4).strip() if match.group(4) else param_name
            
            inputs.append({
                'type': param_type,
                'name': param_name,
                'default': default_value,
                'description': description
            })
        
        result['inputs'] = inputs
        result['input_count'] = len(inputs)
        
        # Try to identify strategy type from comments
        strategy_match = self.STRATEGY_COMMENT_PATTERN.search(content)
        if strategy_match:
            result['strategy_type'] = strategy_match.group(1).strip()
        else:
            result['strategy_type'] = self._infer_strategy_type(content)
        
        # Extract timeframe preferences
        timeframes = list(set(self.TIMEFRAME_PATTERN.findall(content)))
        result['timeframes'] = timeframes if timeframes else ['CURRENT']
        
        # Extract symbol usage
        symbols = self._extract_symbols(content)
        result['symbols'] = symbols if symbols else ['ANY']
        
        # Calculate complexity score
        result['complexity'] = self._calculate_complexity(content)
        
        return result

    def calculate_checksum(self, content: str) -> str:
        """Calculate SHA256 checksum of content."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    def _infer_strategy_type(self, content: str) -> str:
        """Infer strategy type from code patterns."""
        content_lower = content.lower()
        
        if 'rsi' in content_lower:
            return 'RSI Strategy'
        elif 'macd' in content_lower:
            return 'MACD Strategy'
        elif 'movingaverage' in content_lower or 'ma(' in content_lower:
            return 'Moving Average Strategy'
        elif 'bollinger' in content_lower:
            return 'Bollinger Bands Strategy'
        elif 'stochastic' in content_lower:
            return 'Stochastic Strategy'
        elif 'fibonacci' in content_lower:
            return 'Fibonacci Strategy'
        elif 'pivot' in content_lower:
            return 'Pivot Point Strategy'
        elif 'breakout' in content_lower:
            return 'Breakout Strategy'
        elif 'scalp' in content_lower:
            return 'Scalping Strategy'
        elif 'martingale' in content_lower:
            return 'Martingale Strategy'
        elif 'grid' in content_lower:
            return 'Grid Strategy'
        else:
            return 'Generic Strategy'

    def _extract_symbols(self, content: str) -> List[str]:
        """Extract symbol references from code."""
        symbols = set()
        
        # Find explicit symbol strings
        for match in re.finditer(r'"([A-Z]{6})"', content):
            symbols.add(match.group(1))
        
        # Find Symbol() calls that are compared to strings
        for match in re.finditer(r'Symbol\s*\(\s*\)\s*==\s*"([A-Z]{6})"', content):
            symbols.add(match.group(1))
        
        return sorted(list(symbols))

    def _calculate_complexity(self, content: str) -> str:
        """Calculate complexity score based on code patterns."""
        score = 0
        
        # Count various code constructs
        score += content.count('if(') + content.count('if (')
        score += content.count('for(') + content.count('for (')
        score += content.count('while(') + content.count('while (')
        score += content.count('switch(') + content.count('switch (')
        score += content.count('class ')
        score += content.count('OrderSend')
        score += content.count('OrderSelect')
        
        if score < 20:
            return 'Simple'
        elif score < 50:
            return 'Moderate'
        elif score < 100:
            return 'Complex'
        else:
            return 'Very Complex'


class GitHubEASync:
    """
    Synchronizes Expert Advisors from GitHub repositories.
    
    Clones or pulls GitHub repositories containing .mq5 files,
    parses them, and generates BotManifests for the system.
    """
    
    def __init__(
        self,
        repo_url: str,
        library_path: str = "/data/ea-library",
        branch: str = "main"
    ):
        """
        Initialize GitHub EA Sync service.
        
        Args:
            repo_url: GitHub repository URL
            library_path: Local path to store cloned repository
            branch: Branch to clone/pull
        """
        self.repo_url = repo_url
        self.library_path = Path(library_path)
        self.branch = branch
        self.git_client = GitClient(self.library_path)
        self.parser = EAParser()
        
        self._last_sync_time: Optional[datetime] = None
        self._last_commit_hash: Optional[str] = None
        self._sync_errors: List[str] = []

    @property
    def last_sync_time(self) -> Optional[datetime]:
        return self._last_sync_time

    @property
    def last_commit_hash(self) -> Optional[str]:
        return self._last_commit_hash

    @property
    def sync_errors(self) -> List[str]:
        return self._sync_errors.copy()

    def sync_repository(self) -> Dict[str, Any]:
        """
        Clone or pull the GitHub repository.
        
        Returns:
            Dictionary with sync status and information
        """
        self._sync_errors = []
        
        result = {
            'status': 'unknown',
            'repo_url': self.repo_url,
            'branch': self.branch,
            'commit_hash': None,
            'files_changed': [],
            'errors': []
        }
        
        try:
            # Check if repo exists
            if (self.library_path / ".git").exists():
                # Pull updates
                logger.info(f"Pulling updates from {self.repo_url}")
                success = self.git_client.pull_updates(self.branch)
                
                if not success:
                    result['status'] = 'pull_failed'
                    result['errors'].append('Failed to pull updates')
                    return result
            else:
                # Clone repository
                logger.info(f"Cloning repository {self.repo_url}")
                success = self.git_client.clone_remote_repo(self.repo_url, self.branch)
                
                if not success:
                    result['status'] = 'clone_failed'
                    result['errors'].append('Failed to clone repository')
                    return result
            
            # Get current commit hash
            commit_hash = self.git_client.get_commit_hash()
            result['commit_hash'] = commit_hash
            
            # Check for changes
            if self._last_commit_hash and commit_hash != self._last_commit_hash:
                result['files_changed'] = self.git_client.list_changed_files(self._last_commit_hash)
            
            self._last_commit_hash = commit_hash
            self._last_sync_time = datetime.utcnow()
            result['status'] = 'success'
            result['synced_at'] = self._last_sync_time.isoformat()
            
            logger.info(f"Repository synced successfully at commit {commit_hash}")
            return result
            
        except Exception as e:
            error_msg = f"Sync failed: {str(e)}"
            logger.error(error_msg)
            self._sync_errors.append(error_msg)
            result['status'] = 'error'
            result['errors'].append(error_msg)
            return result

    def scan_ea_files(self) -> List[Path]:
        """
        Scan repository for all .mq5 files.
        
        Returns:
            List of paths to EA files
        """
        if not self.library_path.exists():
            logger.warning(f"Library path does not exist: {self.library_path}")
            return []
        
        ea_files = []
        for mq5_file in self.library_path.rglob("*.mq5"):
            if mq5_file.is_file():
                ea_files.append(mq5_file)
        
        logger.info(f"Found {len(ea_files)} EA files")
        return sorted(ea_files)

    def parse_ea_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Parse an EA file and extract metadata.
        
        Args:
            file_path: Path to the .mq5 file
            
        Returns:
            Dictionary with parsed EA data
        """
        return self.parser.parse_file(file_path)

    def generate_bot_manifest(self, ea_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a BotManifest from parsed EA data.
        
        Args:
            ea_data: Parsed EA data from parse_ea_file()
            
        Returns:
            BotManifest dictionary
        """
        manifest = {
            'name': Path(ea_data['filename']).stem,
            'source_type': 'imported_ea',
            'source_path': ea_data['file_path'],
            'description': ea_data.get('description', ''),
            'strategy_type': ea_data.get('strategy_type', 'Unknown'),
            'timeframes': ea_data.get('timeframes', ['CURRENT']),
            'symbols': ea_data.get('symbols', ['ANY']),
            'inputs': ea_data.get('inputs', []),
            'version': ea_data.get('version', '1.00'),
            'author': ea_data.get('author', 'Unknown'),
            'complexity': ea_data.get('complexity', 'Unknown'),
            'checksum': ea_data.get('checksum', ''),
            'imported_at': datetime.utcnow().isoformat(),
            'status': 'pending_review'
        }
        
        return manifest

    def store_imported_ea(self, ea_data: Dict[str, Any], db: Session) -> Optional[ImportedEA]:
        """
        Store imported EA in the database.
        
        Args:
            ea_data: Parsed EA data
            db: Database session
            
        Returns:
            Created ImportedEA record, or None on error
        """
        try:
            # Check if EA already exists
            existing = db.query(ImportedEA).filter(
                ImportedEA.ea_filename == ea_data['filename'],
                ImportedEA.github_path == ea_data['file_path']
            ).first()
            
            if existing:
                # Store the previous checksum BEFORE overwriting
                previous_checksum = existing.checksum
                new_checksum = ea_data['checksum']
                
                # Compare checksums to detect updates
                if previous_checksum != new_checksum:
                    existing.status = 'updated'
                else:
                    existing.status = 'unchanged'
                
                # Update record with new values
                existing.checksum = new_checksum
                existing.last_synced = datetime.utcnow()
                existing.lines_of_code = ea_data.get('lines_of_code', 0)
                existing.strategy_type = ea_data.get('strategy_type', 'Unknown')
                
                db.commit()
                logger.info(f"Updated existing EA: {ea_data['filename']} (status: {existing.status})")
                return existing
            else:
                # Create new record
                imported_ea = ImportedEA(
                    ea_filename=ea_data['filename'],
                    github_path=ea_data['file_path'],
                    lines_of_code=ea_data.get('lines_of_code', 0),
                    strategy_type=ea_data.get('strategy_type', 'Unknown'),
                    checksum=ea_data['checksum'],
                    status='new',
                    imported_at=datetime.utcnow(),
                    last_synced=datetime.utcnow()
                )
                
                db.add(imported_ea)
                db.commit()
                logger.info(f"Created new imported EA: {ea_data['filename']}")
                return imported_ea
                
        except Exception as e:
            logger.error(f"Failed to store imported EA: {e}")
            db.rollback()
            return None

    def check_for_updates(self, db: Session) -> List[Dict[str, Any]]:
        """
        Check for EA updates by comparing checksums.
        
        Args:
            db: Database session
            
        Returns:
            List of EAs with changed status
        """
        updated_eas = []
        
        try:
            ea_files = self.scan_ea_files()
            
            for ea_file in ea_files:
                ea_data = self.parse_ea_file(ea_file)
                
                # Check against database
                existing = db.query(ImportedEA).filter(
                    ImportedEA.github_path == str(ea_file)
                ).first()
                
                if existing:
                    if existing.checksum != ea_data['checksum']:
                        updated_eas.append({
                            'id': existing.id,
                            'filename': existing.ea_filename,
                            'status': 'updated',
                            'old_checksum': existing.checksum,
                            'new_checksum': ea_data['checksum']
                        })
                else:
                    updated_eas.append({
                        'filename': ea_data['filename'],
                        'github_path': str(ea_file),
                        'status': 'new',
                        'checksum': ea_data['checksum']
                    })
            
            return updated_eas
            
        except Exception as e:
            logger.error(f"Failed to check for updates: {e}")
            return []

    def full_sync(self, db: Session) -> Dict[str, Any]:
        """
        Perform a full sync: pull repository, scan EAs, parse, and create BotManifests.
        
        Args:
            db: Database session
            
        Returns:
            Dictionary with sync results
        """
        result = {
            'sync_status': 'unknown',
            'repository': {},
            'eas_found': 0,
            'eas_new': 0,
            'eas_updated': 0,
            'eas_unchanged': 0,
            'manifests_created': 0,
            'errors': []
        }
        
        # Sync repository
        repo_result = self.sync_repository()
        result['repository'] = repo_result
        
        if repo_result['status'] != 'success':
            result['sync_status'] = 'repository_sync_failed'
            result['errors'].extend(repo_result.get('errors', []))
            return result
        
        # Scan for EA files
        ea_files = self.scan_ea_files()
        result['eas_found'] = len(ea_files)
        
        # Import here to avoid circular imports
        from src.router.bot_manifest import BotRegistry, BotManifest, StrategyType, TradeFrequency, TradingMode
        
        # Get or create BotRegistry for manifest persistence
        try:
            bot_registry = BotRegistry()
        except Exception as e:
            logger.warning(f"Could not create BotRegistry: {e}")
            bot_registry = None
        
        # Process each EA file
        for ea_file in ea_files:
            try:
                ea_data = self.parse_ea_file(ea_file)
                
                if 'error' in ea_data:
                    result['errors'].append(f"Parse error for {ea_file}: {ea_data['error']}")
                    continue
                
                imported_ea = self.store_imported_ea(ea_data, db)
                
                if imported_ea:
                    if imported_ea.status == 'new':
                        result['eas_new'] += 1
                    elif imported_ea.status == 'updated':
                        result['eas_updated'] += 1
                    else:
                        result['eas_unchanged'] += 1
                    
                    # Generate and persist BotManifest for new/updated EAs
                    if imported_ea.status in ['new', 'updated'] and bot_registry:
                        try:
                            # Generate manifest data from EA
                            manifest_data = self.generate_bot_manifest(ea_data)
                            
                            # Map strategy type string to enum
                            strategy_type_str = manifest_data.get('strategy_type', 'Unknown')
                            strategy_type = self._map_strategy_type(strategy_type_str)
                            
                            # Map complexity to frequency (simple estimation)
                            complexity = manifest_data.get('complexity', 'Simple')
                            frequency = self._map_complexity_to_frequency(complexity)
                            
                            # Create BotManifest instance
                            bot_id = f"ea_{Path(ea_data['filename']).stem}_{imported_ea.id}"
                            
                            manifest = BotManifest(
                                bot_id=bot_id,
                                name=manifest_data.get('name', ea_data['filename']),
                                description=manifest_data.get('description', ''),
                                strategy_type=strategy_type,
                                frequency=frequency,
                                symbols=manifest_data.get('symbols', []),
                                timeframes=manifest_data.get('timeframes', []),
                                tags=['@primal'],  # Start at primal tag
                                trading_mode=TradingMode.PAPER,  # Start in paper mode
                                source_type='imported_ea',
                                source_path=manifest_data.get('source_path', ''),
                            )
                            
                            # Link ImportedEA to manifest ID
                            imported_ea.bot_manifest_id = bot_id
                            db.commit()
                            
                            # Register in BotRegistry
                            bot_registry._bots[manifest.bot_id] = manifest
                            result['manifests_created'] += 1
                            
                            logger.info(f"Created BotManifest {bot_id} for EA {ea_data['filename']}")
                            
                        except Exception as me:
                            logger.error(f"Failed to create BotManifest for {ea_file}: {me}")
                            result['errors'].append(f"Manifest error for {ea_file}: {str(me)}")
                
            except Exception as e:
                error_msg = f"Failed to process {ea_file}: {str(e)}"
                logger.error(error_msg)
                result['errors'].append(error_msg)
        
        # Save bot registry if we created manifests
        if bot_registry and result['manifests_created'] > 0:
            try:
                bot_registry._save()
                logger.info(f"Saved {result['manifests_created']} new bot manifests to registry")
            except Exception as e:
                logger.error(f"Failed to save bot registry: {e}")
        
        result['sync_status'] = 'success'
        result['synced_at'] = datetime.utcnow().isoformat()
        
        logger.info(
            f"Full sync complete: {result['eas_found']} EAs found, "
            f"{result['eas_new']} new, {result['eas_updated']} updated, "
            f"{result['eas_unchanged']} unchanged, {result['manifests_created']} manifests created"
        )
        
        return result
    
    def _map_strategy_type(self, strategy_type_str: str) -> StrategyType:
        """Map strategy type string to StrategyType enum."""
        strategy_map = {
            'RSI Strategy': StrategyType.SCALPER,
            'MACD Strategy': StrategyType.SCALPER,
            'Moving Average Strategy': StrategyType.STRUCTURAL,
            'Bollinger Bands Strategy': StrategyType.STRUCTURAL,
            'Stochastic Strategy': StrategyType.SCALPER,
            'Fibonacci Strategy': StrategyType.STRUCTURAL,
            'Pivot Point Strategy': StrategyType.STRUCTURAL,
            'Breakout Strategy': StrategyType.STRUCTURAL,
            'Scalping Strategy': StrategyType.SCALPER,
            'Martingale Strategy': StrategyType.SCALPER,
            'Grid Strategy': StrategyType.STRUCTURAL,
            'Generic Strategy': StrategyType.STRUCTURAL,
        }
        return strategy_map.get(strategy_type_str, StrategyType.STRUCTURAL)
    
    def _map_complexity_to_frequency(self, complexity: str) -> TradeFrequency:
        """Map complexity string to TradeFrequency enum."""
        complexity_map = {
            'Simple': TradeFrequency.LOW,
            'Moderate': TradeFrequency.MEDIUM,
            'Complex': TradeFrequency.HIGH,
            'Very Complex': TradeFrequency.HFT,
        }
        return complexity_map.get(complexity, TradeFrequency.MEDIUM)


def get_github_ea_sync() -> GitHubEASync:
    """
    Get configured GitHubEASync instance from environment.
    
    Returns:
        Configured GitHubEASync instance
    """
    import os
    
    repo_url = os.getenv('GITHUB_EA_REPO_URL', '')
    library_path = os.getenv('EA_LIBRARY_PATH', '/data/ea-library')
    branch = os.getenv('GITHUB_EA_BRANCH', 'main')
    
    if not repo_url:
        raise ValueError("GITHUB_EA_REPO_URL environment variable not set")
    
    return GitHubEASync(
        repo_url=repo_url,
        library_path=library_path,
        branch=branch
    )


if __name__ == '__main__':
    # Test the sync service
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python github_ea_sync.py <repo_url> [library_path]")
        sys.exit(1)
    
    repo_url = sys.argv[1]
    library_path = sys.argv[2] if len(sys.argv) > 2 else '/data/ea-library'
    
    sync_service = GitHubEASync(repo_url, library_path)
    
    # Perform sync
    result = sync_service.sync_repository()
    print(f"Sync result: {result}")
    
    # Scan EAs
    ea_files = sync_service.scan_ea_files()
    print(f"\nFound {len(ea_files)} EA files:")
    
    for ea_file in ea_files[:5]:  # Show first 5
        ea_data = sync_service.parse_ea_file(ea_file)
        print(f"\n{ea_data['filename']}:")
        print(f"  Strategy: {ea_data.get('strategy_type', 'Unknown')}")
        print(f"  Inputs: {ea_data.get('input_count', 0)}")
        print(f"  Complexity: {ea_data.get('complexity', 'Unknown')}")