"""
QuantMind IDE API Endpoints

Provides endpoints for the UI shell to connect to backend services:
- Strategy folders (NPRD workspace)
- Shared assets library
- Knowledge hub
- NPRD processing
- Live trading control
- Agent chat
- Database export
- MT5 scanner and launcher
"""

import os
import sys
import json
import csv
import io
import logging
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, TYPE_CHECKING, Union
from enum import Enum

try:
    from src.api.websocket_endpoints import create_websocket_endpoints
except ImportError:
    create_websocket_endpoints = None

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = Path(os.getenv("QUANTMIND_DATA_DIR", "data"))
STRATEGIES_DIR = DATA_DIR / "strategies"
ASSETS_DIR = DATA_DIR / "shared_assets"
KNOWLEDGE_DIR = DATA_DIR / "knowledge"
SCRAPED_ARTICLES_DIR = DATA_DIR / "scraped_articles"  # MQL5 articles scraped from web


# =============================================================================
# Pydantic Models
# =============================================================================

try:
    from pydantic import BaseModel as PydanticBaseModel, Field as PydanticField
    BaseModel = PydanticBaseModel  # type: ignore
    Field = PydanticField  # type: ignore
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    # Fallback for when pydantic is not available
    class BaseModel:  # type: ignore
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    def Field(*args, **kwargs) -> Any:  # type: ignore
        """Fallback Field function that returns the default value or kwargs."""
        if len(args) > 0:
            return args[0]
        return kwargs.get('default')


class StrategyStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    PRIMAL = "primal"
    QUARANTINED = "quarantined"


class StrategyFolder(BaseModel):
    id: str
    name: str
    status: StrategyStatus = StrategyStatus.PENDING
    created_at: str
    has_nprd: bool = False
    has_trd: bool = False
    has_ea: bool = False
    has_backtest: bool = False


class StrategyDetail(BaseModel):
    id: str
    name: str
    status: StrategyStatus
    created_at: str
    nprd: Optional[Dict[str, Any]] = None
    trd: Optional[Dict[str, Any]] = None
    ea: Optional[Dict[str, Any]] = None
    backtests: List[Dict[str, Any]] = []


class SharedAsset(BaseModel):
    id: str
    name: str
    type: str  # indicator, library, template
    path: str
    description: Optional[str] = None
    used_in: List[str] = []


class KnowledgeItem(BaseModel):
    id: str
    name: str
    category: str  # articles, books, logs
    path: str
    size_bytes: int
    indexed: bool = False


class NPRDProcessRequest(BaseModel):
    url: str = Field(..., description="YouTube URL to process")
    strategy_name: str = Field(..., description="Name for the strategy folder")


class NPRDProcessResponse(BaseModel):
    job_id: str
    status: str
    strategy_folder: str


class BotControl(BaseModel):
    bot_id: str
    action: str  # pause, resume, quarantine, kill


class DatabaseExportRequest(BaseModel):
    """Request model for database export."""
    format: str = Field(default="csv", description="Export format: csv or json")
    limit: Optional[int] = Field(default=None, description="Maximum rows to export")


class MT5ScanRequest(BaseModel):
    """Request model for MT5 scan."""
    custom_paths: Optional[List[str]] = Field(default=None, description="Custom paths to scan")


class MT5LaunchRequest(BaseModel):
    """Request model for MT5 launch."""
    terminal_path: str = Field(..., description="Path to MT5 terminal executable")
    login: Optional[int] = Field(default=None, description="Account login number")
    password: Optional[str] = Field(default=None, description="Account password")
    server: Optional[str] = Field(default=None, description="Broker server name")


# =============================================================================
# API Handlers
# =============================================================================

class StrategyAPIHandler:
    """Handler for strategy folder operations."""
    
    def __init__(self):
        STRATEGIES_DIR.mkdir(parents=True, exist_ok=True)
    
    def list_strategies(self) -> List[StrategyFolder]:
        """List all strategy folders."""
        strategies = []
        
        if not STRATEGIES_DIR.exists():
            return strategies
        
        for folder in STRATEGIES_DIR.iterdir():
            if folder.is_dir() and not folder.name.startswith('.'):
                status_file = folder / "status.json"
                status = StrategyStatus.PENDING
                created_at = datetime.fromtimestamp(folder.stat().st_ctime).isoformat()
                
                if status_file.exists():
                    try:
                        with open(status_file) as f:
                            data = json.load(f)
                            status = StrategyStatus(data.get("status", "pending"))
                    except:
                        pass
                
                strategies.append(StrategyFolder(
                    id=folder.name,
                    name=folder.name.replace("_", " "),
                    status=status,
                    created_at=created_at,
                    has_nprd=(folder / "nprd").exists(),
                    has_trd=(folder / "trd").exists(),
                    has_ea=(folder / "ea").exists(),
                    has_backtest=(folder / "backtest").exists()
                ))
        
        return sorted(strategies, key=lambda x: x.created_at, reverse=True)
    
    def get_strategy(self, strategy_id: str) -> Optional[StrategyDetail]:
        """Get detailed strategy folder contents."""
        folder = STRATEGIES_DIR / strategy_id
        
        if not folder.exists():
            return None
        
        status_file = folder / "status.json"
        status = StrategyStatus.PENDING
        
        if status_file.exists():
            try:
                with open(status_file) as f:
                    data = json.load(f)
                    status = StrategyStatus(data.get("status", "pending"))
            except:
                pass
        
        # Gather folder contents
        nprd_data = None
        if (folder / "nprd").exists():
            nprd_data = {
                "files": [f.name for f in (folder / "nprd").iterdir() if f.is_file()]
            }
            metadata_file = folder / "nprd" / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file) as f:
                    nprd_data["metadata"] = json.load(f)
        
        trd_data = None
        if (folder / "trd").exists():
            trd_data = {
                "files": [f.name for f in (folder / "trd").iterdir() if f.is_file()]
            }
        
        ea_data = None
        if (folder / "ea").exists():
            ea_data = {
                "files": [f.name for f in (folder / "ea").iterdir() if f.is_file()]
            }
        
        backtests = []
        if (folder / "backtest").exists():
            for bt_file in (folder / "backtest").iterdir():
                if bt_file.suffix in ['.html', '.json']:
                    backtests.append({
                        "name": bt_file.name,
                        "path": str(bt_file),
                        "mode": bt_file.stem.split('_')[0] if '_' in bt_file.stem else "mode_a"
                    })
        
        return StrategyDetail(
            id=strategy_id,
            name=strategy_id.replace("_", " "),
            status=status,
            created_at=datetime.fromtimestamp(folder.stat().st_ctime).isoformat(),
            nprd=nprd_data,
            trd=trd_data,
            ea=ea_data,
            backtests=backtests
        )
    
    def create_strategy_folder(self, name: str) -> str:
        """Create a new strategy folder."""
        folder_name = name.replace(" ", "_")
        folder = STRATEGIES_DIR / folder_name
        
        # Create folder structure
        (folder / "nprd").mkdir(parents=True, exist_ok=True)
        (folder / "trd").mkdir(exist_ok=True)
        (folder / "ea").mkdir(exist_ok=True)
        (folder / "backtest").mkdir(exist_ok=True)
        
        # Create status file
        with open(folder / "status.json", "w") as f:
            json.dump({
                "status": "pending",
                "created_at": datetime.now().isoformat(),
                "tags": []
            }, f)
        
        return folder_name


class AssetsAPIHandler:
    """Handler for shared assets library."""
    
    def __init__(self):
        ASSETS_DIR.mkdir(parents=True, exist_ok=True)
        (ASSETS_DIR / "indicators").mkdir(exist_ok=True)
        (ASSETS_DIR / "libraries").mkdir(exist_ok=True)
        (ASSETS_DIR / "templates").mkdir(exist_ok=True)
    
    def list_assets(self, category: Optional[str] = None) -> List[SharedAsset]:
        """List all shared assets, optionally filtered by category."""
        assets = []
        
        categories = [category] if category else ["indicators", "libraries", "templates"]
        
        for cat in categories:
            cat_dir = ASSETS_DIR / cat
            if not cat_dir.exists():
                continue
            
            for asset_file in cat_dir.iterdir():
                if asset_file.is_file() and asset_file.suffix in ['.mqh', '.mq5', '.py']:
                    assets.append(SharedAsset(
                        id=f"{cat}/{asset_file.name}",
                        name=asset_file.stem,
                        type=cat.rstrip('s'),  # indicators -> indicator
                        path=str(asset_file),
                        description=self._extract_description(asset_file)
                    ))
        
        return assets
    
    def _extract_description(self, file_path: Path) -> Optional[str]:
        """Extract description from file header comment."""
        try:
            with open(file_path, 'r') as f:
                first_lines = f.read(500)
                # Look for comment block
                if '//' in first_lines:
                    for line in first_lines.split('\n')[:5]:
                        if line.strip().startswith('//'):
                            return line.strip('/ \n')
        except:
            pass
        return None
    
    def get_asset_content(self, asset_id: str) -> Optional[str]:
        """Get the content of an asset file."""
        asset_path = ASSETS_DIR / asset_id
        if asset_path.exists():
            return asset_path.read_text()
        return None


class KnowledgeAPIHandler:
    """Handler for knowledge hub (PageIndex integration)."""
    
    def __init__(self):
        KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
        (KNOWLEDGE_DIR / "articles").mkdir(exist_ok=True)
        (KNOWLEDGE_DIR / "books").mkdir(exist_ok=True)
        (KNOWLEDGE_DIR / "logs").mkdir(exist_ok=True)
    
    def list_knowledge(self, category: Optional[str] = None) -> List[KnowledgeItem]:
        """List knowledge items including scraped articles."""
        items = []
        
        categories = [category] if category else ["articles", "books", "logs"]
        
        # Scan knowledge directory
        for cat in categories:
            cat_dir = KNOWLEDGE_DIR / cat
            if not cat_dir.exists():
                continue
            
            for item in cat_dir.iterdir():
                if item.is_file():
                    items.append(KnowledgeItem(
                        id=f"{cat}/{item.name}",
                        name=item.stem,
                        category=cat,
                        path=str(item),
                        size_bytes=item.stat().st_size,
                        indexed=self._is_indexed(item)
                    ))
        
        # Include scraped articles (1,806 MQL5 articles)
        if category in (None, "articles"):
            items.extend(self._scan_scraped_articles())
        
        return items
    
    def _scan_scraped_articles(self) -> List[KnowledgeItem]:
        """Scan scraped_articles directory structure."""
        items = []
        
        if not SCRAPED_ARTICLES_DIR.exists():
            return items
        
        # Scan subdirectories: expert_advisors, integration, trading, trading_systems
        for subcategory in ["expert_advisors", "integration", "trading", "trading_systems"]:
            subdir = SCRAPED_ARTICLES_DIR / subcategory
            if not subdir.exists():
                continue
            
            for item in subdir.glob("*.md"):
                # Create readable name from filename
                readable_name = item.stem.replace('_', ' ').title()
                
                items.append(KnowledgeItem(
                    id=f"scraped/{subcategory}/{item.name}",
                    name=readable_name,
                    category=f"articles/{subcategory}",
                    path=str(item),
                    size_bytes=item.stat().st_size,
                    indexed=False
                ))
        
        return items
    
    def _is_indexed(self, file_path: Path) -> bool:
        """Check if file is indexed in PageIndex."""
        # TODO: Query PageIndex API
        return False
    
    def get_content(self, item_id: str) -> Optional[str]:
        """Get content of a knowledge item."""
        # Handle scraped articles
        if item_id.startswith("scraped/"):
            article_path = SCRAPED_ARTICLES_DIR / item_id.replace("scraped/", "")
            if article_path.exists():
                try:
                    return article_path.read_text(encoding='utf-8')
                except UnicodeDecodeError:
                    return article_path.read_text(encoding='latin-1')
            return None
        
        # Original knowledge directory items
        item_path = KNOWLEDGE_DIR / item_id

        if not item_path.exists():
            return None

        # Handle markdown and text files
        if item_path.suffix in ['.md', '.txt', '.json', '.yaml', '.yml']:
            try:
                return item_path.read_text(encoding='utf-8')
            except UnicodeDecodeError:
                return item_path.read_text(encoding='latin-1')

        # Handle PDF files - extract text content
        elif item_path.suffix == '.pdf':
            return self._extract_pdf_content(item_path)

        # Handle CSV files
        elif item_path.suffix == '.csv':
            try:
                return item_path.read_text(encoding='utf-8')
            except UnicodeDecodeError:
                return item_path.read_text(encoding='latin-1')

        return None

    def _extract_pdf_content(self, pdf_path: Path) -> Optional[str]:
        """Extract text content from PDF file."""
        try:
            # Try pdfplumber first (better accuracy)
            import pdfplumber  # type: ignore
            text_parts = []
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
            return '\n\n'.join(text_parts) if text_parts else None
        except ImportError:
            # Fallback to PyPDF2
            try:
                import PyPDF2  # type: ignore
                text_parts = []
                with open(pdf_path, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    for page in pdf_reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text_parts.append(page_text)
                return '\n\n'.join(text_parts) if text_parts else None
            except ImportError:
                logger.warning("Neither pdfplumber nor PyPDF2 installed. Cannot extract PDF content.")
                return None
            except Exception as e:
                logger.error(f"Failed to extract PDF content with PyPDF2: {e}")
                return None
        except Exception as e:
            logger.error(f"Failed to extract PDF content: {e}")
            return None


class NPRDAPIHandler:
    """Handler for NPRD processing."""
    
    def __init__(self):
        self.jobs: Dict[str, Dict] = {}
    
    def process(self, request: NPRDProcessRequest) -> NPRDProcessResponse:
        """Start NPRD processing for a YouTube URL."""
        import uuid
        
        job_id = str(uuid.uuid4())
        strategy_handler = StrategyAPIHandler()
        folder_name = strategy_handler.create_strategy_folder(request.strategy_name)
        
        # Store job info
        self.jobs[job_id] = {
            "status": "queued",
            "url": request.url,
            "strategy_folder": folder_name,
            "created_at": datetime.now().isoformat(),
            "progress": 0
        }
        
        # TODO: Trigger actual NPRD processing pipeline
        # This would call the Gemini CLI processor
        
        return NPRDProcessResponse(
            job_id=job_id,
            status="queued",
            strategy_folder=folder_name
        )
    
    def get_job_status(self, job_id: str) -> Optional[Dict]:
        """Get NPRD job status."""
        return self.jobs.get(job_id)


class LiveTradingAPIHandler:
    """Handler for live trading control."""
    
    def get_active_bots(self) -> List[Dict]:
        """Get list of active bots from BotManifest."""
        try:
            from src.router.bot_manifest import BotRegistry
            registry = BotRegistry()

            bots = []
            # Use list_all() to get all registered bots
            for manifest in registry.list_all():
                bots.append({
                    "id": manifest.bot_id,
                    "name": manifest.name or manifest.bot_id,
                    "strategy_type": manifest.strategy_type.value if hasattr(manifest.strategy_type, 'value') else str(manifest.strategy_type),
                    "frequency": manifest.frequency.value if hasattr(manifest.frequency, 'value') else str(manifest.frequency),
                    "symbols": manifest.symbols,
                    "timeframes": manifest.timeframes,
                    "prop_firm_safe": manifest.prop_firm_safe,
                    "total_trades": manifest.total_trades,
                    "win_rate": manifest.win_rate
                })
            return bots
        except Exception as e:
            logger.error(f"Failed to get bots: {e}")
            return []
    
    def control_bot(self, control: BotControl) -> Dict:
        """Control a bot (pause, resume, quarantine, kill)."""
        try:
            from src.router.bot_manifest import BotRegistry
            registry = BotRegistry()

            # Get the bot manifest to verify it exists
            manifest = registry.get(control.bot_id)
            if not manifest:
                return {"success": False, "error": f"Bot {control.bot_id} not found"}

            # TODO: Implement actual bot control logic
            # For now, just log the action
            logger.info(f"Bot control action: {control.action} on bot {control.bot_id}")

            return {"success": True, "bot_id": control.bot_id, "action": control.action}
        except Exception as e:
            logger.error(f"Bot control failed: {e}")
            return {"success": False, "error": str(e)}
    
    def get_system_status(self) -> Dict:
        """Get overall system status."""
        try:
            from src.router.sentinel import get_current_regime
            from src.risk.governor import RiskGovernor
            
            regime = "Unknown"
            kelly = 0.0
            
            try:
                regime = get_current_regime()
            except:
                pass
            
            try:
                gov = RiskGovernor()
                kelly = gov.get_allocation_scalar()
            except:
                pass
            
            return {
                "connected": True,
                "regime": regime,
                "kelly": kelly,
                "active_bots": len(self.get_active_bots()),
                "pnl_today": 0.0  # TODO: Calculate from trade journal
            }
        except Exception as e:
            logger.error(f"Status check failed: {e}")
            return {"connected": False, "error": str(e)}


# =============================================================================
# FastAPI Application Factory
# =============================================================================

def create_ide_api_app():
    """Create FastAPI app with all IDE endpoints."""
    try:
        from fastapi import FastAPI, HTTPException, UploadFile, File, Form
        from fastapi.middleware.cors import CORSMiddleware
    except ImportError:
        logger.warning("FastAPI not available")
        return None
    
    # Import Routers
    try:
        from src.api.chat_endpoints import router as chat_router
    except ImportError:
        logger.warning("Chat endpoints not available")
        chat_router = None
    
    app = FastAPI(
        title="QuantMind IDE API",
        description="API for QuantMind IDE frontend",
        version="1.0.0"
    )
    
    # Register Routers
    if chat_router:
        app.include_router(chat_router)
    
    # CORS for Tauri
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["tauri://localhost", "http://localhost:1420", "http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"]
    )
    
    # Register WebSocket endpoints
    if create_websocket_endpoints:
        create_websocket_endpoints(app)
    
    # Initialize handlers
    strategy_handler = StrategyAPIHandler()
    assets_handler = AssetsAPIHandler()
    knowledge_handler = KnowledgeAPIHandler()
    nprd_handler = NPRDAPIHandler()
    trading_handler = LiveTradingAPIHandler()
    
    # -------------------------------------------------------------------------
    # Strategy Endpoints
    # -------------------------------------------------------------------------
    
    @app.get("/api/strategies")
    async def list_strategies():
        """List all strategy folders."""
        return strategy_handler.list_strategies()
    
    @app.get("/api/strategies/{strategy_id}")
    async def get_strategy(strategy_id: str):
        """Get strategy folder details."""
        result = strategy_handler.get_strategy(strategy_id)
        if not result:
            raise HTTPException(404, "Strategy not found")
        return result
    
    @app.post("/api/strategies")
    async def create_strategy(name: str):
        """Create a new strategy folder."""
        folder_name = strategy_handler.create_strategy_folder(name)
        return {"id": folder_name, "name": name}
    
    # -------------------------------------------------------------------------
    # Shared Assets Endpoints
    # -------------------------------------------------------------------------
    
    @app.get("/api/assets")
    async def list_assets(category: Optional[str] = None):
        """List shared assets."""
        return assets_handler.list_assets(category)
    
    @app.get("/api/assets/{asset_id:path}/content")
    async def get_asset_content(asset_id: str):
        """Get asset file content."""
        content = assets_handler.get_asset_content(asset_id)
        if content is None:
            raise HTTPException(404, "Asset not found")
        return {"content": content}
    
    # -------------------------------------------------------------------------
    # Knowledge Hub Endpoints
    # -------------------------------------------------------------------------
    
    @app.get("/api/knowledge")
    async def list_knowledge(category: Optional[str] = None):
        """List knowledge items."""
        return knowledge_handler.list_knowledge(category)
    
    @app.get("/api/knowledge/{item_id:path}/content")
    async def get_knowledge_content(item_id: str):
        """Get knowledge item content."""
        content = knowledge_handler.get_content(item_id)
        if content is None:
            raise HTTPException(404, "Item not found")
        return {"content": content}
    
    @app.post("/api/ide/knowledge/upload")
    async def upload_knowledge_file(
        file: UploadFile = File(None),
        category: str = Form("books"),
        title: str = Form(None),
        author: str = Form(None),
        url: str = Form(None),
        subcategory: str = Form(None),
        index: str = Form("false"),
        content: str = Form(None)
    ):
        """
        Upload file or note to knowledge base.

        Args:
            file: The file to upload (PDF, MD, TXT, CSV, JSON) - optional for notes
            category: Target category (books, articles, notes)
            title: Optional title for the item
            author: Optional author name
            url: Optional source URL (for articles)
            subcategory: Optional subcategory for better organization
            index: Whether to index the file (for books)
            content: Content for notes (no file)
        """
        try:
            # Validate category
            valid_categories = ["books", "articles", "notes"]
            if category not in valid_categories:
                category = "books"

            # Create category directory if it doesn't exist
            category_dir = KNOWLEDGE_DIR / category
            category_dir.mkdir(parents=True, exist_ok=True)

            # Handle note submission (no file)
            if category == "notes" and content and not file:
                # Create a markdown file from the note content
                safe_title = "".join(c for c in (title or "note") if c.isalnum() or c in (' ', '-', '_')).strip()
                filename = f"{safe_title}.md"
                save_path = category_dir / filename

                # Add metadata header to markdown
                metadata_header = "---\n"
                if title:
                    metadata_header += f"title: {title}\n"
                metadata_header += f"created_at: {datetime.now().isoformat()}\n"
                metadata_header += "---\n\n"

                note_content = metadata_header + content
                with open(save_path, "w", encoding="utf-8") as f:
                    f.write(note_content)

                logger.info(f"Created note: {title} in {category}")

                return {
                    "status": "success",
                    "filename": filename,
                    "title": title,
                    "path": str(save_path),
                    "category": category,
                    "size_bytes": len(note_content.encode('utf-8')),
                    "indexed": False
                }

            # Handle file upload
            if not file:
                raise HTTPException(400, "File is required for books and articles")

            # For books/articles, save the file with optional metadata
            if not file.filename:
                raise HTTPException(400, "Filename is required")
            save_path = category_dir / file.filename

            # Read and save content
            content_bytes = await file.read()
            with open(save_path, "wb") as f:
                f.write(content_bytes)

            # Save metadata separately if provided
            metadata = {
                "filename": file.filename,
                "category": category,
                "uploaded_at": datetime.now().isoformat()
            }
            if title:
                metadata["title"] = title
            if author:
                metadata["author"] = author
            if url:
                metadata["url"] = url
            if subcategory:
                metadata["subcategory"] = subcategory

            # Save metadata as JSON file alongside the content
            metadata_path = category_dir / f"{file.filename}.metadata.json"
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Uploaded {file.filename} to {category}")

            # TODO: Trigger PageIndex indexing for books if index == "true"
            should_index = index.lower() == "true" and category == "books"

            return {
                "status": "success",
                "filename": file.filename,
                "path": str(save_path),
                "category": category,
                "size_bytes": len(content_bytes),
                "indexed": False,  # Will be updated when indexing is implemented
                "will_index": should_index,
                "metadata": metadata
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            raise HTTPException(500, f"Upload failed: {str(e)}")

    @app.post("/api/ide/knowledge/upload/note")
    async def upload_note(
        title: str = Form(...),
        content: str = Form(...),
        category: str = Form("notes")
    ):
        """
        Create a note in the knowledge base (JSON API).

        Args:
            title: Note title
            content: Note content (markdown supported)
            category: Category (should be 'notes')
        """
        try:
            # Validate category
            if category not in ["notes"]:
                category = "notes"

            # Create category directory if it doesn't exist
            category_dir = KNOWLEDGE_DIR / category
            category_dir.mkdir(parents=True, exist_ok=True)

            # Create a markdown file from the note content
            safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).strip()
            filename = f"{safe_title}.md"
            save_path = category_dir / filename

            # Add metadata header to markdown
            metadata_header = "---\n"
            metadata_header += f"title: {title}\n"
            metadata_header += f"created_at: {datetime.now().isoformat()}\n"
            metadata_header += "---\n\n"

            note_content = metadata_header + content
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(note_content)

            logger.info(f"Created note: {title}")

            return {
                "status": "success",
                "filename": filename,
                "title": title,
                "path": str(save_path),
                "category": category,
                "size_bytes": len(note_content.encode('utf-8')),
                "indexed": False
            }
        except Exception as e:
            logger.error(f"Note creation failed: {e}")
            raise HTTPException(500, f"Note creation failed: {str(e)}")
    
    # -------------------------------------------------------------------------
    # NPRD Processing Endpoints
    # -------------------------------------------------------------------------
    
    @app.post("/api/nprd/process")
    async def process_nprd(request: NPRDProcessRequest):
        """Start NPRD processing for a YouTube URL."""
        return nprd_handler.process(request)
    
    @app.get("/api/nprd/jobs/{job_id}")
    async def get_nprd_job(job_id: str):
        """Get NPRD job status."""
        result = nprd_handler.get_job_status(job_id)
        if not result:
            raise HTTPException(404, "Job not found")
        return result
    
    # -------------------------------------------------------------------------
    # Live Trading Endpoints
    # -------------------------------------------------------------------------
    
    @app.get("/api/trading/bots")
    async def get_bots():
        """Get active bots."""
        return trading_handler.get_active_bots()
    
    @app.post("/api/trading/bots/control")
    async def control_bot(control: BotControl):
        """Control a bot."""
        return trading_handler.control_bot(control)
    
    @app.get("/api/trading/status")
    async def get_trading_status():
        """Get system trading status."""
        return trading_handler.get_system_status()
    
    @app.post("/api/trading/kill")
    async def kill_all():
        """Emergency kill all bots."""
        try:
            from src.router.kill_switch import KillSwitch, KillReason
            ks = KillSwitch()
            await ks.trigger(
                reason=KillReason.API_COMMAND,
                triggered_by="api_user",
                message="Emergency kill triggered via API"
            )
            return {"success": True, "message": "Kill switch triggered"}
        except Exception as e:
            raise HTTPException(500, str(e))
    
    # -------------------------------------------------------------------------
    # Agent Chat Endpoint
    # -------------------------------------------------------------------------
    
    @app.post("/api/chat")
    async def chat(request: dict):
        """Send message to agent and get response."""
        message = request.get("message", "")
        agent = request.get("agent", "copilot")
        model = request.get("model", "gemini-2.5-pro")
        context = request.get("context", [])
        
        # TODO: Connect to actual LangGraph agent
        # For now, return demo responses
        response = f"I understand you want to: {message[:50]}... I'll help you with that."
        
        if "backtest" in message.lower():
            response = "I can run backtests in 4 variants. Which strategy would you like to test?"
        elif "nprd" in message.lower():
            response = "To process NPRD: 1. Click Process NPRD in EA Management 2. Paste YouTube URL 3. The system will transcribe and analyze."
        elif "bot" in message.lower() or "active" in message.lower():
            bots = trading_handler.get_active_bots()
            response = f"You have {len(bots)} active bots. Go to Live Trading to manage them."
        
        return {"response": response, "agent": agent, "model": model}
    
    # -------------------------------------------------------------------------
    # File Content Endpoint
    # -------------------------------------------------------------------------
    
    @app.get("/api/files/content")
    async def get_file_content(path: str):
        """Get content of any file."""
        from pathlib import Path as P
        file_path = P(path)
        
        if not file_path.exists():
            raise HTTPException(404, "File not found")
        
        if file_path.suffix in ['.mq5', '.mqh', '.py', '.md', '.txt', '.json']:
            return {"content": file_path.read_text()}
        else:
            return {"content": f"Binary file: {file_path.name}", "binary": True}
    
    # -------------------------------------------------------------------------
    # EA Management Endpoints
    # -------------------------------------------------------------------------

    @app.get("/api/ide/eas")
    async def list_eas():
        """List all Expert Advisors (EAs)."""
        # Mock EA data - in production, scan EA folders and MT5
        eas = [
            {
                "id": "ea_ict_scalper_v2",
                "name": "ICT Scalper v2",
                "symbol": "EURUSD",
                "timeframe": "M15",
                "status": "running",
                "deployed": True,
                "profit": 245.80,
                "trades": 12,
                "win_rate": 75.0,
                "created_at": "2026-02-01T10:00:00Z",
                "modified_at": "2026-02-09T15:30:00Z"
            },
            {
                "id": "ea_smc_reversal",
                "name": "SMC Reversal",
                "symbol": "GBPUSD",
                "timeframe": "H1",
                "status": "stopped",
                "deployed": False,
                "profit": 128.50,
                "trades": 8,
                "win_rate": 62.5,
                "created_at": "2026-02-05T14:00:00Z",
                "modified_at": "2026-02-08T09:15:00Z"
            },
            {
                "id": "ea_breakthrough_eur",
                "name": "Breakthrough EA",
                "symbol": "EURUSD",
                "timeframe": "M5",
                "status": "testing",
                "deployed": False,
                "profit": 0.0,
                "trades": 0,
                "win_rate": 0.0,
                "created_at": "2026-02-08T16:00:00Z",
                "modified_at": "2026-02-09T12:00:00Z"
            }
        ]
        return {"eas": eas, "total": len(eas)}

    @app.get("/api/ide/eas/{ea_id}")
    async def get_ea_details(ea_id: str):
        """Get detailed information about a specific EA."""
        # Mock EA details
        ea_details = {
            "id": ea_id,
            "name": "ICT Scalper v2",
            "symbol": "EURUSD",
            "timeframe": "M15",
            "status": "running",
            "deployed": True,
            "config": {
                "lot_size": 0.01,
                "max_spread": 2.0,
                "trading_hours": {"start": "08:00", "end": "17:00"},
                "risk_mode": "kelly",
                "kelly_fraction": 0.025
            },
            "performance": {
                "profit": 245.80,
                "trades": 12,
                "win_rate": 75.0,
                "max_drawdown": 15.20,
                "profit_factor": 2.1
            },
            "recent_trades": [
                {"id": "t1", "type": "BUY", "profit": 18.0, "time": "2026-02-09T15:20:00Z"},
                {"id": "t2", "type": "SELL", "profit": -9.6, "time": "2026-02-09T14:20:00Z"}
            ]
        }
        return ea_details

    @app.post("/api/ide/eas/{ea_id}/deploy")
    async def deploy_ea(ea_id: str):
        """Deploy EA to MT5 terminal."""
        # TODO: Implement actual MT5 deployment
        return {
            "success": True,
            "ea_id": ea_id,
            "status": "deployed",
            "message": f"EA {ea_id} deployed successfully"
        }

    @app.post("/api/ide/eas/{ea_id}/undeploy")
    async def undeploy_ea(ea_id: str):
        """Undeploy EA from MT5 terminal."""
        # TODO: Implement actual MT5 undeployment
        return {
            "success": True,
            "ea_id": ea_id,
            "status": "stopped",
            "message": f"EA {ea_id} undeployed successfully"
        }

    @app.get("/api/ide/mt5/status")
    async def get_mt5_status():
        """Get MT5 terminal connection status."""
        return {
            "connected": True,
            "terminal": "MetaTrader 5",
            "account": "12345678",
            "broker": "Demo Broker",
            "balance": 10000.00,
            "equity": 10245.80,
            "margin": 150.25,
            "free_margin": 10095.55,
            "currency": "USD"
        }

    # -------------------------------------------------------------------------
    # Backtest V1 Endpoints
    # -------------------------------------------------------------------------

    # Backtest session storage
    _backtest_sessions: Dict[str, Dict[str, Any]] = {}
    _backtest_results: Dict[str, Dict[str, Any]] = {}

    class BacktestRunRequest(BaseModel):
        """Request model for backtest run."""
        symbol: str = Field(..., description="Trading symbol (e.g., EURUSD)")
        timeframe: str = Field(..., description="Timeframe (e.g., H1, M15)")
        variant: str = Field(..., description="Variant: vanilla, spiced, vanilla_full, spiced_full")
        start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
        end_date: str = Field(..., description="End date (YYYY-MM-DD)")
        strategy_code: Optional[str] = Field(None, description="Optional Python strategy code")
        strategy_name: Optional[str] = Field("MyStrategy", description="Strategy name for logging")
        initial_cash: Optional[float] = Field(10000.0, description="Initial account balance")
        commission: Optional[float] = Field(0.001, description="Commission per trade")
        broker_id: Optional[str] = Field("icmarkets_raw", description="Broker ID for fee-aware Kelly")
        enable_ws_streaming: Optional[bool] = Field(True, description="Enable WebSocket streaming for real-time updates")

    @app.post("/api/v1/backtest/run")
    async def run_backtest(request: BacktestRunRequest):
        """
        Run a backtest with the specified parameters.

        Supports 4 backtest variants:
        - vanilla: Historical backtest with static parameters
        - spiced: Vanilla + regime filtering (skip HIGH_CHAOS, NEWS_EVENT)
        - vanilla_full: Vanilla + Walk-Forward optimization
        - spiced_full: Spiced + Walk-Forward optimization

        Uses fee-aware Kelly position sizing when broker_id is provided.
        """
        import uuid
        import asyncio
        from datetime import datetime
        from src.backtesting.mode_runner import run_full_system_backtest, BacktestMode
        from src.backtesting.mt5_engine import MQL5Timeframe
        import pandas as pd
        from src.api.ws_logger import setup_backtest_logging

        # Generate unique backtest ID
        backtest_id = str(uuid.uuid4())

        # Map timeframe string to MQL5 constant
        timeframe_map = {
            "M1": MQL5Timeframe.PERIOD_M1,
            "M5": MQL5Timeframe.PERIOD_M5,
            "M15": MQL5Timeframe.PERIOD_M15,
            "M30": MQL5Timeframe.PERIOD_M30,
            "H1": MQL5Timeframe.PERIOD_H1,
            "H4": MQL5Timeframe.PERIOD_H4,
            "D1": MQL5Timeframe.PERIOD_D1
        }
        timeframe_int = timeframe_map.get(request.timeframe, MQL5Timeframe.PERIOD_H1)

        # Default strategy code if not provided
        if request.strategy_code is None:
            request.strategy_code = '''
def on_bar(tester):
    """Simple moving average crossover strategy for backtest demo."""
    # Get current close and previous close
    symbol = tester.symbol
    if not symbol:
        return

    close = tester.iClose(symbol, 0, 0)
    if close is None:
        return

    prev_close = tester.iClose(symbol, 0, 1)
    if prev_close is None:
        return

    # Simple moving average values
    if tester.iMA(symbol, 0, 0, 20, 0) is None:
        return

    ma_fast = tester.iMA(symbol, 0, 0, 20, 0)
    ma_slow = tester.iMA(symbol, 0, 0, 50, 0)

    if ma_fast is None or ma_slow is None:
        return

    # Simple crossover logic with position sizing
    tester.equity  # Access current equity
    if ma_fast > ma_slow and prev_close <= ma_fast:
        # Buy signal - use Kelly position sizing
        tester.buy(symbol, 0.1)
    elif ma_fast < ma_slow and prev_close >= ma_fast:
        # Sell signal
        tester.sell(symbol, 0.1)
'''

        # Create mock data for demo (in production, fetch from MT5 or database)
        import numpy as np
        date_range = pd.date_range(
            start=request.start_date,
            end=request.end_date,
            freq='H' if 'H' in request.timeframe else 'D'
        )
        data = pd.DataFrame({
            'time': date_range,
            'open': np.random.uniform(1.08, 1.10, len(date_range)),
            'high': np.random.uniform(1.08, 1.11, len(date_range)),
            'low': np.random.uniform(1.07, 1.10, len(date_range)),
            'close': np.random.uniform(1.08, 1.10, len(date_range)),
            'volume': np.random.randint(1000, 10000, len(date_range))
        })

        # Store backtest session
        _backtest_sessions[backtest_id] = {
            "status": "running",
            "backtest_id": backtest_id,
            "symbol": request.symbol,
            "timeframe": request.timeframe,
            "variant": request.variant,
            "start_date": request.start_date,
            "end_date": request.end_date,
            "created_at": datetime.now().isoformat()
        }

        # Get the running event loop for thread-safe WebSocket broadcasts (Comment 1)
        loop = asyncio.get_running_loop()
        
        # Only setup WebSocket logging if enable_ws_streaming is True (Comment 2 fix)
        if request.enable_ws_streaming:
            # Single shared streamer (passed to tester - no duplicates)
            ws_logger, progress_streamer = setup_backtest_logging(backtest_id, loop=loop)
        else:
            ws_logger = None
            progress_streamer = None

        async def run_backtest_task():
            try:
                # Offload sync CPU-heavy backtest to thread pool (unblocks event loop)
                result = await loop.run_in_executor(
                    None,
                    lambda: run_full_system_backtest(
                        mode=request.variant,
                        data=data,
                        symbol=request.symbol,
                        timeframe=timeframe_int,
                        strategy_code=request.strategy_code or "",
                        initial_cash=request.initial_cash if request.initial_cash is not None else 10000.0,
                        commission=request.commission if request.commission is not None else 0.001,
                        broker_id=request.broker_id if request.broker_id is not None else "icmarkets_raw",
                        backtest_id=backtest_id,
                        progress_streamer=progress_streamer,  # Only set if enable_ws_streaming is True
                        ws_logger=ws_logger,  # Comment 2: Pass ws_logger for log_entry streaming
                        enable_ws_streaming=request.enable_ws_streaming or False,  # Pass the flag to control streaming (ensure bool)
                        loop=loop  # Pass FastAPI event loop for thread-safe WS broadcasts (Comment 1)
                    )
                )

                # Handle result
                if isinstance(result, dict):
                    first_key = next(iter(result))
                    backtest_result = result[first_key]
                else:
                    backtest_result = result

                # Store results (tester broadcasts complete via shared streamer)
                _backtest_results[backtest_id] = {
                    "backtest_id": backtest_id,
                    "final_balance": backtest_result.final_cash,
                    "total_trades": backtest_result.trades,
                    "win_rate": getattr(backtest_result, 'win_rate', None),
                    "sharpe_ratio": backtest_result.sharpe,
                    "drawdown": backtest_result.drawdown,
                    "return_pct": backtest_result.return_pct,
                    "duration_seconds": None,
                    "results": backtest_result.to_dict() if hasattr(backtest_result, 'to_dict') else {}
                }

                _backtest_sessions[backtest_id]["status"] = "completed"
                _backtest_sessions[backtest_id]["completed_at"] = datetime.now().isoformat()

                logger.info(f"Backtest {backtest_id} completed")

            except Exception as e:
                logger.error(f"Backtest {backtest_id} failed: {e}")
                _backtest_sessions[backtest_id]["status"] = "error"
                _backtest_sessions[backtest_id]["error"] = str(e)
                # Tester handles error broadcast via shared streamer

        asyncio.create_task(run_backtest_task())

        return {
            "backtest_id": backtest_id,
            "status": "running",
            "message": f"Backtest {backtest_id} started"
        }

    @app.get("/api/v1/backtest/status/{backtest_id}")
    async def get_backtest_status(backtest_id: str):
        """
        Get the status of a running or completed backtest.

        Returns the current status including progress if running.
        """
        session = _backtest_sessions.get(backtest_id)

        if session is None:
            raise HTTPException(404, f"Backtest {backtest_id} not found")

        return {
            "backtest_id": backtest_id,
            "status": session.get("status", "unknown"),
            "progress": 100 if session.get("status") == "completed" else 0
        }

    @app.get("/api/v1/backtest/results/{backtest_id}")
    async def get_backtest_results(backtest_id: str):
        """
        Get the results of a completed backtest.

        Returns detailed metrics including final balance, trades, win rate, etc.
        """
        results = _backtest_results.get(backtest_id)

        if results is None:
            # Check if session exists but hasn't completed
            session = _backtest_sessions.get(backtest_id)
            if session is None:
                raise HTTPException(404, f"Backtest {backtest_id} not found")
            elif session.get("status") != "completed":
                raise HTTPException(400, f"Backtest {backtest_id} has not completed yet")
            else:
                raise HTTPException(404, f"Results not found for backtest {backtest_id}")

        return results

    # -------------------------------------------------------------------------
    # Multi-Timeframe Regime Endpoints (Comment 4)
    # -------------------------------------------------------------------------

    # Module-level storage for shared MultiTimeframeSentinel instance
    _mtf_sentinel_instance = {}

    def _get_multi_timeframe_sentinel():
        """Get or create the shared MultiTimeframeSentinel instance."""
        sentinel_key = "sentinel"
        if sentinel_key not in _mtf_sentinel_instance:
            try:
                from src.router.multi_timeframe_sentinel import MultiTimeframeSentinel, Timeframe
                _mtf_sentinel_instance[sentinel_key] = MultiTimeframeSentinel(timeframes=[Timeframe.M5, Timeframe.H1, Timeframe.H4])
            except Exception as e:
                logger.warning(f"Could not create MultiTimeframeSentinel: {e}")
                _mtf_sentinel_instance[sentinel_key] = None
        return _mtf_sentinel_instance.get(sentinel_key)

    @app.get("/api/trading/timeframes/regimes")
    async def get_all_timeframe_regimes():
        """
        Get all timeframe regimes from the MultiTimeframeSentinel.
        
        Returns dictionary of timeframe -> regime report for all tracked timeframes.
        """
        sentinel = _get_multi_timeframe_sentinel()
        if sentinel is None:
            raise HTTPException(503, "Multi-timeframe sentinel not available")
        
        all_regimes = sentinel.get_all_regimes()
        return {
            "regimes": {
                tf.name: {
                    "regime": report.regime,
                    "chaos_score": report.chaos_score,
                    "regime_quality": report.regime_quality,
                    "susceptibility": report.susceptibility,
                    "news_state": report.news_state,
                }
                for tf, report in all_regimes.items()
            },
            "dominant_regime": sentinel.get_dominant_regime()
        }

    @app.get("/api/trading/timeframes/dominant")
    async def get_dominant_regime():
        """
        Get the dominant regime across all timeframes.
        
        Returns the dominant regime string using voting logic.
        """
        sentinel = _get_multi_timeframe_sentinel()
        if sentinel is None:
            raise HTTPException(503, "Multi-timeframe sentinel not available")
        
        return {
            "dominant_regime": sentinel.get_dominant_regime()
        }

    @app.post("/api/trading/timeframes/tick")
    async def process_timeframe_tick(request: dict):
        """
        Process a tick through the MultiTimeframeSentinel.
        
        Accepts symbol, price, and optional timestamp.
        """
        sentinel = _get_multi_timeframe_sentinel()
        if sentinel is None:
            raise HTTPException(503, "Multi-timeframe sentinel not available")
        
        symbol = request.get("symbol", "EURUSD")
        price = request.get("price")
        if price is None:
            raise HTTPException(400, "price is required")
        
        from datetime import datetime, timezone
        timestamp = request.get("timestamp")
        if timestamp:
            try:
                timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except ValueError:
                timestamp = datetime.now(timezone.utc)
        else:
            timestamp = datetime.now(timezone.utc)
        
        # Process tick
        updated_regimes = sentinel.on_tick(symbol, price, timestamp)
        
        return {
            "symbol": symbol,
            "price": price,
            "timestamp": timestamp.isoformat(),
            "updated_timeframes": [tf.name for tf in updated_regimes.keys()],
            "dominant_regime": sentinel.get_dominant_regime()
        }

    # -------------------------------------------------------------------------
    # Health Check
    # -------------------------------------------------------------------------

    @app.get("/health")
    async def health():
        return {"status": "healthy", "service": "QuantMind IDE API"}

    return app


# =============================================================================
# Run standalone
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    app = create_ide_api_app()
    if app:
        uvicorn.run(app, host="0.0.0.0", port=8000)
