"""
QuantMindX Shared Assets Library MCP Tools
Provides tools for loading, searching, and managing shared trading assets
"""

import json
import os
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

@dataclass
class AssetInfo:
    """Information about a shared asset"""
    id: str
    name: str
    version: str
    file_path: str
    description: str
    tags: List[str]
    parameters: List[str]
    category: str
    created: str

class AssetLibraryManager:
    """Manager for QuantMindX shared assets library"""
    
    def __init__(self, assets_root: str = "data/assets"):
        self.assets_root = Path(assets_root)
        self.registry_path = self.assets_root / "registry.json"
        self.timeframes_path = self.assets_root / "timeframes.yaml"
        self._load_registry()
        self._load_timeframes()
    
    def _load_registry(self) -> None:
        """Load asset registry from JSON file"""
        if not self.registry_path.exists():
            raise FileNotFoundError(f"Asset registry not found at {self.registry_path}")
        
        with open(self.registry_path, 'r') as f:
            self.registry = json.load(f)
    
    def _load_timeframes(self) -> None:
        """Load timeframe configuration from YAML file"""
        if not self.timeframes_path.exists():
            raise FileNotFoundError(f"Timeframes config not found at {self.timeframes_path}")
        
        with open(self.timeframes_path, 'r') as f:
            self.timeframes = yaml.safe_load(f)
    
    def search_assets(self, query: str, category: Optional[str] = None) -> List[AssetInfo]:
        """
        Search for assets by name, description, or tags
        
        Args:
            query: Search term to match against asset fields
            category: Optional category filter (indicators, strategies, etc.)
            
        Returns:
            List of matching AssetInfo objects
        """
        query = query.lower()
        results = []
        
        categories_to_search = [category] if category else self.registry.get("categories", {}).keys()
        
        for cat_name in categories_to_search:
            category_data = self.registry.get("categories", {}).get(cat_name, {})
            assets = category_data.get("assets", [])
            
            for asset_data in assets:
                # Check if query matches any searchable field
                searchable_text = " ".join([
                    asset_data.get("name", "").lower(),
                    asset_data.get("description", "").lower(),
                    " ".join(asset_data.get("tags", [])).lower()
                ])
                
                if query in searchable_text:
                    asset_info = AssetInfo(
                        id=asset_data["id"],
                        name=asset_data["name"],
                        version=asset_data["version"],
                        file_path=asset_data["file"],
                        description=asset_data["description"],
                        tags=asset_data["tags"],
                        parameters=asset_data["parameters"],
                        category=cat_name,
                        created=asset_data["created"]
                    )
                    results.append(asset_info)
        
        return results
    
    def load_indicator(self, indicator_id: str) -> Optional[str]:
        """
        Load indicator code by ID
        
        Args:
            indicator_id: ID of the indicator to load
            
        Returns:
            Indicator MQL5 code as string, or None if not found
        """
        # Find indicator in registry
        indicators = self.registry.get("categories", {}).get("indicators", {}).get("assets", [])
        
        for indicator in indicators:
            if indicator["id"] == indicator_id:
                file_path = self.assets_root / indicator["file"]
                if file_path.exists():
                    with open(file_path, 'r') as f:
                        return f.read()
                break
        
        return None
    
    def list_indicators(self) -> List[Dict[str, Any]]:
        """
        List all available indicators
        
        Returns:
            List of indicator metadata dictionaries
        """
        indicators = self.registry.get("categories", {}).get("indicators", {}).get("assets", [])
        return indicators
    
    def list_strategies(self) -> List[Dict[str, Any]]:
        """
        List all available strategies
        
        Returns:
            List of strategy metadata dictionaries
        """
        strategies = self.registry.get("categories", {}).get("strategies", {}).get("assets", [])
        return strategies
    
    def get_timeframe_config(self, timeframe_id: str) -> Optional[Dict[str, Any]]:
        """
        Get configuration for a specific timeframe
        
        Args:
            timeframe_id: Timeframe identifier (M1, M5, H1, etc.)
            
        Returns:
            Timeframe configuration dictionary or None if not found
        """
        return self.timeframes.get("timeframes", {}).get(timeframe_id)
    
    def get_multi_timeframe_set(self, set_name: str) -> Optional[Dict[str, Any]]:
        """
        Get multi-timeframe analysis configuration
        
        Args:
            set_name: Name of the timeframe set (intraday, swing_trading, etc.)
            
        Returns:
            Multi-timeframe configuration or None if not found
        """
        return self.timeframes.get("multi_timeframe_sets", {}).get(set_name)
    
    def get_strategy_recommendations(self, strategy_type: str) -> Optional[Dict[str, Any]]:
        """
        Get timeframe recommendations for a strategy type
        
        Args:
            strategy_type: Type of strategy (scalping, day_trading, etc.)
            
        Returns:
            Strategy recommendations or None if not found
        """
        return self.timeframes.get("strategy_recommendations", {}).get(strategy_type)
    
    def validate_asset(self, asset_path: str) -> Dict[str, Any]:
        """
        Validate an asset file for syntax and structure
        
        Args:
            asset_path: Path to the asset file
            
        Returns:
            Validation results dictionary
        """
        full_path = self.assets_root / asset_path
        results = {
            "path": str(full_path),
            "exists": full_path.exists(),
            "valid": False,
            "errors": [],
            "warnings": []
        }
        
        if not results["exists"]:
            results["errors"].append("File does not exist")
            return results
        
        try:
            with open(full_path, 'r') as f:
                content = f.read()
            
            # Basic MQL5 syntax checks
            if "#property" not in content:
                results["warnings"].append("No #property declarations found")
            
            if "OnCalculate" not in content and "OnInit" not in content:
                results["warnings"].append("No standard MQL5 function handlers found")
            
            # Check for required sections in different asset types
            if "indicator" in asset_path:
                required_sections = ["#property indicator_", "OnCalculate"]
            elif "strategy" in asset_path:
                required_sections = ["#property", "OnInit", "OnTick"]
            else:
                required_sections = ["#property"]
            
            for section in required_sections:
                if section not in content:
                    results["errors"].append(f"Missing required section: {section}")
            
            results["valid"] = len(results["errors"]) == 0
            
        except Exception as e:
            results["errors"].append(f"Error reading file: {str(e)}")
        
        return results
    
    def get_asset_info(self, asset_id: str, category: str) -> Optional[AssetInfo]:
        """
        Get detailed information about a specific asset
        
        Args:
            asset_id: ID of the asset
            category: Category of the asset
            
        Returns:
            AssetInfo object or None if not found
        """
        assets = self.registry.get("categories", {}).get(category, {}).get("assets", [])
        
        for asset_data in assets:
            if asset_data["id"] == asset_id:
                return AssetInfo(
                    id=asset_data["id"],
                    name=asset_data["name"],
                    version=asset_data["version"],
                    file_path=asset_data["file"],
                    description=asset_data["description"],
                    tags=asset_data["tags"],
                    parameters=asset_data["parameters"],
                    category=category,
                    created=asset_data["created"]
                )
        
        return None

# MCP Tool Functions
def search_assets(query: str, category: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    MCP Tool: Search for assets in the shared library
    
    Args:
        query: Search term
        category: Optional category filter
        
    Returns:
        List of matching assets with metadata
    """
    try:
        manager = AssetLibraryManager()
        results = manager.search_assets(query, category)
        
        return [
            {
                "id": asset.id,
                "name": asset.name,
                "version": asset.version,
                "category": asset.category,
                "description": asset.description,
                "tags": asset.tags,
                "parameters": asset.parameters,
                "file_path": asset.file_path,
                "created": asset.created
            }
            for asset in results
        ]
    except Exception as e:
        return [{"error": f"Search failed: {str(e)}"}]

def load_indicator(indicator_id: str) -> Dict[str, Any]:
    """
    MCP Tool: Load indicator code by ID
    
    Args:
        indicator_id: ID of the indicator to load
        
    Returns:
        Indicator code and metadata
    """
    try:
        manager = AssetLibraryManager()
        code = manager.load_indicator(indicator_id)
        
        if code:
            info = manager.get_asset_info(indicator_id, "indicators")
            return {
                "id": indicator_id,
                "name": info.name if info else "Unknown",
                "code": code,
                "success": True
            }
        else:
            return {
                "id": indicator_id,
                "error": "Indicator not found",
                "success": False
            }
    except Exception as e:
        return {
            "id": indicator_id,
            "error": f"Failed to load indicator: {str(e)}",
            "success": False
        }

def list_indicators() -> List[Dict[str, Any]]:
    """
    MCP Tool: List all available indicators
    
    Returns:
        List of indicator metadata
    """
    try:
        manager = AssetLibraryManager()
        return manager.list_indicators()
    except Exception as e:
        return [{"error": f"Failed to list indicators: {str(e)}"}]

def get_timeframe_config(timeframe_id: str) -> Dict[str, Any]:
    """
    MCP Tool: Get timeframe configuration
    
    Args:
        timeframe_id: Timeframe identifier (M1, M5, H1, etc.)
        
    Returns:
        Timeframe configuration
    """
    try:
        manager = AssetLibraryManager()
        config = manager.get_timeframe_config(timeframe_id)
        
        if config:
            return {"timeframe": timeframe_id, "config": config}
        else:
            return {"timeframe": timeframe_id, "error": "Timeframe not found"}
    except Exception as e:
        return {"timeframe": timeframe_id, "error": f"Failed to get config: {str(e)}"}

def validate_asset(asset_path: str) -> Dict[str, Any]:
    """
    MCP Tool: Validate an asset file
    
    Args:
        asset_path: Path to the asset file relative to assets root
        
    Returns:
        Validation results
    """
    try:
        manager = AssetLibraryManager()
        return manager.validate_asset(asset_path)
    except Exception as e:
        return {"path": asset_path, "error": f"Validation failed: {str(e)}", "valid": False}

# Export functions for MCP server
__all__ = [
    "search_assets",
    "load_indicator",
    "list_indicators",
    "get_timeframe_config",
    "validate_asset",
    "AssetLibraryManager",
    "AssetInfo"
]