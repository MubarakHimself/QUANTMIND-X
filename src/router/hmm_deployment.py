"""
HMM Deployment Workflow
========================

Manages HMM model deployment stages and state transitions.
Implements staged deployment strategy: Training -> Shadow -> Hybrid -> Production.

Features:
- State machine for deployment stages
- Human approval gates for mode transitions
- Performance monitoring and rollback capability
- Integration with Sentinel for mode switching

Deployment Modes:
- ising_only: Use Ising Model only (default)
- hmm_shadow: HMM runs in parallel, predictions logged but not used
- hmm_hybrid_20: 20% HMM weight, 80% Ising
- hmm_hybrid_50: 50% HMM weight, 50% Ising
- hmm_hybrid_80: 80% HMM weight, 20% Ising
- hmm_only: HMM-only predictions

Reference: docs/architecture/components.md
"""

import os
import sys
import json
import logging
import secrets
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.database.models import HMMModel, HMMDeployment, HMMShadowLog
from src.database.engine import engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import desc

logger = logging.getLogger(__name__)


class DeploymentMode(Enum):
    """Deployment mode enumeration."""
    ISING_ONLY = "ising_only"
    HMM_SHADOW = "hmm_shadow"
    HMM_HYBRID_20 = "hmm_hybrid_20"
    HMM_HYBRID_50 = "hmm_hybrid_50"
    HMM_HYBRID_80 = "hmm_hybrid_80"
    HMM_ONLY = "hmm_only"
    
    @classmethod
    def get_hmm_weight(cls, mode: 'DeploymentMode') -> float:
        """Get HMM weight for mode."""
        weights = {
            cls.ISING_ONLY: 0.0,
            cls.HMM_SHADOW: 0.0,  # Shadow doesn't affect decisions
            cls.HMM_HYBRID_20: 0.2,
            cls.HMM_HYBRID_50: 0.5,
            cls.HMM_HYBRID_80: 0.8,
            cls.HMM_ONLY: 1.0
        }
        return weights.get(mode, 0.0)
    
    @classmethod
    def requires_approval(cls, mode: 'DeploymentMode') -> bool:
        """Check if mode transition requires approval."""
        approval_modes = {
            cls.HMM_HYBRID_50,
            cls.HMM_HYBRID_80,
            cls.HMM_ONLY
        }
        return mode in approval_modes


@dataclass
class DeploymentState:
    """Current deployment state."""
    mode: DeploymentMode
    model_version: Optional[str]
    model_id: Optional[int]
    transition_date: datetime
    approved_by: Optional[str]
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    rollback_count: int = 0
    
    def to_dict(self) -> Dict:
        return {
            'mode': self.mode.value,
            'model_version': self.model_version,
            'model_id': self.model_id,
            'transition_date': self.transition_date.isoformat(),
            'approved_by': self.approved_by,
            'performance_metrics': self.performance_metrics,
            'rollback_count': self.rollback_count
        }


class HMMDeploymentManager:
    """
    HMM Deployment Manager.
    
    Manages deployment stages, approval gates, and rollback capability.
    
    Usage:
        ```python
        manager = HMMDeploymentManager()
        
        # Get current state
        state = manager.get_current_state()
        
        # Transition to shadow mode
        manager.transition_to(DeploymentMode.HMM_SHADOW)
        
        # Transition to hybrid (requires approval)
        token = manager.request_approval(DeploymentMode.HMM_HYBRID_20)
        manager.approve_transition(token, "user@example.com")
        ```
    """
    
    # Valid mode transitions
    VALID_TRANSITIONS = {
        DeploymentMode.ISING_ONLY: [DeploymentMode.HMM_SHADOW],
        DeploymentMode.HMM_SHADOW: [DeploymentMode.HMM_HYBRID_20, DeploymentMode.ISING_ONLY],
        DeploymentMode.HMM_HYBRID_20: [DeploymentMode.HMM_HYBRID_50, DeploymentMode.HMM_SHADOW, DeploymentMode.ISING_ONLY],
        DeploymentMode.HMM_HYBRID_50: [DeploymentMode.HMM_HYBRID_80, DeploymentMode.HMM_HYBRID_20, DeploymentMode.ISING_ONLY],
        DeploymentMode.HMM_HYBRID_80: [DeploymentMode.HMM_ONLY, DeploymentMode.HMM_HYBRID_50, DeploymentMode.ISING_ONLY],
        DeploymentMode.HMM_ONLY: [DeploymentMode.HMM_HYBRID_80, DeploymentMode.ISING_ONLY]
    }
    
    def __init__(self, config_path: str = "config/hmm_config.json"):
        """Initialize deployment manager with configuration."""
        self.config = self._load_config(config_path)
        self.deployment_config = self.config.get('deployment', {})
        
        # Database session
        self.Session = sessionmaker(bind=engine)
        
        # Pending approvals
        self._pending_approvals: Dict[str, Dict] = {}
        
        # State change callbacks
        self._state_callbacks: List[Callable] = []
        
        # Initialize state from database
        self._current_state = self._load_state_from_db()
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load config file: {e}")
            return {}
    
    def _load_state_from_db(self) -> DeploymentState:
        """Load current deployment state from database."""
        session = self.Session()
        
        try:
            deployment = session.query(HMMDeployment).filter(
                HMMDeployment.is_active == True
            ).order_by(desc(HMMDeployment.transition_date)).first()
            
            if deployment:
                return DeploymentState(
                    mode=DeploymentMode(deployment.mode),
                    model_version=deployment.model.version if deployment.model else None,
                    model_id=deployment.model_id,
                    transition_date=deployment.transition_date,
                    approved_by=deployment.approved_by,
                    performance_metrics=deployment.performance_metrics or {},
                    rollback_count=deployment.rollback_count
                )
            
        except Exception as e:
            logger.error(f"Failed to load state from database: {e}")
        finally:
            session.close()
        
        # Return default state
        return DeploymentState(
            mode=DeploymentMode.ISING_ONLY,
            model_version=None,
            model_id=None,
            transition_date=datetime.now(timezone.utc),
            approved_by=None
        )
    
    def _save_state_to_db(self, new_state: DeploymentState, 
                          previous_mode: DeploymentMode,
                          approval_token: Optional[str] = None) -> None:
        """Save new deployment state to database."""
        session = self.Session()
        
        try:
            # Deactivate previous deployments
            session.query(HMMDeployment).update({'is_active': False})
            
            # Create new deployment record
            deployment = HMMDeployment(
                model_id=new_state.model_id,
                mode=new_state.mode.value,
                previous_mode=previous_mode.value,
                transition_date=new_state.transition_date,
                approved_by=new_state.approved_by,
                approval_token=approval_token,
                performance_metrics=new_state.performance_metrics,
                rollback_count=new_state.rollback_count,
                is_active=True
            )
            
            session.add(deployment)
            session.commit()
            
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to save state to database: {e}")
            raise
        finally:
            session.close()
    
    def _notify_state_change(self, old_state: DeploymentState, 
                             new_state: DeploymentState) -> None:
        """Notify callbacks of state change."""
        for callback in self._state_callbacks:
            try:
                callback(old_state.to_dict(), new_state.to_dict())
            except Exception as e:
                logger.warning(f"State callback error: {e}")
    
    def add_state_callback(self, callback: Callable) -> None:
        """Add a callback for state changes."""
        self._state_callbacks.append(callback)
    
    def remove_state_callback(self, callback: Callable) -> None:
        """Remove a state change callback."""
        if callback in self._state_callbacks:
            self._state_callbacks.remove(callback)
    
    def get_current_state(self) -> DeploymentState:
        """Get current deployment state."""
        return self._current_state
    
    def get_valid_transitions(self) -> List[DeploymentMode]:
        """Get valid mode transitions from current state."""
        return self.VALID_TRANSITIONS.get(self._current_state.mode, [])
    
    def can_transition_to(self, target_mode: DeploymentMode) -> Tuple[bool, str]:
        """
        Check if transition to target mode is valid.
        
        Returns:
            Tuple of (can_transition, reason)
        """
        current_mode = self._current_state.mode
        
        # Check if already in target mode
        if current_mode == target_mode:
            return False, f"Already in {target_mode.value} mode"
        
        # Check valid transitions
        valid_targets = self.VALID_TRANSITIONS.get(current_mode, [])
        
        if target_mode not in valid_targets:
            return False, f"Cannot transition from {current_mode.value} to {target_mode.value}"
        
        # Check if approval required
        if DeploymentMode.requires_approval(target_mode):
            return True, "Approval required"
        
        return True, "Transition allowed"
    
    def transition_to(self, target_mode: DeploymentMode,
                      model_id: Optional[int] = None,
                      approved_by: Optional[str] = None,
                      approval_token: Optional[str] = None,
                      performance_metrics: Optional[Dict] = None) -> bool:
        """
        Transition to target deployment mode.
        
        Args:
            target_mode: Target deployment mode
            model_id: Optional model ID to use
            approved_by: User who approved (if applicable)
            approval_token: Approval token (if applicable)
            performance_metrics: Optional performance metrics
            
        Returns:
            True if transition succeeded
        """
        can_transition, reason = self.can_transition_to(target_mode)
        
        if not can_transition:
            logger.warning(f"Transition rejected: {reason}")
            return False
        
        # Check approval if required
        if DeploymentMode.requires_approval(target_mode):
            if not approval_token or approval_token not in self._pending_approvals:
                logger.warning(f"Transition to {target_mode.value} requires approval")
                return False
            
            # Validate approval
            approval = self._pending_approvals[approval_token]
            if approval['target_mode'] != target_mode:
                return False
            
            # Clear pending approval
            del self._pending_approvals[approval_token]
        
        # Create new state
        old_state = self._current_state
        
        new_state = DeploymentState(
            mode=target_mode,
            model_version=old_state.model_version,
            model_id=model_id or old_state.model_id,
            transition_date=datetime.now(timezone.utc),
            approved_by=approved_by,
            performance_metrics=performance_metrics or {},
            rollback_count=old_state.rollback_count
        )
        
        # Save to database
        self._save_state_to_db(new_state, old_state.mode, approval_token)
        
        # Update current state
        self._current_state = new_state
        
        # Notify callbacks
        self._notify_state_change(old_state, new_state)
        
        logger.info(f"Transitioned from {old_state.mode.value} to {target_mode.value}")
        
        return True
    
    def request_approval(self, target_mode: DeploymentMode,
                        requester: str = "system") -> str:
        """
        Request approval for mode transition.
        
        Args:
            target_mode: Target deployment mode
            requester: Who requested the approval
            
        Returns:
            Approval token
        """
        can_transition, reason = self.can_transition_to(target_mode)
        
        if not can_transition:
            raise ValueError(f"Cannot request approval: {reason}")
        
        if not DeploymentMode.requires_approval(target_mode):
            raise ValueError(f"Mode {target_mode.value} does not require approval")
        
        # Generate approval token
        token = secrets.token_urlsafe(32)
        
        # Store pending approval
        self._pending_approvals[token] = {
            'target_mode': target_mode,
            'requester': requester,
            'requested_at': datetime.now(timezone.utc).isoformat(),
            'current_mode': self._current_state.mode
        }
        
        logger.info(f"Approval requested for {target_mode.value} by {requester}")
        
        return token
    
    def approve_transition(self, token: str, approver: str) -> Tuple[bool, str]:
        """
        Approve a pending mode transition.
        
        Args:
            token: Approval token
            approver: Who approved the transition
            
        Returns:
            Tuple of (success, message)
        """
        if token not in self._pending_approvals:
            return False, "Invalid or expired approval token"
        
        approval = self._pending_approvals[token]
        target_mode = approval['target_mode']
        
        # Execute transition
        success = self.transition_to(
            target_mode=target_mode,
            approved_by=approver,
            approval_token=token
        )
        
        if success:
            return True, f"Transition to {target_mode.value} approved and executed"
        else:
            return False, "Transition failed"
    
    def reject_approval(self, token: str, reason: str = "") -> bool:
        """Reject a pending approval."""
        if token in self._pending_approvals:
            del self._pending_approvals[token]
            logger.info(f"Approval rejected: {reason}")
            return True
        return False
    
    def rollback(self, reason: str = "Manual rollback") -> bool:
        """
        Rollback to previous mode.
        
        Args:
            reason: Reason for rollback
            
        Returns:
            True if rollback succeeded
        """
        current_mode = self._current_state.mode
        
        # Determine rollback target
        if current_mode == DeploymentMode.ISING_ONLY:
            logger.warning("Cannot rollback from ISING_ONLY mode")
            return False
        
        # Get previous mode from database
        session = self.Session()
        
        try:
            previous_deployment = session.query(HMMDeployment).filter(
                HMMDeployment.mode != current_mode.value,
                HMMDeployment.is_active == False
            ).order_by(desc(HMMDeployment.transition_date)).first()
            
            if not previous_deployment:
                # Default to ISING_ONLY
                target_mode = DeploymentMode.ISING_ONLY
            else:
                target_mode = DeploymentMode(previous_deployment.mode)
            
        finally:
            session.close()
        
        # Execute rollback
        old_state = self._current_state
        
        new_state = DeploymentState(
            mode=target_mode,
            model_version=old_state.model_version,
            model_id=old_state.model_id,
            transition_date=datetime.now(timezone.utc),
            approved_by="rollback",
            performance_metrics={'rollback_reason': reason},
            rollback_count=old_state.rollback_count + 1
        )
        
        self._save_state_to_db(new_state, old_state.mode)
        self._current_state = new_state
        self._notify_state_change(old_state, new_state)
        
        logger.warning(f"Rolled back from {current_mode.value} to {target_mode.value}: {reason}")
        
        return True
    
    def get_deployment_history(self, limit: int = 10) -> List[Dict]:
        """Get deployment history."""
        session = self.Session()
        
        try:
            deployments = session.query(HMMDeployment).order_by(
                desc(HMMDeployment.transition_date)
            ).limit(limit).all()
            
            return [
                {
                    'id': d.id,
                    'mode': d.mode,
                    'previous_mode': d.previous_mode,
                    'transition_date': d.transition_date.isoformat(),
                    'approved_by': d.approved_by,
                    'performance_metrics': d.performance_metrics,
                    'rollback_count': d.rollback_count,
                    'is_active': d.is_active
                }
                for d in deployments
            ]
            
        finally:
            session.close()
    
    def get_pending_approvals(self) -> List[Dict]:
        """Get list of pending approvals."""
        return [
            {
                'token': token,
                'target_mode': approval['target_mode'].value,
                'requester': approval['requester'],
                'requested_at': approval['requested_at'],
                'current_mode': approval['current_mode'].value
            }
            for token, approval in self._pending_approvals.items()
        ]
    
    def update_performance_metrics(self, metrics: Dict[str, float]) -> None:
        """Update current deployment performance metrics."""
        self._current_state.performance_metrics.update(metrics)
        
        # Update in database
        session = self.Session()
        
        try:
            deployment = session.query(HMMDeployment).filter(
                HMMDeployment.is_active == True
            ).first()
            
            if deployment:
                deployment.performance_metrics = self._current_state.performance_metrics
                session.commit()
                
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to update metrics: {e}")
        finally:
            session.close()
    
    def get_hmm_weight_for_mode(self, mode: Optional[DeploymentMode] = None) -> float:
        """
        Get HMM weight for a given mode or current mode.
        
        Args:
            mode: Mode to get weight for (None = current mode)
            
        Returns:
            HMM weight (0.0 to 1.0)
        """
        target_mode = mode or self._current_state.mode
        return DeploymentMode.get_hmm_weight(target_mode)


# Global instance
_deployment_manager: Optional[HMMDeploymentManager] = None


def get_deployment_manager() -> HMMDeploymentManager:
    """Get or create global deployment manager instance."""
    global _deployment_manager
    if _deployment_manager is None:
        _deployment_manager = HMMDeploymentManager()
    return _deployment_manager


# Example usage
if __name__ == "__main__":
    manager = HMMDeploymentManager()
    
    # Get current state
    state = manager.get_current_state()
    print(f"Current mode: {state.mode.value}")
    print(f"Valid transitions: {[m.value for m in manager.get_valid_transitions()]}")
    
    # Transition to shadow mode
    success = manager.transition_to(DeploymentMode.HMM_SHADOW)
    print(f"Transition to shadow: {'success' if success else 'failed'}")
    
    # Request approval for hybrid mode
    token = manager.request_approval(DeploymentMode.HMM_HYBRID_20, "test@example.com")
    print(f"Approval token: {token[:16]}...")
    
    # Approve transition
    success, message = manager.approve_transition(token, "admin@example.com")
    print(f"Approval result: {message}")