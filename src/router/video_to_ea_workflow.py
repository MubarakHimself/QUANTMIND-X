"""
Video-to-EA Workflow Automation

This module provides automated workflow from video analysis to EA creation:
1. Video Ingest - Downloads and processes YouTube videos
2. Video Analysis - Analyzes video for strategy elements
3. TRD Generation - Creates Trading Requirements Document
4. EA Creation - Triggers EA generation workflow
5. Notifications - Uses department mail for notifications

Integrates with:
- src.video_ingest - Video download and processing
- src.agents.tools.strategies_yt.video_analysis_tools - Video analysis
- src.agents.tools.strategies_yt.trd_tools - TRD generation
- src.agents.departments.department_mail - Department notifications
- src.router.workflow_orchestrator - EA creation workflow
"""

import json
import logging
import uuid
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum

logger = logging.getLogger(__name__)

# Import dependencies
try:
    from src.video_ingest.processor import VideoIngestProcessor
    from src.video_ingest.models import JobOptions
    VIDEO_INGEST_AVAILABLE = True
except ImportError:
    VIDEO_INGEST_AVAILABLE = False
    logger.warning("video_ingest module not available")

try:
    from src.agents.tools.strategies_yt.video_analysis_tools import (
        analyze_trading_video,
        extract_indicators,
        extract_entry_rules,
        extract_exit_rules,
        extract_risk_parameters,
        VideoAnalysisResult,
        VideoClip,
    )
    VIDEO_ANALYSIS_AVAILABLE = True
except ImportError:
    VIDEO_ANALYSIS_AVAILABLE = False
    logger.warning("video_analysis_tools not available")

try:
    from src.agents.tools.strategies_yt.trd_tools import TruthObject, TRDGenerator
    TRD_TOOLS_AVAILABLE = True
except ImportError:
    TRD_TOOLS_AVAILABLE = False
    logger.warning("trd_tools not available")

try:
    from src.agents.departments.department_mail import (
        DepartmentMailService,
        MessageType,
        Priority,
    )
    MAIL_AVAILABLE = True
except ImportError:
    MAIL_AVAILABLE = False
    logger.warning("department_mail not available")

try:
    from src.router.workflow_orchestrator import get_orchestrator
    ORCHESTRATOR_AVAILABLE = True
except ImportError:
    ORCHESTRATOR_AVAILABLE = False
    logger.warning("workflow_orchestrator not available")


# =============================================================================
# PATH CONSTANTS
# =============================================================================

PROJECT_ROOT = Path("/home/mubarkahimself/Desktop/QUANTMINDX")
STRATEGIES_YT_DIR = PROJECT_ROOT / "strategies-yt"
WORKFLOW_OUTPUT_DIR = PROJECT_ROOT / "workflows" / "video_to_ea"


# =============================================================================
# WORKFLOW STAGES
# =============================================================================

class VideoToEAStage(str, Enum):
    """Stages in the video-to-EA workflow."""
    VIDEO_INGEST = "video_ingest"
    VIDEO_ANALYSIS = "video_analysis"
    TRD_GENERATION = "trd_generation"
    EA_CREATION = "ea_creation"
    COMPLETED = "completed"
    FAILED = "failed"


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class VideoToEAWorkflowState:
    """State of a video-to-EA workflow."""
    workflow_id: str
    video_url: str
    video_id: str
    strategy_name: str
    current_stage: VideoToEAStage
    status: str  # pending, running, completed, failed
    progress_percent: float = 0.0

    # Data from each stage
    video_metadata: Dict[str, Any] = field(default_factory=dict)
    analysis_result: Dict[str, Any] = field(default_factory=dict)
    trd_content: Optional[str] = None
    ea_workflow_id: Optional[str] = None

    # Timestamps
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

    # Error tracking
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['current_stage'] = self.current_stage.value
        return data


@dataclass
class VideoToEAResult:
    """Result of video-to-EA workflow."""
    success: bool
    workflow_id: str
    video_url: str
    strategy_name: str
    current_stage: VideoToEAStage
    progress_percent: float
    ea_workflow_id: Optional[str] = None
    output_dir: Optional[str] = None
    error: Optional[str] = None
    notifications_sent: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['current_stage'] = self.current_stage.value
        return data


# =============================================================================
# NOTIFICATION SERVICE
# =============================================================================

class WorkflowNotificationService:
    """Service for sending workflow notifications via department mail."""

    def __init__(self, db_path: str = ".quantmind/department_mail.db"):
        self.db_path = db_path
        self.mail_service = None
        if MAIL_AVAILABLE:
            try:
                self.mail_service = DepartmentMailService(db_path=db_path)
            except Exception as e:
                logger.warning(f"Could not initialize mail service: {e}")

    def notify_video_received(
        self,
        video_url: str,
        video_id: str,
        workflow_id: str,
    ) -> bool:
        """Notify that a video has been received for processing."""
        if not self.mail_service:
            logger.info(f"Video received: {video_id}, workflow: {workflow_id}")
            return False

        try:
            # Notify research department
            self.mail_service.send(
                from_dept="video_ingest",
                to_dept="research",
                type=MessageType.STRATEGY_DISPATCH,
                subject=f"Video received for analysis: {video_id}",
                body=json.dumps({
                    "workflow_id": workflow_id,
                    "video_url": video_url,
                    "video_id": video_id,
                    "action": "analyze_video",
                }),
                priority=Priority.HIGH,
            )
            logger.info(f"Sent video received notification for {video_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")
            return False

    def notify_analysis_complete(
        self,
        video_id: str,
        strategy_name: str,
        workflow_id: str,
        analysis_summary: Dict[str, Any],
    ) -> bool:
        """Notify that video analysis is complete."""
        if not self.mail_service:
            logger.info(f"Analysis complete for: {video_id}")
            return False

        try:
            # Notify development department
            self.mail_service.send(
                from_dept="research",
                to_dept="development",
                type=MessageType.TRD_GENERATED,
                subject=f"TRD ready for: {strategy_name}",
                body=json.dumps({
                    "workflow_id": workflow_id,
                    "video_id": video_id,
                    "strategy_name": strategy_name,
                    "analysis_summary": analysis_summary,
                }),
                priority=Priority.HIGH,
            )
            logger.info(f"Sent analysis complete notification for {strategy_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")
            return False

    def notify_ea_created(
        self,
        strategy_name: str,
        ea_workflow_id: str,
        parent_workflow_id: str,
    ) -> bool:
        """Notify that EA has been created."""
        if not self.mail_service:
            logger.info(f"EA created: {strategy_name}, workflow: {ea_workflow_id}")
            return False

        try:
            # Notify trading floor
            self.mail_service.send(
                from_dept="development",
                to_dept="trading",
                type=MessageType.CODE_READY,
                subject=f"EA ready for testing: {strategy_name}",
                body=json.dumps({
                    "workflow_id": parent_workflow_id,
                    "ea_workflow_id": ea_workflow_id,
                    "strategy_name": strategy_name,
                }),
                priority=Priority.NORMAL,
            )
            logger.info(f"Sent EA created notification for {strategy_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")
            return False

    def notify_error(
        self,
        workflow_id: str,
        stage: str,
        error: str,
    ) -> bool:
        """Notify of workflow error."""
        if not self.mail_service:
            logger.error(f"Workflow error: {workflow_id}, stage: {stage}, error: {error}")
            return False

        try:
            self.mail_service.send(
                from_dept="system",
                to_dept="floor_manager",
                type=MessageType.ERROR,
                subject=f"Video-to-EA workflow failed: {stage}",
                body=json.dumps({
                    "workflow_id": workflow_id,
                    "stage": stage,
                    "error": error,
                }),
                priority=Priority.URGENT,
            )
            logger.info(f"Sent error notification for workflow {workflow_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to send error notification: {e}")
            return False

    def close(self):
        """Close the mail service."""
        if self.mail_service:
            self.mail_service.close()


# =============================================================================
# VIDEO-TO-EA WORKFLOW ORCHESTRATOR
# =============================================================================

class VideoToEAWorkflowOrchestrator:
    """
    Orchestrates the complete video-to-EA workflow.

    Flow:
    1. Receive video URL
    2. Download and process video (Video Ingest)
    3. Analyze video for strategy elements (Video Analysis)
    4. Generate TRD (Trading Requirements Document)
    5. Trigger EA creation workflow
    6. Send notifications via department mail
    """

    def __init__(
        self,
        output_dir: Path = WORKFLOW_OUTPUT_DIR,
        mail_db_path: str = ".quantmind/department_mail.db",
        on_progress: Optional[Callable[[str, float, VideoToEAStage], None]] = None,
    ):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.on_progress = on_progress

        # Initialize services
        self.notification_service = WorkflowNotificationService(db_path=mail_db_path)
        self._workflows: Dict[str, VideoToEAWorkflowState] = {}

        logger.info("VideoToEAWorkflowOrchestrator initialized")

    async def start_workflow(
        self,
        video_url: str,
        strategy_name: Optional[str] = None,
        analysis_depth: str = "standard",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> VideoToEAResult:
        """
        Start a new video-to-EA workflow.

        Args:
            video_url: YouTube URL to process
            strategy_name: Name for the strategy (derived from video if not provided)
            analysis_depth: "quick", "standard", or "deep"
            metadata: Optional metadata

        Returns:
            VideoToEAResult with workflow status
        """
        # Generate workflow ID
        workflow_id = f"v2e_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"

        # Extract video ID from URL
        video_id = self._extract_video_id(video_url)
        if not video_id:
            return VideoToEAResult(
                success=False,
                workflow_id=workflow_id,
                video_url=video_url,
                strategy_name=strategy_name or "unknown",
                current_stage=VideoToEAStage.FAILED,
                progress_percent=0.0,
                error="Invalid YouTube URL",
            )

        # Set strategy name
        if not strategy_name:
            strategy_name = f"strategy_{video_id}"

        # Create workflow state
        workflow_state = VideoToEAWorkflowState(
            workflow_id=workflow_id,
            video_url=video_url,
            video_id=video_id,
            strategy_name=strategy_name,
            current_stage=VideoToEAStage.VIDEO_INGEST,
            status="running",
            started_at=datetime.now(timezone.utc).isoformat(),
        )

        self._workflows[workflow_id] = workflow_state
        logger.info(f"Started video-to-EA workflow {workflow_id} for {video_url}")

        # Run the workflow asynchronously
        asyncio.create_task(self._run_workflow(workflow_id, analysis_depth))

        # Return initial result
        return VideoToEAResult(
            success=True,
            workflow_id=workflow_id,
            video_url=video_url,
            strategy_name=strategy_name,
            current_stage=VideoToEAStage.VIDEO_INGEST,
            progress_percent=10.0,
        )

    async def _run_workflow(self, workflow_id: str, analysis_depth: str):
        """Run the workflow stages."""
        workflow_state = self._workflows.get(workflow_id)
        if not workflow_state:
            logger.error(f"Workflow {workflow_id} not found")
            return

        notifications_sent = []

        try:
            # Stage 1: Video Ingest
            workflow_state.current_stage = VideoToEAStage.VIDEO_INGEST
            self._notify_progress(workflow_id, 20.0)

            video_output = await self._process_video(workflow_state.video_url)
            if not video_output.get("success"):
                raise Exception(video_output.get("error", "Video processing failed"))

            workflow_state.video_metadata = video_output
            logger.info(f"Video ingested: {workflow_state.video_id}")

            # Send notification: video received
            self.notification_service.notify_video_received(
                video_url=workflow_state.video_url,
                video_id=workflow_state.video_id,
                workflow_id=workflow_id,
            )
            notifications_sent.append("video_received")

            # Stage 2: Video Analysis
            workflow_state.current_stage = VideoToEAStage.VIDEO_ANALYSIS
            self._notify_progress(workflow_id, 40.0)

            analysis_result = await self._analyze_video(
                video_id=workflow_state.video_id,
                video_url=workflow_state.video_url,
                analysis_depth=analysis_depth,
            )

            if not analysis_result.get("success"):
                raise Exception(analysis_result.get("error", "Video analysis failed"))

            workflow_state.analysis_result = analysis_result
            logger.info(f"Video analyzed: {workflow_state.video_id}")

            # Save analysis to file
            analysis_dir = self.output_dir / workflow_id / "analysis"
            analysis_dir.mkdir(parents=True, exist_ok=True)
            (analysis_dir / "analysis.json").write_text(
                json.dumps(analysis_result, indent=2)
            )

            # Stage 3: TRD Generation
            workflow_state.current_stage = VideoToEAStage.TRD_GENERATION
            self._notify_progress(workflow_id, 60.0)

            trd_result = await self._generate_trd(
                video_id=workflow_state.video_id,
                strategy_name=workflow_state.strategy_name,
                analysis_result=analysis_result,
            )

            if not trd_result.get("success"):
                raise Exception(trd_result.get("error", "TRD generation failed"))

            workflow_state.trd_content = trd_result.get("trd_content", "")
            logger.info(f"TRD generated: {workflow_state.strategy_name}")

            # Save TRD to file
            trd_dir = self.output_dir / workflow_id / "trd"
            trd_dir.mkdir(parents=True, exist_ok=True)
            (trd_dir / f"{workflow_state.strategy_name}.md").write_text(
                workflow_state.trd_content
            )

            # Send notification: analysis complete
            self.notification_service.notify_analysis_complete(
                video_id=workflow_state.video_id,
                strategy_name=workflow_state.strategy_name,
                workflow_id=workflow_id,
                analysis_summary={
                    "indicators_found": len(analysis_result.get("analysis", {}).get("timeline", [])),
                    "confidence": analysis_result.get("summary", {}).get("confidence", 0),
                },
            )
            notifications_sent.append("analysis_complete")

            # Stage 4: EA Creation
            workflow_state.current_stage = VideoToEAStage.EA_CREATION
            self._notify_progress(workflow_id, 80.0)

            ea_result = await self._create_ea(workflow_state)
            workflow_state.ea_workflow_id = ea_result.get("workflow_id")

            if not ea_result.get("success"):
                raise Exception(ea_result.get("error", "EA creation failed"))

            logger.info(f"EA workflow started: {workflow_state.ea_workflow_id}")

            # Send notification: EA created
            self.notification_service.notify_ea_created(
                strategy_name=workflow_state.strategy_name,
                ea_workflow_id=workflow_state.ea_workflow_id,
                parent_workflow_id=workflow_id,
            )
            notifications_sent.append("ea_created")

            # Complete
            workflow_state.current_stage = VideoToEAStage.COMPLETED
            workflow_state.status = "completed"
            workflow_state.completed_at = datetime.now(timezone.utc).isoformat()
            self._notify_progress(workflow_id, 100.0)

            logger.info(f"Video-to-EA workflow {workflow_id} completed")

        except Exception as e:
            logger.error(f"Workflow {workflow_id} failed: {e}", exc_info=True)
            workflow_state.current_stage = VideoToEAStage.FAILED
            workflow_state.status = "failed"
            workflow_state.error = str(e)
            workflow_state.completed_at = datetime.now(timezone.utc).isoformat()

            # Send error notification
            self.notification_service.notify_error(
                workflow_id=workflow_id,
                stage=workflow_state.current_stage.value,
                error=str(e),
            )

    async def _process_video(self, video_url: str) -> Dict[str, Any]:
        """Process video using video_ingest module."""
        if not VIDEO_INGEST_AVAILABLE:
            return {
                "success": False,
                "error": "Video ingest module not available"
            }

        try:
            processor = VideoIngestProcessor()
            options = JobOptions()

            result = processor.process(url=video_url, job_id=f"v2e_{uuid.uuid4().hex[:8]}", options=options)

            return {
                "success": True,
                "video_id": result.video_id,
                "title": result.timeline.title if result.timeline else "Unknown",
                "duration": result.timeline.duration_seconds if result.timeline else 0,
                "output_path": str(result.output_dir),
                "provider_used": result.provider_used,
            }
        except Exception as e:
            logger.error(f"Video processing failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _analyze_video(
        self,
        video_id: str,
        video_url: str,
        analysis_depth: str,
    ) -> Dict[str, Any]:
        """Analyze video using video_analysis_tools."""
        if not VIDEO_ANALYSIS_AVAILABLE:
            return {
                "success": False,
                "error": "Video analysis tools not available"
            }

        try:
            # Analyze the video
            analysis_result = await analyze_trading_video(
                video_id=video_id,
                video_url=video_url,
                analysis_depth=analysis_depth,
            )

            if not analysis_result.get("success"):
                return analysis_result

            # Extract additional elements
            timeline = analysis_result.get("analysis", {}).get("timeline", [])

            indicators = await extract_indicators(video_id=video_id, timeline=timeline)
            entry_rules = await extract_entry_rules(video_id=video_id, timeline=timeline)
            exit_rules = await extract_exit_rules(video_id=video_id, timeline=timeline)
            risk_params = await extract_risk_parameters(video_id=video_id, timeline=timeline)

            # Combine results
            analysis_result["extracted_elements"] = {
                "indicators": indicators.get("indicators", []),
                "entry_rules": entry_rules.get("entry_rules", []),
                "exit_rules": exit_rules.get("exit_rules", []),
                "risk_parameters": risk_params.get("risk_parameters", {}),
            }

            return analysis_result

        except Exception as e:
            logger.error(f"Video analysis failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _generate_trd(
        self,
        video_id: str,
        strategy_name: str,
        analysis_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate TRD from analysis result."""
        if not TRD_TOOLS_AVAILABLE:
            return {
                "success": False,
                "error": "TRD tools not available"
            }

        try:
            # Extract elements from analysis
            elements = analysis_result.get("extracted_elements", {})

            # Create TruthObject from analysis
            truth_object = TruthObject(
                title=strategy_name,
                description=analysis_result.get("summary", {}).get("title", strategy_name),
                entry_conditions=[
                    cond.get("name", "")
                    for cond in elements.get("entry_rules", [])
                ],
                exit_conditions=[
                    cond.get("name", "")
                    for cond in elements.get("exit_rules", [])
                ],
                stop_loss_pips=elements.get("risk_parameters", {}).get("stop_loss_value", 50.0),
                take_profit_pips=elements.get("risk_parameters", {}).get("take_profit_value", 100.0),
                source_video_id=video_id,
            )

            # Generate TRD content
            generator = TRDGenerator()
            trd_content = generator.generate(truth_object)

            return {
                "success": True,
                "trd_content": trd_content,
                "truth_object": truth_object.to_dict(),
            }

        except Exception as e:
            logger.error(f"TRD generation failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _create_ea(self, workflow_state: VideoToEAWorkflowState) -> Dict[str, Any]:
        """Trigger EA creation workflow."""
        if not ORCHESTRATOR_AVAILABLE:
            return {
                "success": False,
                "error": "Workflow orchestrator not available"
            }

        try:
            orchestrator = get_orchestrator()

            # Save TRD to file
            trd_dir = self.output_dir / workflow_state.workflow_id / "trd"
            trd_file = trd_dir / f"{workflow_state.strategy_name}.md"

            # Submit to orchestrator
            ea_workflow_id = await orchestrator.submit_trd_task(
                trd_file=trd_file,
                trd_content=workflow_state.trd_content,
                metadata={
                    "source": "video_to_ea",
                    "video_url": workflow_state.video_url,
                    "video_id": workflow_state.video_id,
                    "parent_workflow_id": workflow_state.workflow_id,
                },
            )

            return {
                "success": True,
                "workflow_id": ea_workflow_id,
            }

        except Exception as e:
            logger.error(f"EA creation failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def _extract_video_id(self, url: str) -> Optional[str]:
        """Extract YouTube video ID from URL."""
        import re

        # Handle different YouTube URL formats
        patterns = [
            r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([a-zA-Z0-9_-]{11})',
            r'^([a-zA-Z0-9_-]{11})$',  # Direct video ID
        ]

        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)

        return None

    def _notify_progress(self, workflow_id: str, progress: float):
        """Notify progress callback."""
        workflow_state = self._workflows.get(workflow_id)
        if workflow_state:
            workflow_state.progress_percent = progress
            if self.on_progress:
                self.on_progress(workflow_id, progress, workflow_state.current_stage)

    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get workflow status."""
        workflow_state = self._workflows.get(workflow_id)
        if workflow_state:
            return workflow_state.to_dict()
        return None

    def get_all_workflows(self) -> List[Dict[str, Any]]:
        """Get all workflows."""
        return [w.to_dict() for w in self._workflows.values()]

    def close(self):
        """Clean up resources."""
        self.notification_service.close()
        logger.info("VideoToEAWorkflowOrchestrator closed")


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_video_to_ea_orchestrator: Optional[VideoToEAWorkflowOrchestrator] = None


def get_video_to_ea_orchestrator() -> VideoToEAWorkflowOrchestrator:
    """Get or create the global video-to-EA orchestrator."""
    global _video_to_ea_orchestrator
    if _video_to_ea_orchestrator is None:
        _video_to_ea_orchestrator = VideoToEAWorkflowOrchestrator()
    return _video_to_ea_orchestrator


def create_video_to_ea_orchestrator(
    output_dir: Path = WORKFLOW_OUTPUT_DIR,
    mail_db_path: str = ".quantmind/department_mail.db",
    on_progress: Optional[Callable[[str, float, VideoToEAStage], None]] = None,
) -> VideoToEAWorkflowOrchestrator:
    """Create a video-to-EA orchestrator instance."""
    return VideoToEAWorkflowOrchestrator(
        output_dir=output_dir,
        mail_db_path=mail_db_path,
        on_progress=on_progress,
    )
