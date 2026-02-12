"""
Job Queue Manager for NPRD system.

This module provides job queue management with SQLite persistence, concurrent processing,
and automatic recovery after system restarts.
"""

import sqlite3
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
from concurrent.futures import ThreadPoolExecutor, Future
import logging

from .models import JobStatus, JobState, JobOptions, TimelineOutput, NPRDConfig
from .exceptions import JobError


logger = logging.getLogger(__name__)


class JobQueueManager:
    """
    Manages video processing jobs with SQLite persistence and concurrent execution.
    
    Features:
    - SQLite database for durable job state storage
    - Concurrent job processing with configurable limits
    - Automatic job recovery after restart
    - Job isolation (failures don't affect other jobs)
    - Automatic job start when resources available
    
    Job States:
    - PENDING: Job submitted, waiting to start
    - DOWNLOADING: Video download in progress
    - PROCESSING: Frame/audio extraction in progress
    - ANALYZING: AI model analysis in progress
    - COMPLETED: Job finished successfully
    - FAILED: Job failed with error
    """
    
    def __init__(self, config: NPRDConfig):
        """
        Initialize Job Queue Manager.
        
        Args:
            config: NPRD configuration with database path and concurrency settings
        """
        self.config = config
        self.db_path = config.job_db_path
        self.max_concurrent = config.max_concurrent_jobs
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Executor for concurrent job processing
        self._executor: Optional[ThreadPoolExecutor] = None
        self._active_jobs: Dict[str, Future] = {}
        self._shutdown = False
        
        # Job processor callback (set externally)
        self._job_processor: Optional[Callable[[str, str, JobOptions], TimelineOutput]] = None
        
        # Initialize database
        self._init_database()
        
        logger.info(f"JobQueueManager initialized with max_concurrent={self.max_concurrent}")
    
    def _init_database(self) -> None:
        """Initialize SQLite database with job tables."""
        # Ensure directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Create jobs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                progress INTEGER NOT NULL DEFAULT 0,
                video_url TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                result_path TEXT,
                error TEXT,
                options TEXT
            )
        """)
        
        # Create job logs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS job_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                message TEXT NOT NULL,
                FOREIGN KEY (job_id) REFERENCES jobs (job_id)
            )
        """)
        
        # Create indexes for performance
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_jobs_status 
            ON jobs (status)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_jobs_created_at 
            ON jobs (created_at)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_job_logs_job_id 
            ON job_logs (job_id)
        """)
        
        conn.commit()
        conn.close()
        
        logger.info(f"Database initialized at {self.db_path}")
    
    def submit_job(self, url: str, options: Optional[JobOptions] = None) -> str:
        """
        Submit a new video processing job.
        
        Args:
            url: Video URL to process
            options: Optional job configuration
            
        Returns:
            job_id: Unique identifier for the submitted job
            
        Raises:
            JobError: If job submission fails
        """
        if options is None:
            options = JobOptions()
        
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        now = datetime.now()
        
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO jobs (
                    job_id, status, progress, video_url, 
                    created_at, updated_at, options
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                job_id,
                JobState.PENDING.value,
                0,
                url,
                now.isoformat(),
                now.isoformat(),
                str(options.to_dict())
            ))
            
            # Log job submission
            cursor.execute("""
                INSERT INTO job_logs (job_id, timestamp, message)
                VALUES (?, ?, ?)
            """, (
                job_id,
                now.isoformat(),
                f"Job submitted for URL: {url}"
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Job {job_id} submitted for URL: {url}")
            
            # Try to start job if resources available
            self._try_start_pending_jobs()
            
            return job_id
            
        except sqlite3.Error as e:
            raise JobError(f"Failed to submit job: {e}")
    
    def get_job_status(self, job_id: str) -> JobStatus:
        """
        Get current status of a job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            JobStatus with current state and progress
            
        Raises:
            JobError: If job not found or database error
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT job_id, status, progress, video_url, 
                       created_at, updated_at, result_path, error
                FROM jobs
                WHERE job_id = ?
            """, (job_id,))
            
            row = cursor.fetchone()
            
            if row is None:
                conn.close()
                raise JobError(f"Job {job_id} not found")
            
            # Fetch logs
            cursor.execute("""
                SELECT message FROM job_logs
                WHERE job_id = ?
                ORDER BY timestamp ASC
            """, (job_id,))
            
            logs = [row[0] for row in cursor.fetchall()]
            
            conn.close()
            
            return JobStatus(
                job_id=row[0],
                status=JobState(row[1]),
                progress=row[2],
                video_url=row[3],
                created_at=datetime.fromisoformat(row[4]),
                updated_at=datetime.fromisoformat(row[5]),
                result_path=row[6],
                error=row[7],
                logs=logs
            )
            
        except sqlite3.Error as e:
            raise JobError(f"Failed to get job status: {e}")
    
    def get_job_result(self, job_id: str) -> Optional[TimelineOutput]:
        """
        Get result of a completed job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            TimelineOutput if job completed successfully, None otherwise
            
        Raises:
            JobError: If job not found or database error
        """
        status = self.get_job_status(job_id)
        
        if status.status != JobState.COMPLETED:
            return None
        
        if status.result_path is None:
            return None
        
        try:
            result_path = Path(status.result_path)
            if not result_path.exists():
                logger.warning(f"Result file not found: {result_path}")
                return None
            
            return TimelineOutput.load_from_file(result_path)
            
        except Exception as e:
            logger.error(f"Failed to load job result: {e}")
            return None
    
    def update_job_status(
        self, 
        job_id: str, 
        status: JobState, 
        progress: Optional[int] = None,
        result_path: Optional[str] = None,
        error: Optional[str] = None,
        log_message: Optional[str] = None
    ) -> None:
        """
        Update job status and optionally add log message.
        
        Args:
            job_id: Job identifier
            status: New job status
            progress: Optional progress percentage (0-100)
            result_path: Optional path to result file
            error: Optional error message
            log_message: Optional log message to add
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            now = datetime.now()
            
            # Build update query dynamically
            updates = ["status = ?", "updated_at = ?"]
            params = [status.value, now.isoformat()]
            
            if progress is not None:
                updates.append("progress = ?")
                params.append(progress)
            
            if result_path is not None:
                updates.append("result_path = ?")
                params.append(result_path)
            
            if error is not None:
                updates.append("error = ?")
                params.append(error)
            
            params.append(job_id)
            
            cursor.execute(f"""
                UPDATE jobs
                SET {', '.join(updates)}
                WHERE job_id = ?
            """, params)
            
            # Add log message if provided
            if log_message:
                cursor.execute("""
                    INSERT INTO job_logs (job_id, timestamp, message)
                    VALUES (?, ?, ?)
                """, (job_id, now.isoformat(), log_message))
            
            conn.commit()
            conn.close()
            
            logger.debug(f"Job {job_id} updated to {status.value}")
            
        except sqlite3.Error as e:
            logger.error(f"Failed to update job status: {e}")
            raise JobError(f"Failed to update job status: {e}")
    
    def list_jobs(
        self, 
        status: Optional[JobState] = None,
        limit: int = 100
    ) -> List[JobStatus]:
        """
        List jobs, optionally filtered by status.
        
        Args:
            status: Optional status filter
            limit: Maximum number of jobs to return
            
        Returns:
            List of JobStatus objects
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            if status:
                cursor.execute("""
                    SELECT job_id, status, progress, video_url,
                           created_at, updated_at, result_path, error
                    FROM jobs
                    WHERE status = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (status.value, limit))
            else:
                cursor.execute("""
                    SELECT job_id, status, progress, video_url,
                           created_at, updated_at, result_path, error
                    FROM jobs
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (limit,))
            
            rows = cursor.fetchall()
            conn.close()
            
            jobs = []
            for row in rows:
                jobs.append(JobStatus(
                    job_id=row[0],
                    status=JobState(row[1]),
                    progress=row[2],
                    video_url=row[3],
                    created_at=datetime.fromisoformat(row[4]),
                    updated_at=datetime.fromisoformat(row[5]),
                    result_path=row[6],
                    error=row[7],
                    logs=[]  # Don't load logs for list view
                ))
            
            return jobs
            
        except sqlite3.Error as e:
            raise JobError(f"Failed to list jobs: {e}")
    
    def set_job_processor(
        self, 
        processor: Callable[[str, str, JobOptions], TimelineOutput]
    ) -> None:
        """
        Set the job processor callback function.
        
        The processor should accept (job_id, url, options) and return TimelineOutput.
        
        Args:
            processor: Job processing function
        """
        self._job_processor = processor
        logger.info("Job processor callback registered")
    
    def start(self) -> None:
        """
        Start the job queue manager.
        
        This initializes the thread pool and recovers any in-progress jobs.
        """
        if self._executor is not None:
            logger.warning("JobQueueManager already started")
            return
        
        self._shutdown = False
        self._executor = ThreadPoolExecutor(
            max_workers=self.max_concurrent,
            thread_name_prefix="nprd_job_"
        )
        
        logger.info("JobQueueManager started")
        
        # Recover in-progress jobs
        self.recover_jobs()
        
        # Start pending jobs
        self._try_start_pending_jobs()
    
    def stop(self, wait: bool = True) -> None:
        """
        Stop the job queue manager.
        
        Args:
            wait: If True, wait for active jobs to complete
        """
        self._shutdown = True
        
        if self._executor:
            self._executor.shutdown(wait=wait)
            self._executor = None
        
        logger.info("JobQueueManager stopped")
    
    def recover_jobs(self) -> List[str]:
        """
        Recover in-progress jobs after system restart.
        
        Jobs in DOWNLOADING, PROCESSING, or ANALYZING states are reset to PENDING
        and will be restarted.
        
        Returns:
            List of recovered job IDs
        """
        recoverable_states = [
            JobState.DOWNLOADING,
            JobState.PROCESSING,
            JobState.ANALYZING
        ]
        
        recovered_jobs = []
        
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            now = datetime.now()
            
            for state in recoverable_states:
                cursor.execute("""
                    SELECT job_id FROM jobs
                    WHERE status = ?
                """, (state.value,))
                
                job_ids = [row[0] for row in cursor.fetchall()]
                
                for job_id in job_ids:
                    cursor.execute("""
                        UPDATE jobs
                        SET status = ?, updated_at = ?
                        WHERE job_id = ?
                    """, (JobState.PENDING.value, now.isoformat(), job_id))
                    
                    cursor.execute("""
                        INSERT INTO job_logs (job_id, timestamp, message)
                        VALUES (?, ?, ?)
                    """, (
                        job_id,
                        now.isoformat(),
                        f"Job recovered from {state.value} state and reset to PENDING"
                    ))
                    
                    recovered_jobs.append(job_id)
            
            conn.commit()
            conn.close()
            
            if recovered_jobs:
                logger.info(f"Recovered {len(recovered_jobs)} jobs: {recovered_jobs}")
            
            return recovered_jobs
            
        except sqlite3.Error as e:
            logger.error(f"Failed to recover jobs: {e}")
            return []
    
    def _try_start_pending_jobs(self) -> None:
        """
        Try to start pending jobs if resources are available.
        
        This is called automatically when jobs are submitted or completed.
        """
        if self._executor is None or self._shutdown:
            return
        
        with self._lock:
            # Check how many slots are available
            available_slots = self.max_concurrent - len(self._active_jobs)
            
            if available_slots <= 0:
                return
            
            # Get pending jobs
            pending_jobs = self.list_jobs(status=JobState.PENDING, limit=available_slots)
            
            for job_status in pending_jobs:
                if len(self._active_jobs) >= self.max_concurrent:
                    break
                
                self._start_job(job_status.job_id, job_status.video_url)
    
    def _start_job(self, job_id: str, url: str) -> None:
        """
        Start processing a job.
        
        Args:
            job_id: Job identifier
            url: Video URL
        """
        if self._job_processor is None:
            logger.error("No job processor registered, cannot start job")
            return
        
        # Get job options
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            cursor.execute("SELECT options FROM jobs WHERE job_id = ?", (job_id,))
            row = cursor.fetchone()
            conn.close()
            
            if row and row[0]:
                import ast
                options_dict = ast.literal_eval(row[0])
                options = JobOptions(**options_dict)
            else:
                options = JobOptions()
        except Exception as e:
            logger.error(f"Failed to load job options: {e}")
            options = JobOptions()
        
        # Submit to executor
        future = self._executor.submit(self._process_job, job_id, url, options)
        self._active_jobs[job_id] = future
        
        # Add callback to clean up when done
        future.add_done_callback(lambda f: self._job_completed(job_id, f))
        
        logger.info(f"Started job {job_id}")
    
    def _process_job(self, job_id: str, url: str, options: JobOptions) -> None:
        """
        Process a job (runs in thread pool).
        
        Args:
            job_id: Job identifier
            url: Video URL
            options: Job options
        """
        try:
            logger.info(f"Processing job {job_id}")
            
            # Call the job processor
            result = self._job_processor(job_id, url, options)
            
            # Save result
            output_dir = options.output_dir or self.config.output_dir
            output_dir.mkdir(parents=True, exist_ok=True)
            result_path = output_dir / f"{job_id}.json"
            result.save_to_file(result_path)
            
            # Update job status
            self.update_job_status(
                job_id,
                JobState.COMPLETED,
                progress=100,
                result_path=str(result_path),
                log_message="Job completed successfully"
            )
            
            logger.info(f"Job {job_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}", exc_info=True)
            
            self.update_job_status(
                job_id,
                JobState.FAILED,
                error=str(e),
                log_message=f"Job failed with error: {e}"
            )
    
    def _job_completed(self, job_id: str, future: Future) -> None:
        """
        Callback when a job completes (success or failure).
        
        Args:
            job_id: Job identifier
            future: Completed future
        """
        with self._lock:
            if job_id in self._active_jobs:
                del self._active_jobs[job_id]
        
        # Try to start more pending jobs
        self._try_start_pending_jobs()
    
    def get_active_job_count(self) -> int:
        """Get number of currently active jobs."""
        with self._lock:
            return len(self._active_jobs)
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop(wait=True)
