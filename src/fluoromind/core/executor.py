"""Executor module for managing different execution modes and cluster computing.

This module provides classes for managing different execution modes (sequential,
parallel, cluster) and handling cluster computing configurations.
"""

from typing import Dict, Any, Optional, Union, List, Callable
import os
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import importlib.util
import logging
import asyncio

logger = logging.getLogger(__name__)

# Check for optional dependencies
has_dask = importlib.util.find_spec("dask") is not None
has_dask_jobqueue = importlib.util.find_spec("dask_jobqueue") is not None

if has_dask:
    from dask.distributed import Client, LocalCluster

    if has_dask_jobqueue:
        from dask_jobqueue import SLURMCluster
else:
    Client = LocalCluster = None
    SLURMCluster = None


class ExecutionMode(Enum):
    """Execution modes for pipeline processing."""

    SEQUENTIAL = "sequential"
    PARALLEL_THREAD = "parallel_thread"
    PARALLEL_PROCESS = "parallel_process"
    CLUSTER = "cluster"
    CLOUD = "cloud"


class ExecuteMixin:
    """Manages execution modes and cluster computing for pipeline processing.

    This class provides functionality to:
    - Set up and manage different execution modes
    - Configure and manage cluster computing resources (if available)
    - Handle cleanup of computing resources

    Parameters
    ----------
    execution_mode : ExecutionMode, optional
        The execution mode to use, by default ExecutionMode.SEQUENTIAL
    max_workers : int, optional
        Maximum number of workers for parallel processing
    cluster_config : Dict, optional
        Configuration for cluster computing
    """

    def __init__(
        self,
        execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL,
        max_workers: int = None,
        cluster_config: Optional[Dict] = None,
        retry_count: int = 1,
        retry_delay: float = 1.0,
    ):
        """Initialize ExecuteMixin.

        Parameters
        ----------
        execution_mode : ExecutionMode, optional
            The execution mode to use, by default ExecutionMode.SEQUENTIAL
        max_workers : int, optional
            Maximum number of workers for parallel processing
        cluster_config : Dict, optional
            Configuration for cluster computing
        retry_count : int, optional
            Number of times to retry failed tasks, by default 1
        retry_delay : float, optional
            Delay in seconds between retries, by default 1.0

        Raises
        ------
        ValueError
            If any parameters are invalid
        """
        self._validate_params(execution_mode, max_workers, retry_count, retry_delay)

        if execution_mode == ExecutionMode.CLUSTER and not has_dask:
            logger.warning(
                "Dask is not installed. Cluster execution mode is not available. "
                "Falling back to sequential mode. "
                "Install dask[distributed] to enable cluster computing."
            )
            execution_mode = ExecutionMode.SEQUENTIAL

        self.execution_mode = execution_mode
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        self.cluster_config = cluster_config or {}
        self.retry_count = retry_count
        self.retry_delay = retry_delay
        self._client: Optional[Any] = None

    def _validate_params(
        self, execution_mode: ExecutionMode, max_workers: Optional[int], retry_count: int, retry_delay: float
    ) -> None:
        """Validate initialization parameters.

        Parameters
        ----------
        execution_mode : ExecutionMode
            The execution mode to validate
        max_workers : Optional[int]
            Number of workers to validate
        retry_count : int
            Number of retries to validate
        retry_delay : float
            Retry delay to validate

        Raises
        ------
        ValueError
            If any parameters are invalid
        """
        if not isinstance(execution_mode, ExecutionMode):
            raise ValueError("execution_mode must be an instance of ExecutionMode")

        if max_workers is not None and (not isinstance(max_workers, int) or max_workers <= 0):
            raise ValueError("max_workers must be a positive integer")

        if not isinstance(retry_count, int) or retry_count < 0:
            raise ValueError("retry_count must be a non-negative integer")

        if not isinstance(retry_delay, (int, float)) or retry_delay < 0:
            raise ValueError("retry_delay must be a non-negative number")

    def setup_cluster(self) -> Optional[Any]:
        """Initialize Dask cluster based on configuration.

        Returns
        -------
        Optional[Client]
            Dask distributed client for cluster computing if available

        Raises
        ------
        RuntimeError
            If cluster computing is requested but dependencies are not installed
        ValueError
            If an unsupported cluster type is specified
        """
        if not has_dask:
            raise RuntimeError(
                "Dask is not installed. Cannot use cluster computing. "
                "Install dask[distributed] to enable this feature."
            )

        if self._client is not None:
            return self._client

        cluster_type = self.cluster_config.get("type", "local")

        if cluster_type == "local":
            cluster = LocalCluster(
                n_workers=self.cluster_config.get("n_workers", 4),
                threads_per_worker=self.cluster_config.get("threads_per_worker", 1),
                memory_limit=self.cluster_config.get("memory_limit", "4GB"),
            )
        elif cluster_type == "ssh":
            if not has_dask:
                raise RuntimeError("Dask is required for SSH cluster support")
            from dask.distributed import SSHCluster

            cluster = SSHCluster(
                hosts=self.cluster_config["hosts"],
                connect_options=self.cluster_config.get("connect_options", {}),
                worker_options=self.cluster_config.get("worker_options", {}),
            )
        elif cluster_type == "slurm":
            if not has_dask_jobqueue:
                raise RuntimeError("dask-jobqueue is required for SLURM cluster support")

            cluster = SLURMCluster(
                queue=self.cluster_config.get("queue"),
                project=self.cluster_config.get("project"),
                cores=self.cluster_config.get("cores", 1),
                memory=self.cluster_config.get("memory", "4GB"),
                walltime=self.cluster_config.get("walltime", "01:00:00"),
            )
            cluster.scale(jobs=self.cluster_config.get("n_jobs", 1))
        else:
            raise ValueError(f"Unsupported cluster type: {cluster_type}")

        self._client = Client(cluster)
        return self._client

    def get_executor(self) -> Optional[Union[ThreadPoolExecutor, ProcessPoolExecutor, Any]]:
        """Get the appropriate executor based on execution mode.

        Returns
        -------
        Optional[Union[ThreadPoolExecutor, ProcessPoolExecutor, Client]]
            The executor instance for the current execution mode
        """
        try:
            if self.execution_mode == ExecutionMode.PARALLEL_THREAD:
                return ThreadPoolExecutor(max_workers=self.max_workers)
            elif self.execution_mode == ExecutionMode.PARALLEL_PROCESS:
                return ProcessPoolExecutor(max_workers=self.max_workers)
            elif self.execution_mode == ExecutionMode.CLUSTER:
                return self.setup_cluster()
        except Exception as e:
            logger.warning(f"Failed to create executor: {str(e)}. Falling back to sequential mode.")
            self.execution_mode = ExecutionMode.SEQUENTIAL
            return None

        return None

    def cleanup(self):
        """Cleanup cluster resources."""
        if self._client is not None:
            try:
                self._client.close()
            except Exception as e:
                logger.warning(f"Error during cleanup: {str(e)}")
            finally:
                self._client = None

    async def execute_tasks(self, tasks: List[Callable]) -> List[Any]:
        """Execute a list of tasks using the configured execution mode.

        Parameters
        ----------
        tasks : List[Callable]
            List of task functions to execute
        *args, **kwargs
            Additional arguments passed to each task

        Returns
        -------
        List[Any]
            Results from executing all tasks
        """
        if self.execution_mode == ExecutionMode.CLUSTER:
            return await self._execute_cluster(tasks)
        elif self.execution_mode in {ExecutionMode.PARALLEL_THREAD, ExecutionMode.PARALLEL_PROCESS}:
            return await self._execute_parallel(tasks)
        else:  # SEQUENTIAL or fallback
            return await self._execute_sequential(tasks)

    async def _execute_sequential(self, tasks: List[Callable]) -> List[Any]:
        """Execute tasks sequentially."""
        results = []
        for task in tasks:
            result = task()
            results.append(result)
        return results

    async def _execute_parallel(self, tasks: List[Callable]) -> List[Any]:
        """Execute tasks in parallel using thread or process pool."""
        executor = self.get_executor()
        if executor is None:
            return await self._execute_sequential(tasks)

        loop = asyncio.get_event_loop()
        async_tasks = []

        with executor:
            for task in tasks:
                # 创建一个将任务包装到线程或进程池中的协程
                async_task = loop.run_in_executor(executor, task)
                async_tasks.append(async_task)

            try:
                # 等待所有任务完成
                results = await asyncio.gather(*async_tasks)
                return results
            except Exception as e:
                raise RuntimeError(f"Parallel execution failed: {str(e)}. Falling back to sequential execution.") from e

    async def _execute_cluster(self, tasks: List[Callable]) -> List[Any]:
        """Execute tasks using Dask cluster."""
        try:
            client = self.setup_cluster()
            if client is None:
                raise RuntimeError("Failed to set up cluster. Falling back to sequential execution.")

            futures = []
            for task in tasks:
                future = client.submit(task)
                futures.append(future)

            return await client.gather(futures)

        except Exception as e:
            raise RuntimeError(f"Cluster execution failed: {str(e)}. Falling back to sequential execution.") from e
