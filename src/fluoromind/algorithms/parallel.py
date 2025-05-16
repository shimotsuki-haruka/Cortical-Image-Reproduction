from typing import Optional, Literal, List, Any, Callable
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor


class ParallelMixin:
    """Mixin class for parallel processing capabilities."""

    def __init__(
        self,
        parallel: bool = True,
        n_jobs: Optional[int] = None,
        parallel_backend: Literal["processes", "threads"] = "processes",
        chunk_size: Optional[int] = None,
        parallel_threshold: int = 1000,
    ):
        """Initialize parallel processing parameters.

        Parameters
        ----------
        parallel : bool, optional
            Whether to use parallel computing, by default True
        n_jobs : int, optional
            Number of parallel jobs. If None, uses all available cores
        parallel_backend : str, optional
            Parallelization method: "processes" or "threads", by default "processes"
        chunk_size : int, optional
            Size of chunks for parallel processing. If None, auto-determined
        parallel_threshold : int, optional
            Minimum size to trigger parallel computing, by default 1000
        """
        self._parallel = parallel
        self._n_jobs = n_jobs if n_jobs is not None else cpu_count()
        self._parallel_backend = parallel_backend
        self._chunk_size = chunk_size
        self._parallel_threshold = parallel_threshold

    def _parallel_map(
        self,
        func: Callable,
        tasks: List[Any],
        chunk_size: Optional[int] = None,
    ) -> List[Any]:
        """Generic parallel mapping function.

        Parameters
        ----------
        func : callable
            Function to apply to each task
        tasks : list
            List of task arguments to process
        chunk_size : int, optional
            Size of chunks for parallel processing

        Returns
        -------
        list
            Results from parallel computation
        """
        if self._parallel_backend == "processes":
            with Pool(processes=self._n_jobs) as pool:
                results = pool.starmap(
                    func,
                    tasks,
                    chunksize=chunk_size or self._chunk_size,
                )
        else:  # threads
            with ThreadPoolExecutor(max_workers=self._n_jobs) as executor:
                results = list(
                    executor.map(
                        lambda args: func(*args),
                        tasks,
                        chunksize=chunk_size or self._chunk_size,
                    )
                )
        return results

    def should_use_parallel(self, size: int) -> bool:
        """Determine if parallel processing should be used based on size.

        Parameters
        ----------
        size : int
            Size of the computation (e.g., matrix size, number of items)

        Returns
        -------
        bool
            True if parallel processing should be used
        """
        return self._parallel and size > self._parallel_threshold
