"""
Local GPU Execution Adapter for Wan2GP.

Executes generation requests directly on the local machine using Wan2GP's
existing high-performance pipelines, memory offloading (mmgp), and attention kernels.
"""
from __future__ import annotations

import time
import traceback
from typing import Any, Callable, Dict, List, Optional

from shared.execution.base import ExecutionAdapter
from shared.execution.types import ExecutionMode, ExecutionResult, GenerationRequest


class LocalExecutionAdapter(ExecutionAdapter):
    """
    Executes generation requests on the local GPU.

    Delegates directly to Wan2GP's `generate_media(...)` pipeline, ensuring
    100% backward compatibility with upstream behavior, offloading, and Gradio events.
    """

    def __init__(self, generation_fn: Optional[Callable[..., Any]] = None):
        """
        Args:
            generation_fn: Callable implementing local generation (typically `wgp.generate_media`).
                           If omitted, imported lazily from `wgp`.
        """
        self._generation_fn = generation_fn

    @property
    def name(self) -> str:
        return "Local GPU Adapter"

    @property
    def mode(self) -> ExecutionMode:
        return ExecutionMode.LOCAL

    def _get_generation_fn(self) -> Callable[..., Any]:
        if self._generation_fn is not None:
            return self._generation_fn
        try:
            import wgp
            return wgp.generate_media
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(f"Could not resolve local generation pipeline: {exc}") from exc

    def execute(
        self,
        request: GenerationRequest,
        send_cmd: Optional[Callable[[str, Any], None]] = None,
        **kwargs: Any,
    ) -> ExecutionResult:
        """
        Execute the normalized request locally.

        Args:
            request: The normalized GenerationRequest.
            send_cmd: Event streaming callback (status, progress, preview, output, error).
            kwargs: Extra parameters passed through if provided.
        """
        fn_send_cmd = send_cmd if send_cmd is not None else (lambda cmd, data=None: None)
        generation_fn = self._get_generation_fn()

        task = request.to_task()
        params = request.to_params(include_runtime_state=True)
        plugin_data = copy_dict = dict(request.plugin_data)

        # Track baseline output count if gen_info is accessible
        state = params.get("state") or request._runtime_state
        initial_file_count = 0
        gen_info = None
        if isinstance(state, dict) and "gen" in state:
            gen_info = state["gen"]
            initial_file_count = len(gen_info.get("file_list", []))

        start_time = time.time()
        try:
            # Execute existing pipeline
            success = generation_fn(
                task,
                fn_send_cmd,
                plugin_data=plugin_data,
                **{k: v for k, v in params.items() if k not in ("task", "send_cmd", "plugin_data")},
            )

            execution_time = time.time() - start_time
            is_successful = bool(success is True or success is None)

            # Discover output files
            output_files: List[str] = []
            if gen_info and "file_list" in gen_info:
                current_files = gen_info["file_list"]
                if len(current_files) > initial_file_count:
                    output_files = list(current_files[initial_file_count:])

            status = "completed" if is_successful else "failed"
            if gen_info and gen_info.get("abort", False):
                status = "aborted"

            return ExecutionResult(
                success=is_successful,
                status=status,
                task_id=request.task_id,
                output_files=output_files,
                metadata={
                    "adapter": self.name,
                    "execution_mode": self.mode.value,
                    "model_type": request.model_type,
                },
                generation_info={"execution_time_seconds": execution_time},
                execution_time=execution_time,
            )

        except Exception as exc:
            execution_time = time.time() - start_time
            error_msg = str(exc)
            traceback.print_exc()
            fn_send_cmd("error", error_msg)
            return ExecutionResult(
                success=False,
                status="failed",
                task_id=request.task_id,
                error=error_msg,
                execution_time=execution_time,
                metadata={"adapter": self.name, "execution_mode": self.mode.value},
            )
