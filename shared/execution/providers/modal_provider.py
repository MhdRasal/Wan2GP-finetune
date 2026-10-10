"""
Modal Cloud Provider for Wan2GP.

Dispatches generation requests to a remote Modal GPU worker.
Supports both Modal Web Endpoints (HTTP webhook) and direct Modal Python SDK function invocation.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Dict, Optional

from shared.execution.providers.base import BaseCloudProvider
from shared.execution.types import ExecutionResult, GenerationRequest


class ModalCloudProvider(BaseCloudProvider):
    """
    Executes generation on Modal (https://modal.com).

    Supports:
    1. Modal HTTP Webhook: POST to MODAL_ENDPOINT or WAN2GP_CLOUD_ENDPOINT
    2. Modal SDK: modal.Function.lookup(app_name, function_name)
    """

    def __init__(
        self,
        endpoint_url: Optional[str] = None,
        app_name: Optional[str] = None,
        function_name: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 3600.0,
        poll_interval: float = 2.0,
        **kwargs: Any,
    ):
        self.endpoint_url = (endpoint_url or os.environ.get("MODAL_ENDPOINT") or os.environ.get("WAN2GP_CLOUD_ENDPOINT", "")).rstrip("/")
        self.app_name = app_name or os.environ.get("MODAL_APP_NAME", "wan2gp-prod3")
        self.function_name = function_name or os.environ.get("MODAL_FUNCTION_NAME", "generate")
        self.api_key = api_key or os.environ.get("MODAL_API_KEY") or os.environ.get("WAN2GP_CLOUD_API_KEY", "")
        self.timeout = float(os.environ.get("WAN2GP_CLOUD_TIMEOUT", str(timeout)))
        self.poll_interval = float(poll_interval)
        self._active_job_id: Optional[str] = None

    def cancel(self, task_id: Optional[Any] = None) -> bool:
        """Cancel an ongoing generation task on Modal."""
        target_id = self._active_job_id or (str(task_id) if task_id else "")
        if not target_id:
            return False
        return self._stop_job_http(target_id)

    def _stop_job_http(self, job_id: str) -> bool:
        if not self.endpoint_url or not job_id:
            return False
        try:
            url = f"{self.endpoint_url}/stop/{job_id}"
            headers = {"Content-Type": "application/json", "User-Agent": "Wan2GP-Modal/1.0"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
                headers["X-API-Key"] = self.api_key
            req = urllib.request.Request(url, data=b"{}", headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=10.0) as resp:
                return resp.status == 200
        except Exception:
            return False

    @property
    def name(self) -> str:
        return "modal"

    def is_configured(self) -> bool:
        if bool(self.endpoint_url):
            return True
        # Check if Modal SDK is available and credentials are set
        has_modal_auth = bool(os.environ.get("MODAL_TOKEN_ID") and os.environ.get("MODAL_TOKEN_SECRET"))
        if has_modal_auth:
            return True
        # Check standard ~/.modal.toml
        modal_config_file = os.path.expanduser("~/.modal.toml")
        return os.path.isfile(modal_config_file)

    def get_configuration_help(self) -> str:
        return (
            "Modal Cloud Provider configuration:\n"
            "Option A (Recommended Web Endpoint):\n"
            "  - Set MODAL_ENDPOINT (e.g. export MODAL_ENDPOINT=https://your-workspace--wan2gp.modal.run)\n"
            "Option B (Modal SDK):\n"
            "  - Set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET\n"
            "  - Set MODAL_APP_NAME (default: wan2gp-runtime)\n"
            "  - Or run 'modal setup' on your system"
        )

    def execute(
        self,
        request: GenerationRequest,
        send_cmd: Optional[Callable[[str, Any], None]] = None,
        **kwargs: Any,
    ) -> ExecutionResult:
        fn_send_cmd = send_cmd if send_cmd is not None else (lambda cmd, data=None: None)
        start_time = time.time()

        if not self.is_configured():
            msg = f"Modal Cloud Provider is not configured.\n\n{self.get_configuration_help()}"
            fn_send_cmd("error", msg)
            return ExecutionResult(success=False, status="failed", task_id=request.task_id, error=msg)

        # 1. Prefer HTTP Web Endpoint if configured
        if self.endpoint_url:
            return self._execute_http(request, fn_send_cmd, start_time)

        # 2. Fallback to Modal SDK
        return self._execute_sdk(request, fn_send_cmd, start_time)

    def _execute_http(
        self,
        request: GenerationRequest,
        fn_send_cmd: Callable[[str, Any], None],
        start_time: float,
    ) -> ExecutionResult:
        job_id = None
        try:
            fn_send_cmd("status", f"Modal: Submitting generation request to {self.endpoint_url}...")
            req_dict = request.to_dict()
            gpu_type = os.environ.get("MODAL_DEFAULT_GPU", "L40S")
            if "settings" in req_dict and isinstance(req_dict["settings"], dict):
                req_dict["settings"]["model_type"] = request.model_type or req_dict.get("model_type", "t2v")
                gpu_type = req_dict["settings"].pop("gpu_type", None) or gpu_type
                if "resolution" in req_dict["settings"]:
                    req_dict["settings"].pop("width", None)
                    req_dict["settings"].pop("height", None)
            url = f"{self.endpoint_url}/generate?gpu_type={gpu_type}"
            payload = json.dumps(req_dict).encode("utf-8")
            headers = {"Content-Type": "application/json", "User-Agent": "Wan2GP-Modal/1.0"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
                headers["X-API-Key"] = self.api_key

            req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))

            job_id = resp_data.get("job_id") or resp_data.get("id")
            self._active_job_id = job_id
            status = str(resp_data.get("status", "queued")).lower()

            # If already completed or failed synchronously
            if status == "completed":
                outputs = resp_data.get("output_files", []) or resp_data.get("outputs", [])
                fn_send_cmd("status", "Modal: Remote generation completed successfully")
                fn_send_cmd("output", None)
                return ExecutionResult(
                    success=True,
                    status="completed",
                    task_id=request.task_id or job_id,
                    output_files=list(outputs),
                    metadata={"provider": "modal", "endpoint": self.endpoint_url},
                    generation_info=resp_data.get("generation_info", {}),
                    execution_time=time.time() - start_time,
                )

            # Asynchronous execution polling
            if status in ("queued", "running") and job_id:
                fn_send_cmd("status", f"Modal: Job {job_id} submitted to cloud GPU ({resp_data.get('gpu_selected', 'auto')}). Polling progress...")
                poll_interval = 2.0
                deadline = time.time() + self.timeout
                last_log_len = 0

                while time.time() < deadline:
                    time.sleep(poll_interval)
                    status_url = f"{self.endpoint_url}/status/{job_id}"
                    status_req = urllib.request.Request(status_url, headers=headers, method="GET")
                    try:
                        with urllib.request.urlopen(status_req, timeout=30.0) as s_resp:
                            s_data = json.loads(s_resp.read().decode("utf-8"))
                    except Exception:
                        continue

                    state = str(s_data.get("status", "running")).lower()

                    # Fetch real-time log lines for progress feedback
                    try:
                        logs_url = f"{self.endpoint_url}/logs/{job_id}"
                        l_req = urllib.request.Request(logs_url, headers=headers, method="GET")
                        with urllib.request.urlopen(l_req, timeout=10.0) as l_resp:
                            logs_text = l_resp.read().decode("utf-8", errors="replace")
                            if len(logs_text) > last_log_len:
                                new_logs = logs_text[last_log_len:].strip()
                                last_log_len = len(logs_text)
                                if new_logs:
                                    for line in new_logs.splitlines():
                                        clean_l = line.strip()
                                        if clean_l:
                                            print(f"[Modal] {clean_l}", flush=True)
                                    last_line = new_logs.splitlines()[-1]
                                    fn_send_cmd("status", f"Modal GPU: {last_line[:120]}")
                    except Exception:
                        pass

                    if state == "completed":
                        fn_send_cmd("status", "Modal: Cloud generation completed. Downloading output files...")
                        output_files = s_data.get("files", [])
                        if not output_files:
                            try:
                                dl_list_url = f"{self.endpoint_url}/download/{job_id}"
                                dl_list_req = urllib.request.Request(dl_list_url, headers=headers, method="GET")
                                with urllib.request.urlopen(dl_list_req, timeout=15.0) as dl_list_resp:
                                    dl_list_data = json.loads(dl_list_resp.read().decode("utf-8"))
                                    output_files = dl_list_data.get("files", [])
                            except Exception:
                                pass

                        save_dir_setting = os.environ.get("WAN2GP_OUTPUT_DIR") or (request.settings.get("save_path") if hasattr(request, "settings") and isinstance(request.settings, dict) else None) or "outputs"
                        out_dir = Path(save_dir_setting)
                        out_dir.mkdir(parents=True, exist_ok=True)
                        downloaded_paths = []

                        for fname in output_files:
                            if fname in ("input.zip", "status.json"):
                                continue
                            dest_file = out_dir / fname
                            dl_url = f"{self.endpoint_url}/download/{job_id}/{urllib.parse.quote(fname)}"
                            dl_req = urllib.request.Request(dl_url, headers=headers, method="GET")
                            try:
                                with urllib.request.urlopen(dl_req, timeout=120.0) as dl_resp:
                                    with open(dest_file, "wb") as f_out:
                                        f_out.write(dl_resp.read())
                                downloaded_paths.append(str(dest_file.resolve()))
                            except Exception as dl_err:
                                fn_send_cmd("status", f"Modal: Warning downloading {fname}: {dl_err}")

                        fn_send_cmd("status", f"Modal: Outputs successfully downloaded to {out_dir}")
                        fn_send_cmd("output", downloaded_paths[0] if downloaded_paths else None)
                        return ExecutionResult(
                            success=True,
                            status="completed",
                            task_id=request.task_id or job_id,
                            output_files=downloaded_paths,
                            metadata={"provider": "modal", "endpoint": self.endpoint_url, "job_id": job_id},
                            execution_time=time.time() - start_time,
                        )

                    if state in ("failed", "error", "stopped"):
                        # Ensure remote worker is confirmed stopped immediately (true serverless)
                        self._stop_job_http(job_id)
                        err_msg = s_data.get("error") or f"Job {job_id} {state} on Modal"
                        fn_send_cmd("error", f"Modal generation error: {err_msg}")
                        return ExecutionResult(
                            success=False,
                            status="failed",
                            task_id=request.task_id or job_id,
                            error=str(err_msg),
                            execution_time=time.time() - start_time,
                        )

                # Deadline reached
                if job_id:
                    self._stop_job_http(job_id)
                timeout_msg = f"Modal generation timed out after {self.timeout}s"
                fn_send_cmd("error", timeout_msg)
                return ExecutionResult(
                    success=False,
                    status="failed",
                    task_id=request.task_id or job_id,
                    error=timeout_msg,
                    execution_time=time.time() - start_time,
                )

            err = resp_data.get("error", "Modal execution returned non-success.")
            if job_id:
                self._stop_job_http(job_id)
            fn_send_cmd("error", f"Modal generation error: {err}")
            return ExecutionResult(
                success=False,
                status="failed",
                task_id=request.task_id,
                error=str(err),
                execution_time=time.time() - start_time,
            )

        except urllib.error.URLError as exc:
            if job_id:
                self._stop_job_http(job_id)
            err_msg = f"Modal connection error: {exc.reason}"
            fn_send_cmd("error", err_msg)
            return ExecutionResult(success=False, status="failed", task_id=request.task_id, error=err_msg, execution_time=time.time() - start_time)
        except (KeyboardInterrupt, GeneratorExit):
            if job_id:
                self._stop_job_http(job_id)
            err_msg = "Generation cancelled / disconnected."
            fn_send_cmd("error", err_msg)
            return ExecutionResult(success=False, status="failed", task_id=request.task_id, error=err_msg, execution_time=time.time() - start_time)
        except Exception as exc:
            if job_id:
                self._stop_job_http(job_id)
            err_msg = f"Modal error: {exc}"
            fn_send_cmd("error", err_msg)
            return ExecutionResult(success=False, status="failed", task_id=request.task_id, error=err_msg, execution_time=time.time() - start_time)
        finally:
            self._active_job_id = None

    def _execute_sdk(
        self,
        request: GenerationRequest,
        fn_send_cmd: Callable[[str, Any], None],
        start_time: float,
    ) -> ExecutionResult:
        try:
            import modal
        except ImportError:
            msg = (
                "Modal Python SDK is not installed.\n"
                "Install it using: pip install modal\n"
                "Or configure MODAL_ENDPOINT to use HTTP webhook mode."
            )
            fn_send_cmd("error", msg)
            return ExecutionResult(success=False, status="failed", task_id=request.task_id, error=msg)

        try:
            fn_send_cmd("status", f"Modal SDK: Looking up function '{self.function_name}' in app '{self.app_name}'...")
            remote_fn = modal.Function.lookup(self.app_name, self.function_name)

            fn_send_cmd("status", "Modal SDK: Spawning remote cloud GPU execution...")
            fn_send_cmd("progress", [(1, 3), "Modal: Invoking remote worker..."])

            payload = request.to_dict()
            result_payload = remote_fn.remote(payload)

            fn_send_cmd("progress", [(3, 3), "Modal: Remote execution completed."])
            fn_send_cmd("status", "Modal: Generation completed successfully")
            fn_send_cmd("output", None)

            outputs = result_payload.get("output_files", []) if isinstance(result_payload, dict) else [str(result_payload)]
            return ExecutionResult(
                success=True,
                status="completed",
                task_id=request.task_id,
                output_files=list(outputs),
                metadata={"provider": "modal", "app_name": self.app_name, "function": self.function_name},
                execution_time=time.time() - start_time,
            )
        except Exception as exc:
            err_msg = f"Modal SDK execution failed: {exc}"
            fn_send_cmd("error", err_msg)
            return ExecutionResult(success=False, status="failed", task_id=request.task_id, error=err_msg, execution_time=time.time() - start_time)
