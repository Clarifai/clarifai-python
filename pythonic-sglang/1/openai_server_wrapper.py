import os
import signal
import subprocess
import sys
import threading
from typing import List

import psutil
from clarifai.utils.logging import logger

PYTHON_EXEC = sys.executable


def kill_process_tree(parent_pid, include_parent: bool = True, skip_pid: int = None):
  """Kill the process and all its child processes."""
  if parent_pid is None:
    parent_pid = os.getpid()
    include_parent = False

  try:
    itself = psutil.Process(parent_pid)
  except psutil.NoSuchProcess:
    return

  children = itself.children(recursive=True)
  for child in children:
    if child.pid == skip_pid:
      continue
    try:
      child.kill()
    except psutil.NoSuchProcess:
      pass

  if include_parent:
    try:
      itself.kill()

      # Sometime processes cannot be killed with SIGKILL (e.g, PID=1 launched by kubernetes),
      # so we send an additional signal to kill them.
      itself.send_signal(signal.SIGQUIT)
    except psutil.NoSuchProcess:
      pass


class OpenAI_APIServer:

  def __init__(self, **kwargs):
    self.server_started_event = threading.Event()
    self.process = None
    self.backend = None
    self.server_thread = None

  def __del__(self, *exc):
    # This is important
    # close the server when exit the program
    self.close()

  def close(self):
    if self.process:
      try:
        kill_process_tree(self.process.pid)
      except:
        self.process.terminate()
    if self.server_thread:
      self.server_thread.join()

  def wait_for_startup(self):
    self.server_started_event.wait()

  def validate_if_server_start(self, line: str):
    line_lower = line.lower()
    if self.backend in ["vllm", "sglang", "lmdeploy"]:
      if self.backend == "vllm":
        return "application startup complete" in line_lower or "vllm api server on" in line_lower
      else:
        return f" running on http://{self.host}:" in line.strip()
    elif self.backend == "llamacpp":
      return "waiting for new tasks" in line_lower
    elif self.backend == "tgi":
      return "Connected" in line.strip()

  def _start_server(self, cmds):
    try:
      env = os.environ.copy()
      env["VLLM_USAGE_SOURCE"] = "production-docker-image"
      self.process = subprocess.Popen(
          cmds,
          stdout=subprocess.PIPE,
          stderr=subprocess.STDOUT,
          text=True,
      )
      for line in self.process.stdout:
        logger.info("Server Log:  " + line.strip())
        if self.validate_if_server_start(line):
          self.server_started_event.set()
          # break
    except Exception as e:
      if self.process:
        self.process.terminate()
      raise RuntimeError(f"Failed to start Server server: {e}")

  def start_server_thread(self, cmds: str):
    try:
      # Start the  server in a separate thread
      self.server_thread = threading.Thread(target=self._start_server, args=(cmds,), daemon=None)
      self.server_thread.start()

      # Wait for the server to start
      self.wait_for_startup()
    except Exception as e:
      raise Exception(e)

  @classmethod
  def from_sglang_backend(
      cls,
      checkpoints,
      dtype: str = "auto",
      kv_cache_dtype: str = "auto",
      tp_size: int = 1,
      quantization: str = None,
      load_format: str = "auto",
      context_length: str = None,
      device: str = "cuda",
      port=23333,
      host="0.0.0.0",
      chat_template: str = None,
      mem_fraction_static: float = 0.8,
      max_running_requests: int = None,
      max_total_tokens: int = None,
      max_prefill_tokens: int = None,
      schedule_policy: str = "fcfs",
      schedule_conservativeness: float = 1.0,
      cpu_offload_gb: int = 0,
      additional_list_args: List[str] = [],
  ):
    """Start SGlang OpenAI compatible server.

    Args:
        checkpoints (str): model id or path.
        dtype (str, optional): Dtype used for the model {"auto", "half", "float16", "bfloat16", "float", "float32"}. Defaults to "auto".
        kv_cache_dtype (str, optional): Dtype of the kv cache, defaults to the dtype. Defaults to "auto".
        tp_size (int, optional): The number of GPUs the model weights get sharded over. Mainly for saving memory rather than for high throughput. Defaults to 1.
        quantization (str, optional): Quantization format {"awq","fp8","gptq","marlin","gptq_marlin","awq_marlin","bitsandbytes","gguf","modelopt","w8a8_int8"}. Defaults to None.
        load_format (str, optional): The format of the model weights to load:\n* `auto`: will try to load the weights in the safetensors format and fall back to the pytorch bin format if safetensors format is not available.\n* `pt`: will load the weights in the pytorch bin format. \n* `safetensors`: will load the weights in the safetensors format. \n* `npcache`: will load the weights in pytorch format and store a numpy cache to speed up the loading. \n* `dummy`: will initialize the weights with random values, which is mainly for profiling.\n* `gguf`: will load the weights in the gguf format. \n* `bitsandbytes`: will load the weights using bitsandbytes quantization."\n* `layered`: loads weights layer by layer so that one can quantize a layer before loading another to make the peak memory envelope smaller.\n. Defaults to "auto".\n
        context_length (str, optional): The model's maximum context length. Defaults to None (will use the value from the model's config.json instead). Defaults to None.
        device (str, optional): The device type {"cuda", "xpu", "hpu", "cpu"}. Defaults to "cuda".
        port (int, optional): Port number. Defaults to 23333.
        host (str, optional): Host name. Defaults to "0.0.0.0".
        chat_template (str, optional): The buliltin chat template name or the path of the chat template file. This is only used for OpenAI-compatible API server.. Defaults to None.
        mem_fraction_static (float, optional): The fraction of the memory used for static allocation (model weights and KV cache memory pool). Use a smaller value if you see out-of-memory errors. Defaults to 0.8.
        max_running_requests (int, optional): The maximum number of running requests.. Defaults to None.
        max_total_tokens (int, optional): The maximum number of tokens in the memory pool. If not specified, it will be automatically calculated based on the memory usage fraction. This option is typically used for development and debugging purposes.. Defaults to None.
        max_prefill_tokens (int, optional): The maximum number of tokens in a prefill batch. The real bound will be the maximum of this value and the model's maximum context length. Defaults to None.
        schedule_policy (str, optional): The scheduling policy of the requests {"lpm", "random", "fcfs", "dfs-weight"}. Defaults to "fcfs".
        schedule_conservativeness (float, optional): How conservative the schedule policy is. A larger value means more conservative scheduling. Use a larger value if you see requests being retracted frequently. Defaults to 1.0.
        cpu_offload_gb (int, optional): How many GBs of RAM to reserve for CPU offloading. Defaults to 0.
        additional_list_args (List[str], optional): additional args to run subprocess cmd e.g. ["--arg-name", "arg value"]. See more at [github](https://github.com/sgl-project/sglang/blob/1baa9e6cf90b30aaa7dae51c01baa25229e8f7d5/python/sglang/srt/server_args.py#L298). Defaults to [].

    Returns:
        _type_: _description_
    """

    from sglang.utils import execute_shell_command, wait_for_server

    cmds = [
        PYTHON_EXEC, '-m', 'sglang.launch_server', '--model-path', checkpoints, '--dtype',
        str(dtype), '--device',
        str(device), '--kv-cache-dtype',
        str(kv_cache_dtype), '--tp-size',
        str(tp_size), '--load-format',
        str(load_format), '--mem-fraction-static',
        str(mem_fraction_static), '--schedule-policy',
        str(schedule_policy), '--schedule-conservativeness',
        str(schedule_conservativeness), '--port',
        str(port), '--host',
        host, "--trust-remote-code"
    ]
    if chat_template:
      cmds += ["--chat-template", chat_template]
    if quantization:
      cmds += [
          '--quantization',
          quantization,
      ]
    if context_length:
      cmds += [
          '--context-length',
          context_length,
      ]
    if max_running_requests:
      cmds += [
          '--max-running-requests',
          max_running_requests,
      ]
    if max_total_tokens:
      cmds += [
          '--max-total-tokens',
          max_total_tokens,
      ]
    if max_prefill_tokens:
      cmds += [
          '--max-prefill-tokens',
          max_prefill_tokens,
      ]

    if additional_list_args:
      cmds += additional_list_args

    print("CMDS to run `sglang` server: ", " ".join(cmds), "\n")
    _self = cls()

    _self.host = host
    _self.port = port
    _self.backend = "sglang"
    # _self.start_server_thread(cmds)
    # new_path = os.environ["PATH"] + ":/sbin"
    # _self.process = subprocess.Popen(cmds, text=True, stderr=subprocess.STDOUT, env={**os.environ, "PATH": new_path})
    _self.process = execute_shell_command(" ".join(cmds))

    logger.info("Waiting for " + f"http://{_self.host}:{_self.port}")
    wait_for_server(f"http://{_self.host}:{_self.port}")
    logger.info("Done")

    return _self

