"""GPU/compute resource discovery via Clarifai API with hardcoded fallbacks.

This module provides:
1. Dynamic GPU/instance type lookup via ListInstanceTypes API (across all cloud providers)
2. Hardcoded fallback presets for offline / CI usage
3. Auto-create compute cluster & nodepool configs for model deployment
"""

import re

import requests

from clarifai.utils.logging import logger

# Kubernetes quantity suffixes and their multipliers (in bytes or millicores)
_K8S_SUFFIXES = {
    '': 1,
    'm': 0.001,  # millicores (CPU)
    'k': 1e3,
    'K': 1e3,
    'Ki': 1024,
    'M': 1e6,
    'Mi': 1024**2,
    'G': 1e9,
    'Gi': 1024**3,
    'T': 1e12,
    'Ti': 1024**4,
}


def parse_k8s_quantity(value):
    """Parse a Kubernetes quantity string to a numeric value.

    Handles formats like: "24Gi", "16Mi", "4", "100m", "4.5", "1500Mi", "3Gi"
    Returns a float (bytes for memory, cores for CPU).

    Args:
        value: K8s quantity string or numeric value.

    Returns:
        float: Parsed numeric value, or 0 if parsing fails.
    """
    if value is None:
        return 0
    if isinstance(value, (int, float)):
        return float(value)

    value = str(value).strip()
    if not value:
        return 0

    match = re.match(r'^([0-9]*\.?[0-9]+)\s*([A-Za-z]*)$', value)
    if not match:
        return 0

    number = float(match.group(1))
    suffix = match.group(2)

    multiplier = _K8S_SUFFIXES.get(suffix)
    if multiplier is None:
        return 0

    return number * multiplier


# Hardcoded fallback presets (used when API is unavailable)
FALLBACK_GPU_PRESETS = {
    "CPU": {
        "description": "CPU only (no GPU)",
        "instance_type_id": "t3a.2xlarge",
        "cloud_provider": "aws",
        "region": "us-east-1",
        "inference_compute_info": {
            "cpu_limit": "4",
            "cpu_memory": "16Gi",
            "num_accelerators": 0,
            "accelerator_type": [],
            "accelerator_memory": "",
        },
    },
    "A10G": {
        "description": "NVIDIA A10G 24GB",
        "instance_type_id": "gpu-nvidia-a10g",
        "cloud_provider": "aws",
        "region": "us-east-1",
        "inference_compute_info": {
            "cpu_limit": "4",
            "cpu_memory": "16Gi",
            "num_accelerators": 1,
            "accelerator_type": ["NVIDIA-A10G"],
            "accelerator_memory": "24Gi",
        },
    },
    "L40S": {
        "description": "NVIDIA L40S 48GB",
        "instance_type_id": "gpu-nvidia-l40s",
        "cloud_provider": "aws",
        "region": "us-east-1",
        "inference_compute_info": {
            "cpu_limit": "8",
            "cpu_memory": "32Gi",
            "num_accelerators": 1,
            "accelerator_type": ["NVIDIA-L40S"],
            "accelerator_memory": "48Gi",
        },
    },
    "G6E": {
        "description": "NVIDIA L40S 2x48GB",
        "instance_type_id": "gpu-nvidia-g6e-2x-large",
        "cloud_provider": "aws",
        "region": "us-east-1",
        "inference_compute_info": {
            "cpu_limit": "16",
            "cpu_memory": "64Gi",
            "num_accelerators": 2,
            "accelerator_type": ["NVIDIA-L40S"],
            "accelerator_memory": "96Gi",
        },
    },
}


def get_deploy_compute_cluster_id(cloud_provider="aws", region="us-east-1"):
    """Return a deterministic compute cluster ID for the given cloud/region."""
    return f"deploy-cc-{cloud_provider}-{region}"


def get_deploy_nodepool_id(instance_type_id):
    """Return a deterministic nodepool ID for the given instance type."""
    import re

    # Sanitize: only alphanumeric, hyphens, underscores allowed in IDs
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '-', instance_type_id)
    # Collapse consecutive hyphens
    sanitized = re.sub(r'-{2,}', '-', sanitized)
    return f"deploy-np-{sanitized}"


# Module-level cache for instance types (avoids repeated API calls in one session)
_instance_types_cache = None


def _try_list_all_instance_types(pat=None, base_url=None):
    """Fetch instance types across all cloud providers and regions.

    Queries the API for all available cloud providers, their regions,
    and the instance types in each. Results are cached for the session.

    Returns:
        list of InstanceType protos (with cloud_provider and region set), or None on failure.
    """
    global _instance_types_cache
    if _instance_types_cache is not None:
        return _instance_types_cache

    try:
        from clarifai.client.user import User

        user = User(pat=pat, base_url=base_url) if (pat or base_url) else User()

        all_instance_types = []
        providers = user.list_cloud_providers()

        for provider in providers:
            try:
                regions = user.list_cloud_regions(provider.id)
                for region in regions:
                    region_id = getattr(region, 'id', None) or str(region)
                    try:
                        instance_types = user.list_instance_types(provider.id, region_id)
                        all_instance_types.extend(instance_types)
                    except Exception as e:
                        logger.debug(
                            f"Failed to list instance types for {provider.id}/{region_id}: {e}"
                        )
            except Exception as e:
                logger.debug(f"Failed to list regions for {provider.id}: {e}")

        if all_instance_types:
            _instance_types_cache = all_instance_types
            return all_instance_types
        return None
    except Exception as e:
        logger.debug(f"Failed to fetch instance types from API: {e}")
        return None


def _try_list_instance_types(cloud_provider="aws", region="us-east-1", pat=None, base_url=None):
    """Fetch instance types for a specific cloud provider and region.

    Returns list of InstanceType protos, or None on failure.
    """
    try:
        from clarifai.client.user import User

        user = User(pat=pat, base_url=base_url) if (pat or base_url) else User()
        return user.list_instance_types(cloud_provider, region)
    except Exception as e:
        logger.debug(f"Failed to fetch instance types from API: {e}")
        return None


def _sort_instance_types(instance_types):
    """Sort instance types by cloud provider, GPU count desc, then ID.

    This ensures consistent priority: aws before gcp before vultr,
    matching the display order of --instance-info.
    """
    return sorted(
        instance_types,
        key=lambda it: (
            it.cloud_provider.id if it.cloud_provider else "",
            -(it.compute_info.num_accelerators if it.compute_info else 0),
            it.id,
        ),
    )


def _match_gpu_name_to_instance_type(gpu_name, instance_types):
    """Match a GPU shorthand name (e.g. 'A10G') to an API InstanceType."""
    gpu_upper = gpu_name.upper()
    for it in instance_types:
        it_id_upper = it.id.upper()
        # Match by GPU name in instance type ID
        if gpu_upper in it_id_upper:
            return it
        # Match by accelerator type
        if it.compute_info and it.compute_info.accelerator_type:
            for acc_type in it.compute_info.accelerator_type:
                if gpu_upper in acc_type.upper():
                    return it
    return None


def _instance_type_to_preset(instance_type):
    """Convert an API InstanceType proto to a preset dict."""
    ci = instance_type.compute_info
    return {
        "description": instance_type.description or instance_type.id,
        "instance_type_id": instance_type.id,
        "cloud_provider": instance_type.cloud_provider.id if instance_type.cloud_provider else "",
        "region": instance_type.region or "",
        "inference_compute_info": {
            "cpu_limit": ci.cpu_limit or "4",
            "cpu_memory": ci.cpu_memory or "16Gi",
            "num_accelerators": ci.num_accelerators,
            "accelerator_type": list(ci.accelerator_type) if ci.accelerator_type else [],
            "accelerator_memory": ci.accelerator_memory or "",
        },
    }


def _normalize_gpu_name(gpu_name):
    """Extract GPU shorthand from various formats.

    Handles formats like:
    - 'gpu-nvidia-a10g' → 'A10G'
    - 'gpu-nvidia-g6e-2x-large' → 'G6E-2X-LARGE'
    - 'A10G' → 'A10G' (already short)
    - 'g5.xlarge' → 'G5.XLARGE' (no prefix to strip)
    """
    name = gpu_name.strip()
    # Strip 'gpu-nvidia-' prefix if present
    lower = name.lower()
    if lower.startswith("gpu-nvidia-"):
        name = name[len("gpu-nvidia-") :]
    elif lower.startswith("gpu-"):
        name = name[len("gpu-") :]
    return name.upper()


def resolve_gpu(gpu_name, pat=None, base_url=None, cloud_provider=None, region=None):
    """Resolve a GPU/instance type name to its full preset info.

    Accepts either:
    - Instance type IDs from the API (e.g. 'g5.xlarge', 'g6e.2xlarge', 't3a.2xlarge')
    - GPU shorthand names (e.g. 'A10G', 'L40S', 'CPU') as fallback aliases
    - Legacy nodepool-style names (e.g. 'gpu-nvidia-a10g') — normalized to GPU shorthand

    Queries all cloud providers/regions unless cloud_provider/region are specified.
    If --cloud/--region are given, only that provider+region is queried.

    Args:
        gpu_name: Instance type ID or GPU shorthand name.
        pat: Optional PAT for API auth.
        base_url: Optional API base URL.
        cloud_provider: Optional cloud provider filter (e.g. 'aws', 'gcp').
        region: Optional region filter (e.g. 'us-east-1').

    Returns:
        dict with keys: description, instance_type_id, cloud_provider, region, inference_compute_info

    Raises:
        ValueError: If GPU name is not found.
    """
    # If user specified cloud/region, query only that combination
    if cloud_provider and region:
        instance_types = _try_list_instance_types(
            cloud_provider, region, pat=pat, base_url=base_url
        )
    else:
        # Query all providers/regions
        instance_types = _try_list_all_instance_types(pat=pat, base_url=base_url)

    if instance_types:
        # Sort for consistent priority (aws first, matching --instance-info order)
        instance_types = _sort_instance_types(instance_types)

        # Optionally filter by cloud_provider (even when querying all)
        filtered = instance_types
        if cloud_provider:
            filtered = [
                it
                for it in instance_types
                if (it.cloud_provider and it.cloud_provider.id == cloud_provider)
            ]
        if region:
            filtered = [it for it in filtered if it.region == region]

        # 1. Exact match by instance type ID (e.g. 'g5.xlarge')
        for it in filtered:
            if it.id.lower() == gpu_name.lower():
                return _instance_type_to_preset(it)

        # 2. Fuzzy match by GPU shorthand in instance type ID or accelerator type
        #    Normalize 'gpu-nvidia-a10g' → 'A10G' for better matching
        normalized = _normalize_gpu_name(gpu_name)
        matched = _match_gpu_name_to_instance_type(normalized, filtered)
        if matched:
            return _instance_type_to_preset(matched)

        # If filtering narrowed results too much, try unfiltered
        if filtered != instance_types:
            for it in instance_types:
                if it.id.lower() == gpu_name.lower():
                    return _instance_type_to_preset(it)
            matched = _match_gpu_name_to_instance_type(normalized, instance_types)
            if matched:
                return _instance_type_to_preset(matched)

    # Fallback to hardcoded presets (by shorthand name, then by instance_type_id)
    gpu_upper = gpu_name.upper()
    if gpu_upper in FALLBACK_GPU_PRESETS:
        return dict(FALLBACK_GPU_PRESETS[gpu_upper])
    gpu_lower = gpu_name.lower()
    for preset in FALLBACK_GPU_PRESETS.values():
        if preset["instance_type_id"].lower() == gpu_lower:
            return dict(preset)

    available = "Run 'clarifai model deploy --instance-info' to see available options."
    raise ValueError(f"Unknown instance type '{gpu_name}'. {available}")


def get_inference_compute_for_gpu(gpu_name, pat=None, base_url=None):
    """Get inference_compute_info dict for a GPU name.

    Args:
        gpu_name: GPU shorthand name (e.g. 'A10G').
        pat: Optional PAT for API auth.
        base_url: Optional API base URL.

    Returns:
        dict: inference_compute_info suitable for config.yaml.
    """
    preset = resolve_gpu(gpu_name, pat=pat, base_url=base_url)
    return dict(preset["inference_compute_info"])


def infer_gpu_from_config(config):
    """Infer GPU shorthand name from an existing inference_compute_info config.

    Args:
        config: dict with inference_compute_info section.

    Returns:
        str or None: GPU name like 'A10G', or None if not recognized.
    """
    ici = config.get("inference_compute_info")
    if not ici:
        return None

    acc_types = ici.get("accelerator_type", [])
    if not acc_types or ici.get("num_accelerators", 0) == 0:
        return "CPU"

    # Try to match accelerator types against known presets
    for gpu_name, preset in FALLBACK_GPU_PRESETS.items():
        if gpu_name == "CPU":
            continue
        preset_acc = preset["inference_compute_info"].get("accelerator_type", [])
        preset_num = preset["inference_compute_info"].get("num_accelerators", 0)
        if (
            preset_acc
            and set(preset_acc) == set(acc_types)
            and preset_num == ici.get("num_accelerators", 0)
        ):
            return gpu_name

    return None


def list_gpu_presets(pat=None, base_url=None, cloud_provider=None, region=None):
    """Return a formatted table of available GPU presets.

    Queries all cloud providers/regions via the API, falls back to hardcoded presets.

    Args:
        pat: Optional PAT for API auth.
        base_url: Optional API base URL.
        cloud_provider: Optional filter by cloud provider (e.g. 'aws', 'gcp').
        region: Optional filter by region (e.g. 'us-east-1').

    Returns:
        str: Formatted table string.
    """
    rows = []
    header = "Available instance types (use the ID with --instance flag):\n"

    # Try API first (all providers/regions)
    instance_types = _try_list_all_instance_types(pat=pat, base_url=base_url)
    if instance_types:
        # Filter by cloud/region if specified
        filtered = instance_types
        if cloud_provider:
            filtered = [
                it
                for it in filtered
                if (it.cloud_provider and it.cloud_provider.id == cloud_provider)
            ]
            header = f"Available instance types for {cloud_provider}"
            if region:
                header += f" / {region}"
            header += ":\n"
        if region:
            filtered = [it for it in filtered if it.region == region]

        # Deduplicate by (instance_type_id, cloud_provider) - keep first per combo
        seen = set()
        deduped = []
        for it in filtered:
            cp = it.cloud_provider.id if it.cloud_provider else ""
            key = (it.id, cp)
            if key not in seen:
                seen.add(key)
                deduped.append(it)

        # Sort: cloud first, then GPU count desc, then ID
        sorted_types = _sort_instance_types(deduped)
        for it in sorted_types:
            ci = it.compute_info
            acc_type = ", ".join(ci.accelerator_type) if ci.accelerator_type else "-"
            gpu_mem = ci.accelerator_memory if ci.accelerator_memory else "-"
            cloud = it.cloud_provider.id if it.cloud_provider else "-"
            rgn = it.region or "-"
            rows.append(
                {
                    "--instance value": it.id,
                    "Cloud": cloud,
                    "Region": rgn,
                    "GPUs": ci.num_accelerators or 0,
                    "Accelerator": acc_type,
                    "GPU Memory": gpu_mem,
                    "CPU": ci.cpu_limit,
                    "CPU Memory": ci.cpu_memory,
                }
            )
    else:
        return (
            "Could not fetch instance types from API.\nMake sure you are logged in: clarifai login"
        )

    from tabulate import tabulate

    table = tabulate(rows, headers="keys", tablefmt="simple")
    example = "\nExample: clarifai model deploy ./my-model --instance a10g"
    return header + table + example


def _get_hf_model_info(repo_id):
    """Fetch model metadata from HuggingFace API.

    Returns dict with: num_params, quant_method, quant_bits, dtype_breakdown, pipeline_tag.
    Returns None on failure.
    """
    try:
        url = f"https://huggingface.co/api/models/{repo_id}"
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        logger.debug(f"Failed to fetch HF model info for {repo_id}")
        return None

    result = {
        "num_params": None,
        "quant_method": None,
        "quant_bits": None,
        "dtype_breakdown": None,
        "pipeline_tag": data.get("pipeline_tag"),
    }

    # Extract parameter count and dtype breakdown from safetensors metadata
    safetensors = data.get("safetensors")
    if safetensors:
        result["num_params"] = safetensors.get("total")
        params_by_dtype = safetensors.get("parameters")
        if params_by_dtype:
            result["dtype_breakdown"] = dict(params_by_dtype)

    # Extract quantization config
    config = data.get("config") or {}
    quant_config = config.get("quantization_config") or {}
    if quant_config:
        result["quant_method"] = quant_config.get("quant_method")
        result["quant_bits"] = quant_config.get("bits")

    return result


def _detect_quant_from_repo_name(repo_id):
    """Detect quantization from repo name. Returns (quant_method, bits) or (None, None)."""
    name = repo_id.lower()
    patterns = [
        ("-awq", "awq", 4),
        ("-gptq", "gptq", 4),
        ("-bnb-4bit", "bnb", 4),
        ("-int8", None, 8),
        ("-int4", None, 4),
        ("-4bit", None, 4),
        ("-fp16", "fp16", 16),
    ]
    for suffix, method, bits in patterns:
        if suffix in name:
            return (method, bits)
    return (None, None)


# Bytes per parameter by dtype/quantization
_BYTES_PER_PARAM = {
    "BF16": 2.0,
    "F16": 2.0,
    "FP16": 2.0,
    "F32": 4.0,
    "FP32": 4.0,
    "I8": 1.0,
    "I32": 4.0,
    "U8": 1.0,
}

# Framework overhead (2 GiB) for CUDA context, KV cache init, etc.
_FRAMEWORK_OVERHEAD_BYTES = 2 * 1024**3
# KV cache overhead as fraction of model weight
_KV_CACHE_FRACTION = 0.20


def _estimate_vram_bytes(num_params, quant_method=None, quant_bits=None, dtype_breakdown=None):
    """Estimate VRAM bytes needed for inference. Returns int."""
    # If we have per-dtype parameter counts, compute a weighted sum
    if dtype_breakdown:
        weight_bytes = 0
        for dtype, count in dtype_breakdown.items():
            bpp = _BYTES_PER_PARAM.get(dtype.upper(), 2.0)
            weight_bytes += count * bpp
    elif quant_method in ("awq", "gptq"):
        bpp = 0.5 if (quant_bits is None or quant_bits == 4) else 1.0
        weight_bytes = num_params * bpp
    elif quant_bits:
        bpp = quant_bits / 8.0
        weight_bytes = num_params * bpp
    else:
        # Default: BF16
        weight_bytes = num_params * 2.0

    # Total: weights + KV cache overhead + framework overhead
    return int(weight_bytes + (weight_bytes * _KV_CACHE_FRACTION) + _FRAMEWORK_OVERHEAD_BYTES)


# Pre-Ampere GPU indicators (compute capability < 8.0).
# SGLang requires Ampere+ for CUDA graph capture (RMSNorm kernels).
# Checked case-insensitively against instance IDs across all clouds:
#   AWS: "g4dn.xlarge", "p3.2xlarge"
#   Azure: "Standard_NC4as_T4_v3"
#   GCP: "n1-standard-4-nvidia-tesla-t4"
_PRE_AMPERE_INDICATORS = ("t4", "v100", "k80", "p100", "p40", "m60", "g4dn", "g4ad", "p3.", "p2.")


def _select_instance_by_vram(vram_bytes, pat=None, base_url=None, exclude_pre_ampere=False):
    """Select smallest instance whose VRAM >= vram_bytes.

    Args:
        vram_bytes: Minimum required VRAM in bytes.
        pat: Clarifai PAT for API lookups.
        base_url: Clarifai API base URL.
        exclude_pre_ampere: If True, skip pre-Ampere instances (T4, V100, etc.).
            Required by SGLang which needs compute capability >= 8.0.

    Returns (instance_type_id, reason) or (None, reason).
    """
    vram_gib = vram_bytes / (1024**3)

    def _is_excluded(inst_id):
        if not exclude_pre_ampere:
            return False
        inst_lower = inst_id.lower()
        return any(indicator in inst_lower for indicator in _PRE_AMPERE_INDICATORS)

    # Try API first for real available instances
    instance_types = _try_list_all_instance_types(pat=pat, base_url=base_url)
    if instance_types:
        # Build list of (instance_id, vram_bytes) for GPU instances, sorted by VRAM ascending
        gpu_instances = []
        for it in instance_types:
            ci = it.compute_info if it.compute_info else None
            if not ci or not ci.num_accelerators or ci.num_accelerators == 0:
                continue
            acc_mem = ci.accelerator_memory
            if not acc_mem:
                continue
            mem_bytes = parse_k8s_quantity(acc_mem)
            if mem_bytes > 0:
                gpu_instances.append((it.id, mem_bytes))

        # Deduplicate by instance ID, keeping largest VRAM for each
        seen = {}
        for inst_id, mem in gpu_instances:
            if inst_id not in seen or mem > seen[inst_id]:
                seen[inst_id] = mem
        sorted_instances = sorted(seen.items(), key=lambda x: x[1])

        for inst_id, mem in sorted_instances:
            if _is_excluded(inst_id):
                continue
            if mem >= vram_bytes:
                mem_gib = mem / (1024**3)
                return (
                    inst_id,
                    f"Estimated {vram_gib:.1f} GiB VRAM, fits {inst_id} ({mem_gib:.0f} GiB)",
                )

        if sorted_instances:
            max_gib = sorted_instances[-1][1] / (1024**3)
            return (
                None,
                f"Estimated {vram_gib:.1f} GiB VRAM, exceeds max available {max_gib:.0f} GiB",
            )

    # Fallback to hardcoded GPU tiers
    fallback_tiers = [
        ("gpu-nvidia-a10g", 24 * 1024**3),  # A10G: 24 GiB
        ("gpu-nvidia-l40s", 48 * 1024**3),  # L40S: 48 GiB
        ("gpu-nvidia-g6e-2x-large", 96 * 1024**3),  # G6E 2x: 96 GiB
    ]
    for inst_id, mem in fallback_tiers:
        if _is_excluded(inst_id):
            continue
        if mem >= vram_bytes:
            mem_gib = mem / (1024**3)
            return (
                inst_id,
                f"Estimated {vram_gib:.1f} GiB VRAM, fits {inst_id} ({mem_gib:.0f} GiB)",
            )

    return (None, f"Estimated {vram_gib:.1f} GiB VRAM, exceeds max 96 GiB")


def _detect_toolkit_from_config(config, model_path=None):
    """Detect inference toolkit from build_info.image or requirements.txt.

    Checks build_info.image first (e.g. "lmsysorg/sglang:latest"), then
    falls back to scanning requirements.txt for known toolkit packages.

    Returns toolkit name ('vllm', 'sglang') or empty string.
    """
    # Check build_info.image
    build_image = (config.get('build_info', {}).get('image') or '').lower()
    if 'sglang' in build_image:
        return 'sglang'
    elif 'vllm' in build_image:
        return 'vllm'

    # Check requirements.txt
    if model_path:
        try:
            from clarifai.utils.cli import parse_requirements

            deps = parse_requirements(model_path)
            for name in ('vllm', 'sglang'):
                if name in deps:
                    return name
        except Exception:
            pass

    return ''


def recommend_instance(config, pat=None, base_url=None, toolkit=None, model_path=None):
    """Recommend instance type based on model config.

    Args:
        config: Parsed config.yaml dict.
        pat: Clarifai PAT for API lookups.
        base_url: Clarifai API base URL.
        toolkit: Explicit toolkit name (e.g. 'vllm', 'sglang'). If not provided,
            detected from build_info.image or requirements.txt.
        model_path: Path to model directory (for requirements.txt-based toolkit detection).

    Returns (instance_type_id, reason) or (None, reason).
    """
    model_config = config.get('model', {})
    model_type_id = model_config.get('model_type_id', '')

    # MCP models run on CPU
    if model_type_id in ("mcp", "mcp-stdio"):
        return ("t3a.2xlarge", "MCP models run on CPU")

    checkpoints = config.get('checkpoints', {})
    repo_id = checkpoints.get('repo_id') if checkpoints else None

    if not toolkit:
        toolkit = _detect_toolkit_from_config(config, model_path=model_path)

    if not repo_id:
        # Check if this is a GPU toolkit (vllm/sglang) that needs a repo_id
        if toolkit in ('vllm', 'sglang'):
            return (None, "Cannot estimate without checkpoints.repo_id")
        # No checkpoints, no GPU toolkit → default to CPU
        return ("t3a.2xlarge", "No model checkpoints, defaulting to CPU")

    # SGLang requires Ampere+ GPUs (compute capability >= 8.0).
    # Skip pre-Ampere instances like T4 (g4dn) which fail with CUDA graph errors.
    exclude_pre_ampere = toolkit == 'sglang'

    # Try HF metadata API for parameter count + quantization
    hf_info = _get_hf_model_info(repo_id)
    num_params = hf_info.get("num_params") if hf_info else None

    if num_params:
        quant_method = hf_info.get("quant_method")
        quant_bits = hf_info.get("quant_bits")
        dtype_breakdown = hf_info.get("dtype_breakdown")

        # Also check repo name for quantization hints if API didn't report any
        if not quant_method:
            name_method, name_bits = _detect_quant_from_repo_name(repo_id)
            if name_method:
                quant_method = name_method
                quant_bits = name_bits

        vram = _estimate_vram_bytes(num_params, quant_method, quant_bits, dtype_breakdown)
        return _select_instance_by_vram(
            vram, pat=pat, base_url=base_url, exclude_pre_ampere=exclude_pre_ampere
        )

    # Fallback: file-size-based estimate via HuggingFaceLoader
    try:
        from clarifai.runners.utils.loader import HuggingFaceLoader

        file_size = HuggingFaceLoader.get_huggingface_checkpoint_total_size(repo_id)
        if file_size and file_size > 0:
            # File size ≈ model weights; add 30% overhead for runtime buffers + KV cache
            vram = int(file_size * 1.3) + _FRAMEWORK_OVERHEAD_BYTES
            return _select_instance_by_vram(
                vram, pat=pat, base_url=base_url, exclude_pre_ampere=exclude_pre_ampere
            )
    except Exception:
        logger.debug(f"Failed to get checkpoint size for {repo_id}")

    return (None, "Could not determine model size for " + repo_id)


def get_compute_cluster_config(user_id, cloud_provider="aws", region="us-east-1"):
    """Get auto-create config for a compute cluster.

    Args:
        user_id: The user ID for the compute cluster.
        cloud_provider: Cloud provider ID (e.g. 'aws', 'gcp', 'vultr').
        region: Region ID (e.g. 'us-east-1', 'us-central1').

    Returns:
        dict: Compute cluster config suitable for User.create_compute_cluster().
    """
    return {
        "compute_cluster": {
            "id": get_deploy_compute_cluster_id(cloud_provider, region),
            "description": f"Auto-created compute cluster for {cloud_provider}/{region}",
            "cloud_provider": {"id": cloud_provider},
            "region": region,
            "managed_by": "clarifai",
            "cluster_type": "dedicated",
        }
    }


def get_nodepool_config(instance_type_id, compute_cluster_id, user_id, compute_info=None):
    """Build nodepool config from instance type info.

    Args:
        instance_type_id: The instance type ID (e.g. 'gpu-nvidia-a10g').
        compute_cluster_id: The compute cluster ID.
        user_id: The user ID that owns the compute cluster.
        compute_info: Optional dict of compute info. If None, minimal config is used.

    Returns:
        dict: Nodepool config suitable for ComputeCluster.create_nodepool().
    """
    instance_type = {"id": instance_type_id}
    if compute_info:
        instance_type["compute_info"] = compute_info

    return {
        "nodepool": {
            "id": get_deploy_nodepool_id(instance_type_id),
            "description": f"Auto-created nodepool for {instance_type_id}",
            "compute_cluster": {
                "id": compute_cluster_id,
                "user_id": user_id,
            },
            "instance_types": [instance_type],
            "node_capacity_type": {
                "capacity_types": [1],
            },
            "min_instances": 0,
            "max_instances": 5,
        }
    }
