"""GPU/compute resource discovery via Clarifai API with hardcoded fallbacks.

This module provides:
1. Dynamic GPU/instance type lookup via ListInstanceTypes API (across all cloud providers)
2. Hardcoded fallback presets for offline / CI usage
3. Auto-create compute cluster & nodepool configs for model deployment
"""

from clarifai.utils.logging import logger

# Hardcoded fallback presets (used when API is unavailable)
FALLBACK_GPU_PRESETS = {
    "CPU": {
        "description": "CPU only (no GPU)",
        "instance_type_id": "cpu-t3a-2xlarge",
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
    return f"deploy-np-{instance_type_id}"


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


def resolve_gpu(gpu_name, pat=None, base_url=None, cloud_provider=None, region=None):
    """Resolve a GPU/instance type name to its full preset info.

    Accepts either:
    - Instance type IDs from the API (e.g. 'g5.xlarge', 'g6e.2xlarge', 't3a.2xlarge')
    - GPU shorthand names (e.g. 'A10G', 'L40S', 'CPU') as fallback aliases

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
        matched = _match_gpu_name_to_instance_type(gpu_name.upper(), filtered)
        if matched:
            return _instance_type_to_preset(matched)

        # If filtering narrowed results too much, try unfiltered
        if filtered != instance_types:
            for it in instance_types:
                if it.id.lower() == gpu_name.lower():
                    return _instance_type_to_preset(it)
            matched = _match_gpu_name_to_instance_type(gpu_name.upper(), instance_types)
            if matched:
                return _instance_type_to_preset(matched)

    # Fallback to hardcoded presets (by shorthand name)
    gpu_upper = gpu_name.upper()
    if gpu_upper in FALLBACK_GPU_PRESETS:
        return dict(FALLBACK_GPU_PRESETS[gpu_upper])

    available = "Run 'clarifai model deploy --gpu-info' to see available options."
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
    header = "Available instance types (use the ID with --gpu flag):\n"

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
        sorted_types = sorted(
            deduped,
            key=lambda it: (
                it.cloud_provider.id if it.cloud_provider else "",
                -(it.compute_info.num_accelerators or 0),
                it.id,
            ),
        )
        for it in sorted_types:
            ci = it.compute_info
            acc_type = ", ".join(ci.accelerator_type) if ci.accelerator_type else "-"
            gpu_mem = ci.accelerator_memory if ci.accelerator_memory else "-"
            cloud = it.cloud_provider.id if it.cloud_provider else "-"
            rgn = it.region or "-"
            rows.append(
                {
                    "--gpu value": it.id,
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
    example = "\nExample: clarifai model deploy ./my-model --gpu g5.xlarge"
    return header + table + example


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
            "cluster_type": "k8s",
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
