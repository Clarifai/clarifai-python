import click

from clarifai.cli.base import cli
from clarifai.utils.cli import validate_context


@cli.command(['list-instances', 'li'], short_help='List available compute instances.')
@click.option('--cloud', default=None, help='Filter by cloud provider (aws, gcp, vultr, azure).')
@click.option('--region', default=None, help='Filter by region (us-east-1, us-central1).')
@click.option('--gpu', default=None, help='Filter by GPU name (A10G, H100, L40S).')
@click.option('--min-gpus', type=int, default=None, help='Minimum GPU count.')
@click.option('--min-gpu-mem', default=None, help='Minimum GPU memory (e.g., 80Gi, 48Gi).')
@click.pass_context
def list_instances(ctx, cloud, region, gpu, min_gpus, min_gpu_mem):
    """List available compute instance types with GPU, memory, and cloud info.

    \b
    Examples:
      clarifai list-instances                          # all instances
      clarifai li --cloud aws                          # AWS only
      clarifai li --gpu H100                           # H100 instances
      clarifai li --min-gpus 2                         # multi-GPU instances
      clarifai li --min-gpu-mem 48Gi                   # 48+ GiB GPU memory
      clarifai li --cloud aws --gpu L40S               # combined filters
    """
    from clarifai.utils.compute_presets import list_gpu_presets

    pat_val = None
    base_url_val = None
    try:
        validate_context(ctx)
        pat_val = ctx.obj.current.pat
        base_url_val = ctx.obj.current.api_base
    except Exception:
        pass

    click.echo(
        list_gpu_presets(
            pat=pat_val,
            base_url=base_url_val,
            cloud_provider=cloud,
            region=region,
            gpu_name=gpu,
            min_gpus=min_gpus,
            min_gpu_mem=min_gpu_mem,
        )
    )
