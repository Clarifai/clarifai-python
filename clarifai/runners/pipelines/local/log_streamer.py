"""Stream pod logs from an Argo Workflow running on the local K8s cluster."""

import subprocess
import sys
import threading
import time

from clarifai.utils.logging import logger

_TERMINAL_PHASES = frozenset({'Succeeded', 'Failed', 'Error'})


def _get_workflow_status(wf_name, namespace):
    """Get the current phase of an Argo Workflow."""
    result = subprocess.run(
        [
            'kubectl',
            'get',
            'workflow',
            wf_name,
            '-n',
            namespace,
            '-o',
            'jsonpath={.status.phase}',
        ],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else ''


def _get_workflow_pods(wf_name, namespace):
    """Get pod names associated with a workflow, mapped by node name."""
    result = subprocess.run(
        [
            'kubectl',
            'get',
            'pods',
            '-n',
            namespace,
            '-l',
            f'workflows.argoproj.io/workflow={wf_name}',
            '-o',
            'jsonpath={range .items[*]}{.metadata.name}{"\\t"}{.metadata.labels.workflows\\.argoproj\\.io/node-name}{"\\n"}{end}',
        ],
        capture_output=True,
        text=True,
    )
    pods = {}
    for line in result.stdout.strip().split('\n'):
        parts = line.strip().split('\t')
        if len(parts) >= 2 and parts[0]:
            pods[parts[0]] = parts[1]
        elif len(parts) == 1 and parts[0]:
            pods[parts[0]] = parts[0]
    return pods


def _stream_pod_logs(pod_name, display_name, namespace):
    """Stream logs from a single pod to stdout with a prefix."""
    # Wait for the pod's main container to be ready
    for _ in range(30):
        check = subprocess.run(
            ['kubectl', 'wait', '--for=condition=Ready', 'pod', pod_name,
             '-n', namespace, '--timeout=5s'],
            capture_output=True, text=True,
        )
        if check.returncode == 0:
            break
        # Also check if the pod already completed
        phase_check = subprocess.run(
            ['kubectl', 'get', 'pod', pod_name, '-n', namespace,
             '-o', 'jsonpath={.status.phase}'],
            capture_output=True, text=True,
        )
        if phase_check.stdout.strip() in ('Succeeded', 'Failed'):
            break
        time.sleep(1)

    try:
        process = subprocess.Popen(
            [
                'kubectl',
                'logs',
                '-f',
                pod_name,
                '-c',
                'main',
                '-n',
                namespace,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for line in process.stdout:
            print(f'[{display_name}] {line}', end='', file=sys.stdout, flush=True)
        process.wait()
    except Exception as e:
        logger.debug(f'Log streaming ended for {pod_name}: {e}')


def stream_workflow_logs(wf_name, namespace='clarifai-local', poll_interval=3, timeout=3600):
    """Monitor a workflow and stream pod logs in real-time.

    Polls for new pods and streams their logs in background threads.
    Blocks until the workflow reaches a terminal phase or times out.

    Returns the final workflow phase (e.g. 'Succeeded', 'Failed').
    """
    logger.info(f'Monitoring workflow {wf_name} in namespace {namespace} ...')

    seen_pods = set()
    log_threads = []
    start_time = time.time()

    while True:
        elapsed = time.time() - start_time
        if elapsed > timeout:
            logger.error(f'Timeout waiting for workflow {wf_name} after {timeout}s.')
            return 'Timeout'

        # Check for new pods and start log streaming
        pods = _get_workflow_pods(wf_name, namespace)
        for pod_name, node_name in pods.items():
            if pod_name not in seen_pods:
                seen_pods.add(pod_name)
                # Use the node name (step name) as the display prefix
                display_name = node_name or pod_name
                logger.info(f'Streaming logs from pod {pod_name} ({display_name}) ...')
                thread = threading.Thread(
                    target=_stream_pod_logs,
                    args=(pod_name, display_name, namespace),
                    daemon=True,
                )
                thread.start()
                log_threads.append(thread)

        # Check workflow status
        phase = _get_workflow_status(wf_name, namespace)
        if phase in _TERMINAL_PHASES:
            # Give log threads a moment to flush
            for thread in log_threads:
                thread.join(timeout=5)
            return phase

        time.sleep(poll_interval)
