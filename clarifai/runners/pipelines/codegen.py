import ast
import inspect
import os
import textwrap
from typing import Any, Dict, List, Sequence, Set

import yaml

from clarifai.versions import CLIENT_VERSION


def _ensure_list_requirements(requirements) -> List[str]:
    if requirements is None:
        entries = []
    elif isinstance(requirements, str):
        if os.path.exists(requirements):
            with open(requirements, 'r', encoding='utf-8') as handle:
                entries = [line.strip() for line in handle.readlines() if line.strip()]
        else:
            entries = [requirements]
    else:
        entries = [str(item).strip() for item in requirements if str(item).strip()]

    if not any(entry.startswith('clarifai') for entry in entries):
        entries.insert(0, f'clarifai=={CLIENT_VERSION}')
    return entries


def _get_node_source(source_lines: Sequence[str], node: ast.AST) -> str:
    return ''.join(source_lines[node.lineno - 1:node.end_lineno])


def _collect_helper_functions(tree: ast.Module, source_lines: Sequence[str], root_name: str) -> List[str]:
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    imports = [
        _get_node_source(source_lines, node)
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]

    visited: Set[str] = set()
    ordered: List[str] = []

    def visit(name: str):
        node = functions.get(name)
        if node is None or name in visited:
            return
        visited.add(name)
        for child in ast.walk(node):
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                called_name = child.func.id
                if called_name != name:
                    visit(called_name)
        ordered.append(name)

    visit(root_name)

    helper_names = [name for name in ordered if name != root_name]
    helper_sources = [_get_node_source(source_lines, functions[name]) for name in helper_names]
    root_source = _get_node_source(source_lines, functions[root_name])
    return imports + helper_sources + [root_source]


def _annotation_to_argparse(annotation) -> str:
    if annotation in (int, float, str):
        return annotation.__name__
    return 'str'


def _build_step_script(step_definition) -> str:
    source_file = inspect.getsourcefile(step_definition.func)
    if source_file is None:
        raise ValueError(f'Could not determine source file for step {step_definition.id}')

    with open(source_file, 'r', encoding='utf-8') as handle:
        module_source = handle.read()
    source_lines = module_source.splitlines(keepends=True)
    tree = ast.parse(module_source)
    extracted_sources = _collect_helper_functions(tree, source_lines, step_definition.func.__name__)
    function_source = '\n'.join(textwrap.dedent(block).rstrip() for block in extracted_sources if block.strip())

    parser_lines = []
    call_args = []
    for param in step_definition.signature.parameters.values():
        if param.kind not in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
            continue
        annotation = _annotation_to_argparse(param.annotation)
        required = param.default is inspect._empty
        default_line = '' if required else f", default={param.default!r}"
        parser_lines.append(
            f"    parser.add_argument('--{param.name}', type={annotation}, required={str(required)}{default_line}, help='Auto-generated parameter')"
        )
        call_args.append(f"{param.name}=args.{param.name}")

    parser_block = '\n'.join(parser_lines) if parser_lines else '    pass'
    call_block = ', '.join(call_args)

    return (
        "import argparse\n"
        "import json\n\n"
        f"{function_source}\n\n"
        "def _serialize_output(value):\n"
        "    if value is None:\n"
        "        return None\n"
        "    if isinstance(value, (dict, list)):\n"
        "        return json.dumps(value)\n"
        "    return str(value)\n\n"
        "def main():\n"
        f"    parser = argparse.ArgumentParser(description='{step_definition.id} processing step.')\n"
        f"{parser_block}\n\n"
        "    args = parser.parse_args()\n"
        f"    result = {step_definition.func.__name__}({call_block})\n"
        "    serialized = _serialize_output(result)\n"
        "    if serialized is not None:\n"
        "        print(serialized)\n\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    )


def generate_step_directory(step_definition, output_dir: str, user_id: str, app_id: str) -> str:
    step_dir = os.path.join(output_dir, step_definition.id)
    version_dir = os.path.join(step_dir, '1')
    os.makedirs(version_dir, exist_ok=True)

    config = {
        'pipeline_step': {
            'id': step_definition.id,
            'user_id': user_id,
            'app_id': app_id,
        },
        'pipeline_step_input_params': step_definition.get_input_params(),
        'build_info': {'python_version': step_definition.python_version},
        'pipeline_step_compute_info': step_definition.compute.to_dict(),
    }

    with open(os.path.join(step_dir, 'config.yaml'), 'w', encoding='utf-8') as handle:
        yaml.safe_dump(config, handle, default_flow_style=False, sort_keys=False)

    with open(os.path.join(step_dir, 'requirements.txt'), 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(_ensure_list_requirements(step_definition.requirements)) + '\n')

    with open(os.path.join(version_dir, 'pipeline_step.py'), 'w', encoding='utf-8') as handle:
        handle.write(_build_step_script(step_definition))

    return step_dir