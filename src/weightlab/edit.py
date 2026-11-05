"""Verified, separate-file F16 GGUF rollback baseline. Never edit originals."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil

import numpy as np

from .evidence import digest, file_sha256, git_state, utc_now, write_json
from .precision import floating_tensor_values, round_bf16_values


def blend_tensor(original, parent, alpha):
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("Rollback alpha must be finite and between 0 and 1")
    if original.shape != parent.shape or original.dtype != parent.dtype:
        raise ValueError("Tensor shape and dtype must match")
    if original.dtype not in (np.dtype('float16'), np.dtype('float32')):
        raise ValueError("This operation accepts only floating tensors")
    if not np.isfinite(original).all() or not np.isfinite(parent).all():
        raise ValueError("Non-finite source tensor")
    if alpha == 0:
        return original.copy()
    if alpha == 1:
        return parent.copy()
    result = (original.astype(np.float32) * (1 - alpha) + parent.astype(np.float32) * alpha).astype(original.dtype)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite edited tensor after cast")
    return result


def tensor_hash(tensor):
    return hashlib.sha256(memoryview(np.ascontiguousarray(tensor)).cast('B')).hexdigest()


def create_rollback(config, base_key, parent_key, output, manifest_path, alpha, layers=None,
                    pattern=r'blk\.(\d+)\.ffn_down\.weight', parent_round_bf16=False):
    from gguf import GGUFReader
    output = Path(output)
    manifest_path = Path(manifest_path)
    root = Path.cwd().resolve()
    if not any(output.resolve().is_relative_to(root / d) for d in ('artifacts', 'local')):
        raise ValueError("Edited binaries must go under ignored artifacts/ or local/")
    if output.exists() or manifest_path.exists():
        raise FileExistsError("Output and manifest must be new")
    state = git_state()
    if any(line[3:].startswith(('src/', 'configs/', 'tests/')) for line in state['status'].splitlines()):
        raise RuntimeError("Commit source and configuration before producing an edit")
    base_spec, parent_spec = config['models'][base_key], config['models'][parent_key]
    base_path, parent_path = Path(base_spec['path']).expanduser(), Path(parent_spec['path']).expanduser()
    for spec, path in [(base_spec, base_path), (parent_spec, parent_path)]:
        if file_sha256(path) != spec['sha256']:
            raise ValueError(f"Source checksum mismatch: {path}")
    original, parent = GGUFReader(base_path), GGUFReader(parent_path)
    for key in ['general.architecture', 'tokenizer.ggml.tokens', 'tokenizer.ggml.bos_token_id', 'tokenizer.ggml.eos_token_id']:
        a, b = original.get_field(key), parent.get_field(key)
        if (a is None) != (b is None) or (a is not None and a.contents() != b.contents()):
            raise ValueError(f"Source metadata mismatch: {key}")
    source_tensors = {t.name: t for t in original.tensors}
    parent_tensors = {t.name: t for t in parent.tensors}
    if source_tensors.keys() != parent_tensors.keys():
        raise ValueError("Source tensor sets differ")
    selected = []
    for name, t in source_tensors.items():
        match = re.fullmatch(pattern, name)
        layer_match = re.match(r'blk\.(\d+)\.', name)
        if match and (layers is None or (layer_match and int(layer_match[1]) in layers)):
            p = parent_tensors[name]
            if not np.array_equal(t.shape, p.shape) or t.data.dtype.kind != 'f':
                raise ValueError(f"Selected tensor must be aligned floating values: {name}")
            if p.data.dtype.kind != 'f' and getattr(p.tensor_type, 'name', None) != 'BF16':
                raise ValueError('Only floating parent encodings are supported')
            selected.append(name)
    if not selected:
        raise ValueError("No matching floating tensors")
    if layers is not None and {int(n.split('.')[1]) for n in selected} != set(layers):
        raise ValueError("Some requested layers were not found")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + '.partial')
    if partial.exists():
        raise FileExistsError(partial)
    report = {'schema_version': 1, 'started_utc': utc_now(), 'git': state,
              'operation': 'partial_parent_rollback_of_selected_floating_tensors',
              'status': 'started', 'base': base_spec, 'parent': parent_spec,
              'alpha': alpha, 'selected': selected, 'output': str(output),
              'selection_pattern': pattern,
              'parent_transformation': 'round_bf16_values_then_cast_back' if parent_round_bf16 else 'none',
              'notes': ['No gradients or training. Not a sparse/rank-one edit.',
                        'F16 casting is part of the evaluated operation.'], 'tensors': []}
    write_json(manifest_path, report)
    try:
        shutil.copyfile(base_path, partial)
        edited = GGUFReader(partial, mode='r+')
        edited_tensors = {t.name: t for t in edited.tensors}
        for name in selected:
            o = source_tensors[name].data
            raw_parent = parent_tensors[name].data
            decoded_parent = floating_tensor_values(parent_tensors[name], dtype=o.dtype)
            b = round_bf16_values(decoded_parent) if parent_round_bf16 else decoded_parent
            new = blend_tensor(o, b, alpha)
            delta = new.astype(np.float32) - o.astype(np.float32)
            norm = float(np.sqrt(np.square(o.astype(np.float32)).sum(dtype=np.float64)))
            delta_norm = float(np.sqrt(np.square(delta).sum(dtype=np.float64)))
            changed = int(np.count_nonzero(new != o))
            edited_tensors[name].data[:] = new
            report['tensors'].append({'name': name, 'shape': list(o.shape), 'dtype': str(o.dtype),
                                      'base_sha256': tensor_hash(o), 'parent_sha256': tensor_hash(raw_parent),
                                      'parent_encoding': str(parent_tensors[name].tensor_type),
                                      'effective_parent_sha256': tensor_hash(b),
                                      'edited_sha256': tensor_hash(new), 'changed_values': changed,
                                      'relative_frobenius_change': delta_norm / norm if norm else None})
            write_json(manifest_path, report)
            print(f'{name}: {changed:,} changed values, relative norm {delta_norm / norm:.6g}', flush=True)
        edited.data.flush()
        for name, tensor in edited_tensors.items():
            if name not in selected and tensor_hash(tensor.data) != tensor_hash(source_tensors[name].data):
                raise ValueError(f"Unexpected change outside selected tensors: {name}")
        for row in report['tensors']:
            if tensor_hash(edited_tensors[row['name']].data) != row['edited_sha256']:
                raise ValueError('Edited tensor verification failed')
        report['untouched_tensors_verified'] = len(edited_tensors) - len(selected)
        report['changed_values'] = sum(t['changed_values'] for t in report['tensors'])
        report['edited_file_sha256'] = file_sha256(partial)
        if alpha == 0 and report['edited_file_sha256'] != base_spec['sha256']:
            raise ValueError("Identity rollback changed file bytes")
        if file_sha256(base_path) != base_spec['sha256'] or file_sha256(parent_path) != parent_spec['sha256']:
            raise ValueError("An original source file changed")
        partial.replace(output)
        report.update(status='verified', ended_utc=utc_now(), bytes=output.stat().st_size)
        write_json(manifest_path, report)
    except BaseException as exc:
        report.update(status='failed', ended_utc=utc_now(), error=repr(exc))
        write_json(manifest_path, report)
        raise
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/local_models.json')
    p.add_argument('--base', default='fable_f16')
    p.add_argument('--parent', default='parent_f16')
    p.add_argument('--alpha', type=float, required=True)
    p.add_argument('--layers', help='Comma-separated layer indices; default all FFN output matrices')
    p.add_argument('--pattern', default=r'blk\.(\d+)\.ffn_down\.weight', help='Full-match tensor selector')
    p.add_argument('--parent-round-bf16', action='store_true', help='Explicit precision-alignment/control transform')
    p.add_argument('--output', required=True)
    p.add_argument('--manifest', required=True)
    a = p.parse_args()
    config = json.loads(Path(a.config).read_text())
    layers = None if a.layers is None else [int(x) for x in a.layers.split(',')]
    report = create_rollback(config, a.base, a.parent, a.output, a.manifest, a.alpha, layers, a.pattern, a.parent_round_bf16)
    print(json.dumps({k:report[k] for k in ['status','changed_values','edited_file_sha256','output']}, indent=2))


if __name__ == '__main__':
    main()
