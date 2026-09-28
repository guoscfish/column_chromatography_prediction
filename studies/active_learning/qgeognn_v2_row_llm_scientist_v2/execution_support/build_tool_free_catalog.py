"""Derive a pinned, experiment-only tool configuration from the vendor catalog."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
BINARY = Path('/private/tmp/scientist-codex-v0.157.1/codex-aarch64-apple-darwin')
raw = subprocess.check_output([str(BINARY), 'debug', 'models', '--bundled'])
catalog = json.loads(raw)
model = copy.deepcopy(next(m for m in catalog['models'] if m['slug'] == 'gpt-6-sol'))
before = {key: model.get(key) for key in ('tool_mode', 'multi_agent_version',
           'node_repl_disabled', 'experimental_supported_tools', 'supports_search_tool', 'apply_patch_tool_type')}
model.update(tool_mode=None, multi_agent_version=None, node_repl_disabled=True,
             experimental_supported_tools=[], supports_search_tool=False, apply_patch_tool_type=None)
record = {'models': [model]}
catalog_path = HERE / 'gpt-6-sol-no-native-tools.json'
if catalog_path.exists():
    raise RuntimeError('catalog already exists; preserve the registered configuration')
catalog_path.write_text(json.dumps(record, indent=2) + '\n')
archive = HERE / 'revisions/desktop-v1'
archive.mkdir(parents=True, exist_ok=False)
shutil.copy2(HERE / 'codex', archive / 'codex')
manifest = {'source_cli_version': subprocess.check_output([str(BINARY), '--version'], text=True).strip(),
            'source_binary_sha256': hashlib.sha256(BINARY.read_bytes()).hexdigest(),
            'source_catalog_sha256': hashlib.sha256(raw).hexdigest(),
            'catalog_sha256': hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
            'previous_tool_fields': before,
            'configured_tool_fields': {key: model[key] for key in before},
            'model_identity_unchanged': True}
(HERE / 'tool_free_catalog_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(manifest, indent=2))
