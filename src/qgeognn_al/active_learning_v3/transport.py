"""Stateless bounded relay. Infrastructure failure stops; no retry/revision/recovery."""
from __future__ import annotations
import copy
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import tomllib
from ..active_learning_v2.scientist_transport import audit_cli
from .artifacts import Audit, digest, file_hash, once
from .schema import *
from .selector import SYSTEM_PROMPT, validate_selection


class TransportFailure(RuntimeError):
    def __init__(self, category, *, exit_code=None, cli_version=None):
        super().__init__(category)
        self.category, self.exit_code, self.cli_version = category, exit_code, cli_version


class ContextBudgetError(ValueError):
    def __init__(self, audit):
        self.audit = audit
        super().__init__('context hard limit exceeded: ' + dumps(audit))


def settings(backend='codex_cli', model='gpt-6-sol', base_url='https://token4research.cn', effort='high'):
    from urllib.parse import urlsplit
    url = urlsplit(base_url)
    if backend not in ('codex_cli', 'responses') or url.scheme != 'https' or not url.hostname or url.username or url.password or url.query or url.fragment:
        raise ValueError('invalid transport configuration')
    if not re.fullmatch(r'[A-Za-z0-9._-]+', model) or effort not in ('low', 'medium', 'high', 'xhigh'):
        raise ValueError('invalid model/effort')
    result = {'backend': backend, 'model': model, 'base_url': base_url.rstrip('/'),
        'reasoning_effort': effort, 'key_env': 'SCIENTIST_API_KEY', 'automatic_retry': False,
        'automatic_fallback': False, 'store': False, 'context_hard_chars': CONTEXT_HARD_CHARS}
    if backend == 'codex_cli':
        executable = shutil.which('codex')
        if not executable:
            raise TransportFailure('cli_unavailable')
        version = subprocess.run([executable, '--version'], capture_output=True, text=True, timeout=10)
        if version.returncode or not re.fullmatch(r'codex(?:-cli)? [A-Za-z0-9.+_-]+\s*', version.stdout):
            raise TransportFailure('cli_version_unavailable', exit_code=version.returncode)
        result['cli'] = {'executable': str(Path(executable).resolve()), 'sha256': file_hash(executable),
                         'version': version.stdout.strip(), 'native_tools': 'disabled_and_audited',
                         'home': 'fresh temporary configuration and copied authentication only'}
    return result


def build_request(packet, working_state, latest, *, hard_limit=CONTEXT_HARD_CHARS):
    context = copy.deepcopy(packet)
    memory = context['scientific_memory']
    ledger, replay = memory.pop('hypothesis_ledger'), memory.pop('balanced_replay_bank')
    sizes = {'system_prompt': len(SYSTEM_PROMPT), 'round_context': len(dumps(context)),
             'hypothesis_ledger': len(dumps(ledger)), 'replay_bank': len(dumps(replay)),
             'working_state': len(dumps(working_state)), 'latest_query_result': len(dumps(latest))}
    messages = [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': dumps({
        'compact_round_context': packet, 'working_state': working_state, 'latest_query_results': latest})}]
    count = len(dumps(messages))
    audit = {'input_char_count': count, 'input_tokens': None, 'token_estimate_chars_div_4': (count+3)//4,
             'token_estimate_is_not_provider_usage': True, 'component_chars': sizes,
             'largest_component': max(sizes, key=sizes.get), 'hard_limit_chars': hard_limit}
    if count > hard_limit:
        raise ContextBudgetError(audit)
    return messages, audit


def cli_call(messages, config):
    cli = config['cli']
    if file_hash(cli['executable']) != cli['sha256']:
        raise TransportFailure('cli_binary_drift', cli_version=cli['version'])
    # Read only endpoint/auth configuration. Never import history, project instructions or MCP config.
    home = Path.home() / '.codex'
    personal = tomllib.loads((home / 'config.toml').read_text())
    provider_id = personal.get('model_provider', 'openai')
    provider = personal.get('model_providers', {}).get(provider_id, {})
    if provider.get('base_url', 'https://api.openai.com/v1').rstrip('/') != config['base_url'] or provider.get('wire_api', 'responses') != 'responses':
        raise TransportFailure('provider_configuration_drift', cli_version=cli['version'])
    with tempfile.TemporaryDirectory(prefix='scientist-v3-') as tmp:
        root = Path(tmp)
        work, isolated = root/'work', root/'home'
        work.mkdir(); isolated.mkdir(mode=0o700)
        if (home/'auth.json').exists():
            shutil.copyfile(home/'auth.json', isolated/'auth.json')
            (isolated/'auth.json').chmod(0o600)
        fields = {'name': 'Frozen Scientist V3 provider', 'base_url': config['base_url'], 'wire_api': 'responses'}
        for name in ('env_key', 'requires_openai_auth'):
            if name in provider:
                fields[name] = provider[name]
        config_text = 'model_provider = "scientist_v3"\n[model_providers.scientist_v3]\n'
        config_text += ''.join(f'{k} = {json.dumps(v)}\n' for k, v in fields.items())
        (isolated/'config.toml').write_text(config_text)
        instruction = root/'instructions.txt'
        instruction.write_text(SYSTEM_PROMPT)
        overrides = {'model_reasoning_effort': config['reasoning_effort'], 'approval_policy': 'never',
            'web_search': 'disabled', 'features.shell_tool': False, 'features.multi_agent': False,
            'features.goals': False, 'features.memories': False, 'features.code_mode_host': False,
            'tools.experimental_request_user_input.enabled': False, 'mcp_servers': {},
            'model_instructions_file': str(instruction)}
        command = [cli['executable'], 'exec', '--ignore-rules', '--skip-git-repo-check', '--sandbox',
                   'read-only', '--json', '--color', 'never', '--model', config['model'], '--cd', str(work)]
        for k, v in overrides.items():
            command += ['-c', f'{k}={json.dumps(v)}']
        command += ['-']
        env = {k: os.environ[k] for k in ('PATH', 'SYSTEMROOT', 'LANG', 'TMPDIR') if k in os.environ}
        env.update(HOME=str(root), CODEX_HOME=str(isolated))
        if provider.get('env_key') and provider['env_key'] in os.environ:
            env[provider['env_key']] = os.environ[provider['env_key']]
        try:
            result = subprocess.run(command, input=dumps(messages), text=True, capture_output=True,
                                    timeout=600, cwd=work, env=env)
        except subprocess.TimeoutExpired as error:
            raise TransportFailure('cli_timeout', cli_version=cli['version']) from error
        if result.returncode:
            raise TransportFailure('cli_exit', exit_code=result.returncode, cli_version=cli['version'])
        try:
            events = [json.loads(s) for s in result.stdout.splitlines() if s.strip()]
            files = list((isolated/'sessions').rglob('*.jsonl'))
            if len(files) != 1:
                raise RuntimeError('missing rollout')
            rows = [json.loads(s) for s in files[0].read_text().splitlines() if s.strip()]
            answer, audit = audit_cli(events, rows, config['model'], config['reasoning_effort'])
        except (ValueError, RuntimeError, KeyError, TypeError) as error:
            raise TransportFailure('cli_provenance_or_native_tool_failure', cli_version=cli['version']) from error
        usage = [e.get('usage', {}) for e in events if e.get('type') in ('turn.completed', 'task_complete')]
        return answer, {'audit': audit, 'cli_version': cli['version'], 'usage': usage,
                        'model': config['model'], 'effort': config['reasoning_effort']}


def responses_call(messages, config):
    from openai import OpenAI
    if not os.environ.get(config['key_env']):
        raise TransportFailure('missing_api_key')
    client = OpenAI(api_key=os.environ[config['key_env']], base_url=config['base_url'], max_retries=0, timeout=600)
    response = client.responses.create(model=config['model'], input=messages, tools=[], store=False,
                                       reasoning={'effort': config['reasoning_effort']})
    if response.model != config['model'] and not response.model.startswith(config['model']+'-'):
        raise TransportFailure('model_provenance_mismatch')
    if any(item.type not in ('message', 'reasoning') for item in response.output):
        raise TransportFailure('native_tool_or_unknown_output')
    return response.output_text, {'response_id': response.id, 'served_model': response.model,
        'usage': response.usage.model_dump(mode='json') if response.usage else {}, 'native_tool_calls': 0}


def failure_receipt(error, request_hash, config):
    # Deliberately no str(error), stderr, provider body, headers, or environment values.
    category = error.category if isinstance(error, TransportFailure) else (
        'cli_exit' if isinstance(error, subprocess.CalledProcessError) else
        'transport_timeout' if isinstance(error, (TimeoutError, subprocess.TimeoutExpired)) else 'runtime_or_provider_failure')
    return {'request_sha256': request_hash, 'model': config['model'], 'effort': config['reasoning_effort'],
        'cli_version': getattr(error, 'cli_version', None) or config.get('cli', {}).get('version'),
        'exit_code': getattr(error, 'exit_code', None) or getattr(error, 'returncode', None),
        'error_category': category, 'diagnostic': 'Operation stopped. Raw provider/subprocess diagnostics withheld.',
        'automatic_retry': False, 'protocol_changed': False}


def usage_counts(provenance):
    usage = provenance.get('usage') or {}
    if isinstance(usage, list):
        usage = usage[-1] if usage else {}
    return {'input_tokens': usage.get('input_tokens'), 'output_tokens': usage.get('output_tokens'),
            'reasoning_tokens': usage.get('reasoning_output_tokens',
                (usage.get('output_tokens_details') or {}).get('reasoning_tokens'))}


def run_selector(packet, catalog, ledger, directory, config, call=None, guard=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory/'selector.started.json').exists():
        raise RuntimeError('selector already started; no automatic retry or trajectory repair')
    once(directory/'selector.started.json', {'packet_sha256': packet['packet_sha256'], 'config_sha256': digest(config)})
    audit = Audit(directory, packet)
    state, latest = empty_working_state(), []
    call = call or (cli_call if config['backend'] == 'codex_cli' else responses_call)
    for turn in range(MODEL_CALL_BUDGET):
        if guard:
            guard()
        try:
            messages, sizes = build_request(packet, state, latest, hard_limit=config['context_hard_chars'])
        except ContextBudgetError as error:
            once(directory/f'context_failure_{turn:02d}.json', error.audit)
            audit.append('context_budget_failure', error.audit, turn=turn, request_sha256=None)
            raise
        request_hash = digest({'messages': messages, 'config': config})
        sizes.update(query_count=len(catalog.queries), distinct_candidate_views=len(catalog.viewed))
        audit.append('request', {'messages': messages, 'config': config, 'context_budget': sizes},
                     turn=turn, request_sha256=request_hash)
        try:
            answer, provenance = call(messages, config)
        except Exception as error:
            receipt = failure_receipt(error, request_hash, config)
            once(directory/f'failure_{turn:02d}.json', receipt)
            audit.append('transport_failure', receipt, turn=turn, request_sha256=request_hash)
            raise RuntimeError('STOP: transport failure; see sanitized failure receipt') from None
        receipt = {'answer': answer, 'provenance': provenance, 'request_sha256': request_hash,
                   'context_budget': {**sizes, **usage_counts(provenance)}}
        once(directory/f'turn_{turn:02d}.json', receipt)
        audit.append('response', receipt, turn=turn, request_sha256=request_hash)
        if guard:
            guard()
        try:
            value = json.loads(answer)
            if not isinstance(value, dict):
                raise ValueError('expected JSON object')
            if value.get('type') == 'selection':
                result, merged = validate_selection(value, catalog, packet, ledger)
                once(directory/'selection.json', result)
                once(directory/'ledger_after_selection.json', merged)
                once(directory/f'context_audit_{turn:02d}.json', {**receipt['context_budget'],
                     'query_count_after': len(catalog.queries), 'distinct_candidate_views_after': len(catalog.viewed)})
                audit.append('selection', {'selection': result, 'ledger_sha256': digest(merged)},
                             turn=turn, request_sha256=request_hash)
                return result, merged
            exact(value, ('type', 'queries', 'working_state'), 'query answer')
            if value['type'] != 'query':
                raise ValueError('expected query or selection')
            queries = value['queries']
            items(queries, QUERIES_PER_TURN, 'queries')
            if not queries:
                raise ValueError('empty queries')
            state = validate_working_state(value['working_state'], catalog, ledger['hypotheses'])
            if len(catalog.queries)+len(queries) > catalog.query_budget:
                raise RuntimeError('query budget exhausted')
            audit.append('working_state', state, turn=turn, request_sha256=request_hash)
            latest = []
            for query in queries:
                try:
                    # Route even malformed operation/args through the counted catalog boundary.
                    if not isinstance(query, dict) or set(query) != {'operation', 'args'}:
                        response = catalog.query('__invalid_query__', {})
                    else:
                        response = catalog.query(query['operation'], query['args'])
                except ValueError as error:
                    response = {'error': 'query_validation', 'message': str(error),
                                'remaining_queries': catalog.query_budget-len(catalog.queries)}
                latest.append({'query': query, 'result': response})
                audit.append('query', latest[-1], turn=turn, request_sha256=request_hash)
        except (ValueError, TypeError, KeyError) as error:
            latest = [{'error': 'response_validation', 'message': str(error)}]
            audit.append('response_validation_error', latest, turn=turn, request_sha256=request_hash)
        once(directory/f'context_audit_{turn:02d}.json', {**receipt['context_budget'],
             'query_count_after': len(catalog.queries), 'distinct_candidate_views_after': len(catalog.viewed)})
    audit.append('model_call_budget_exhausted', {}, turn=MODEL_CALL_BUDGET, request_sha256=None)
    raise RuntimeError('model call budget exhausted; no selection accepted')
