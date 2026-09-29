"""Stateless relay with bounded, audited retries for transient Responses failures."""
from __future__ import annotations
import copy
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
import tomllib
from ..active_learning_v2.scientist_transport import audit_cli
from .artifacts import Audit, digest, file_hash, once
from .schema import *
from .selector import SYSTEM_PROMPT, validate_selection
from .catalog import batch_diagnostic
from .response_contract import RESPONSE_SCHEMA, decode_json, decode_wire


class TransportFailure(RuntimeError):
    def __init__(self, category, *, exit_code=None, cli_version=None):
        super().__init__(category)
        self.category, self.exit_code, self.cli_version = category, exit_code, cli_version


class ContextBudgetError(ValueError):
    def __init__(self, audit):
        self.audit = audit
        super().__init__('context hard limit exceeded: ' + dumps(audit))


def settings(backend='responses', model='gpt-6-sol', base_url='https://token4research.cn', effort='high',
             transport_retries=None):
    from urllib.parse import urlsplit
    url = urlsplit(base_url)
    if backend not in ('codex_cli', 'responses') or url.scheme != 'https' or not url.hostname or url.username or url.password or url.query or url.fragment:
        raise ValueError('invalid transport configuration')
    if not re.fullmatch(r'[A-Za-z0-9._-]+', model) or effort not in ('low', 'medium', 'high', 'xhigh'):
        raise ValueError('invalid model/effort')
    if transport_retries is None:
        transport_retries = 3 if backend == 'responses' else 0
    if type(transport_retries) is not int or not 0 <= transport_retries <= 3:
        raise ValueError('transport_retries must be an integer from 0 to 3')
    if backend != 'responses' and transport_retries:
        raise ValueError('transport retries are supported only for Responses')
    result = {'backend': backend, 'model': model, 'base_url': base_url.rstrip('/'),
        'reasoning_effort': effort, 'key_env': 'SCIENTIST_API_KEY', 'automatic_retry': bool(transport_retries),
        'retry_policy': {'max_retries': transport_retries, 'backoff_seconds': [2, 4, 8][:transport_retries],
                         'scope': 'transient_responses_transport_only', 'sdk_max_retries': 0,
                         'max_transport_calls_per_selection': MODEL_CALL_BUDGET*(transport_retries+1)},
        'automatic_fallback': False, 'response_format': 'json_schema', 'response_schema': RESPONSE_SCHEMA, 'store': False, 'context_hard_chars': CONTEXT_HARD_CHARS}
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
                                       reasoning={'effort': config['reasoning_effort']},
                                       text={'format': {'type': 'json_schema', 'name': 'scientist_response', 'strict': True, 'schema': config['response_schema']}})
    if response.model != config['model'] and not response.model.startswith(config['model']+'-'):
        raise TransportFailure('model_provenance_mismatch')
    if any(item.type not in ('message', 'reasoning') for item in response.output):
        raise TransportFailure('native_tool_or_unknown_output')
    return response.output_text, {'response_id': response.id, 'served_model': response.model,
        'usage': response.usage.model_dump(mode='json') if response.usage else {}, 'native_tool_calls': 0}


def transport_error(error):
    """Return allowlisted diagnostics only; never persist error text or provider bodies."""
    from openai import APIConnectionError, APITimeoutError, APIStatusError
    status = None
    retryable = False
    if isinstance(error, TransportFailure):
        category = error.category
    elif isinstance(error, (APITimeoutError, TimeoutError, subprocess.TimeoutExpired)):
        category, retryable = 'transport_timeout', True
    elif isinstance(error, (APIConnectionError, ConnectionError)):
        category, retryable = 'transport_connection', True
    elif isinstance(error, APIStatusError):
        status = error.status_code
        quota_exhausted = getattr(error, 'code', None) in ('insufficient_quota', 'billing_hard_limit_reached')
        category = 'provider_quota_exhausted' if quota_exhausted else 'provider_http_error'
        retryable = not quota_exhausted and (status in (408, 409, 429) or 500 <= status <= 599)
    elif isinstance(error, subprocess.CalledProcessError):
        category = 'cli_exit'
    else:
        category = 'runtime_or_provider_failure'
    return {'error_category': category, 'http_status': status, 'retryable': retryable}


def failure_receipt(error, request_hash, config, *, attempts=1):
    # Deliberately no str(error), stderr, provider body, headers, or environment values.
    return {'request_sha256': request_hash, 'model': config['model'], 'effort': config['reasoning_effort'],
        'cli_version': getattr(error, 'cli_version', None) or config.get('cli', {}).get('version'),
        'exit_code': getattr(error, 'exit_code', None) or getattr(error, 'returncode', None),
        **transport_error(error), 'diagnostic': 'Operation stopped. Raw provider/subprocess diagnostics withheld.',
        'attempts': attempts, 'retries_performed': attempts-1,
        'automatic_retry': attempts > 1, 'protocol_changed': False}


def call_with_retries(call, messages, config, directory, audit, turn, request_hash, guard=None):
    policy = config.get('retry_policy', {'max_retries': 0, 'backoff_seconds': []})
    max_retries = policy['max_retries'] if config['backend'] == 'responses' else 0
    for attempt in range(1, max_retries+2):
        if guard:
            guard()
        audit.append('transport_attempt', {'attempt': attempt, 'max_attempts': max_retries+1},
                     turn=turn, request_sha256=request_hash)
        try:
            answer, provenance = call(messages, config)
        except Exception as error:
            detail = transport_error(error)
            retry = detail['retryable'] and attempt <= max_retries
            delay = policy['backoff_seconds'][attempt-1] if retry else 0
            receipt = {**failure_receipt(error, request_hash, config, attempts=attempt),
                       'will_retry': retry, 'retry_delay_seconds': delay}
            if retry:
                receipt['diagnostic'] = 'Transient transport failure; identical request will be retried.'
            once(directory/f'transport_attempt_{turn:02d}_{attempt:02d}.json', receipt)
            audit.append('transport_attempt_failed', receipt, turn=turn, request_sha256=request_hash)
            if not retry:
                once(directory/f'failure_{turn:02d}.json', receipt)
                audit.append('transport_failure', receipt, turn=turn, request_sha256=request_hash)
                raise RuntimeError('STOP: transport failure; see sanitized failure receipt') from None
            time.sleep(delay)
        else:
            audit.append('transport_attempt_succeeded', {'attempt': attempt},
                         turn=turn, request_sha256=request_hash)
            return answer, {**provenance, 'transport_attempts': attempt, 'transport_retries': attempt-1}


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
    repairs, finalize_calls = 0, 0
    call = call or (cli_call if config['backend'] == 'codex_cli' else responses_call)
    for turn in range(MODEL_CALL_BUDGET):
        if guard:
            guard()
        mode = 'FINALIZE_ONLY' if len(catalog.queries) >= catalog.query_budget or turn >= MODEL_CALL_BUDGET-2 else 'SEARCHING'
        current_packet = copy.deepcopy(packet)
        current_packet['interaction_state'] = {'mode': mode, 'accepted_queries': len(catalog.queries),
            'remaining_queries': max(0, catalog.query_budget-len(catalog.queries)), 'validation_repairs_used': repairs}
        if mode == 'FINALIZE_ONLY':
            finalize_calls += 1
            current_packet['finalization_instruction'] = 'No more queries are available. Use current working_state and viewed candidates. Return selection only.'
            current_packet['viewed_candidate_inventory'] = [{'id': i, 'smiles': catalog.candidates[i]['smiles']} for i in sorted(catalog.viewed)]
            audit.append('finalization_mode', current_packet['interaction_state'], turn=turn, request_sha256=None)
        try:
            messages, sizes = build_request(current_packet, state, latest, hard_limit=config['context_hard_chars'])
        except ContextBudgetError as error:
            once(directory/f'context_failure_{turn:02d}.json', error.audit)
            audit.append('context_budget_failure', error.audit, turn=turn, request_sha256=None)
            raise
        request_hash = digest({'messages': messages, 'config': config})
        sizes.update(query_count=len(catalog.queries), distinct_candidate_views=len(catalog.viewed))
        audit.append('request', {'messages': messages, 'config': config, 'context_budget': sizes},
                     turn=turn, request_sha256=request_hash)
        answer, provenance = call_with_retries(call, messages, config, directory, audit,
                                               turn, request_hash, guard)
        receipt = {'answer': answer, 'provenance': provenance, 'request_sha256': request_hash,
                   'context_budget': {**sizes, **usage_counts(provenance)}}
        once(directory/f'turn_{turn:02d}.json', receipt)
        audit.append('response', receipt, turn=turn, request_sha256=request_hash)
        if guard:
            guard()
        try:
            value, recovered = decode_json(answer)
            if recovered:
                audit.append('json_recovery', {'noncanonical_json_recovered': True}, turn=turn, request_sha256=request_hash)
            if set(value) == {'response'}:
                try:
                    value = decode_wire(value)
                except Exception as error:
                    raise ValueError('structured response schema mismatch') from error
            if value.get('type') == 'selection':
                result, merged = validate_selection(value, catalog, packet, ledger)
                once(directory/'batch_diagnostic.json', batch_diagnostic(result['choices'], catalog))
                once(directory/'interaction_summary.json', {'total_llm_calls': turn+1, 'accepted_scientific_queries': len(catalog.queries),
                    'invalid_query_attempts': len(catalog.invalid_queries), 'validation_repairs_used': repairs, 'finalization_mode_calls': finalize_calls,
                    'distinct_viewed_candidates': len(catalog.viewed)})
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
            if mode == 'FINALIZE_ONLY':
                raise ValueError('FINALIZE_ONLY: no more queries are available; return selection only')
            audit.append('working_state', state, turn=turn, request_sha256=request_hash)
            latest = []
            for query in queries:
                if len(catalog.queries) >= catalog.query_budget:
                    latest.append({'query': query, 'result': {'status': 'FINALIZE_ONLY', 'remaining_queries': 0, 'message': 'No more queries are available; return selection only'}})
                    audit.append('query_not_executed_budget_exhausted', latest[-1], turn=turn, request_sha256=request_hash)
                    continue
                try:
                    # Route even malformed operation/args through the counted catalog boundary.
                    if not isinstance(query, dict) or set(query) != {'operation', 'args'}:
                        response = catalog.query('__invalid_query__', {})
                    else:
                        response = catalog.query(query['operation'], query['args'])
                except ValueError as error:
                    repairs += 1
                    response = {'error': 'query_validation', 'message': str(error),
                                'remaining_queries': catalog.query_budget-len(catalog.queries)}
                latest.append({'query': query, 'result': response})
                audit.append('query', latest[-1], turn=turn, request_sha256=request_hash)
                if repairs > REPAIR_BUDGET:
                    raise RuntimeError('STOP: query validation repair budget exceeded')
        except (ValueError, TypeError, KeyError) as error:
            repairs += 1
            latest = [{'error': 'response_validation', 'message': str(error)}]
            audit.append('response_validation_error', latest, turn=turn, request_sha256=request_hash)
            if repairs > REPAIR_BUDGET:
                raise RuntimeError('STOP: response validation repair budget exceeded') from error
        once(directory/f'context_audit_{turn:02d}.json', {**receipt['context_budget'],
             'query_count_after': len(catalog.queries), 'distinct_candidate_views_after': len(catalog.viewed)})
    audit.append('model_call_budget_exhausted', {}, turn=MODEL_CALL_BUDGET, request_sha256=None)
    raise RuntimeError('model call budget exhausted; no selection accepted')
