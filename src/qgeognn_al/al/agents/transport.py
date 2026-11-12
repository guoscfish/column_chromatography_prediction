"""Explicit recorded/mock/Responses transport; no ambient credentials or retries.

The Responses request shape is migrated from b1a6ed7a scientist_transport.py.
An unknown dispatch outcome is deliberately not retried automatically. No API is
called by importing, validating or preparing this module.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
import json
import os
from urllib.parse import urlsplit
from ...infra.contracts import content_hash
from ..workflow.state import read_envelope, write_envelope


class AmbiguousRequest(RuntimeError):
    """A dispatched request has no durable answer; reconcile before resuming."""


@dataclass(frozen=True)
class ProviderConfig:
    model: str
    reasoning_effort: str
    base_url: str
    key_env: str = 'SCIENTIST_API_KEY'
    timeout_seconds: int = 600
    max_output_tokens: int | None = None

    def __post_init__(self):
        u = urlsplit(self.base_url)
        if u.scheme != 'https' or not u.hostname or u.username or u.password or u.query or u.fragment:
            raise ValueError('provider requires an explicit HTTPS URL without credentials/query')
        if not self.model or self.reasoning_effort not in ('low','medium','high','xhigh','max'):
            raise ValueError('explicit historical model/effort required')
        if not self.key_env.isidentifier() or not 1 <= self.timeout_seconds <= 3600:
            raise ValueError('invalid credential environment name or timeout')
        if self.max_output_tokens is not None and (type(self.max_output_tokens) is not int or self.max_output_tokens<1):
            raise ValueError('invalid explicit token limit')

    @property
    def sha256(self):
        return content_hash(asdict(self))


def request_hash(messages, config):
    if not isinstance(messages, list) or not messages:
        raise ValueError('complete explicit conversation required')
    if any(set(m) != {'role','content'} or m['role'] not in ('system','user','assistant') or not isinstance(m['content'],str) for m in messages):
        raise ValueError('only text conversations without native tools are supported')
    return content_hash({'messages':messages,'config':asdict(config)})


class RecordedTransport:
    kind = 'recorded'
    paid = False

    def __init__(self, recordings):
        self.recordings = dict(recordings)
        self.calls = 0

    def call(self, messages, config):
        key = request_hash(messages, config)
        if key not in self.recordings:
            raise LookupError('BLOCKED_MISSING_FIXTURE: no exact recorded request binding')
        item = self.recordings[key]
        if not isinstance(item.get('answer'), str):
            raise ValueError('recorded answer must preserve raw response text')
        self.calls += 1
        return item['answer'], {'source_kind':'recorded','request_sha256':key,
                                 'original_provenance':item.get('provenance',{}),'paid_request':False}


class ResponsesTransport:
    kind = 'responses'
    paid = True

    def __init__(self, *, allow_provider=False, client_factory=None):
        self.allow_provider = allow_provider
        self.client_factory = client_factory

    def call(self, messages, config):
        # No secret lookup, SDK import, or client creation occurs before this gate.
        if not self.allow_provider:
            raise PermissionError('new paid provider execution requires separate explicit authorization')
        request_hash(messages, config)
        key = os.environ.get(config.key_env, '').strip()
        if not key or any(c.isspace() for c in key):
            raise RuntimeError('explicit provider credential is unavailable')
        factory = self.client_factory
        if factory is None:
            from openai import OpenAI
            factory = OpenAI
        try:
            with factory(api_key=key, base_url=config.base_url, max_retries=0, timeout=config.timeout_seconds) as client:
                with client.responses.stream(model=config.model, input=messages,
                        reasoning={'effort':config.reasoning_effort}, tools=[], store=False,
                        **({} if config.max_output_tokens is None else {'max_output_tokens':config.max_output_tokens})) as stream:
                    response = stream.get_final_response()
            if not isinstance(response.id,str) or not response.id.strip():
                raise ValueError('missing provider response identity')
            if response.status != 'completed':
                raise ValueError('incomplete provider response')
            if response.model != config.model and not response.model.startswith(config.model+'-'):
                raise ValueError('served model drift')
            if any(item.type not in ('message','reasoning') for item in response.output):
                raise ValueError('native tool output rejected')
            if not isinstance(response.output_text,str) or not response.output_text.strip():
                raise ValueError('empty provider response')
        except Exception as error:
            # Never persist SDK error text, stderr, credential or provider body.
            raise RuntimeError('provider call/validation failed: '+type(error).__name__) from None
        return response.output_text, {'source_kind':'online','response_id':response.id,
            'served_model':response.model,'usage':response.usage.model_dump(mode='json') if response.usage else None,
            'native_tool_calls':0,'paid_request':True}


class DurableTransport:
    """One receipt per logical request; successful answers survive state crashes.

    A started-only request, timeout, or crash fails closed on resume. Format
    corrections use new, bounded logical turns with their complete conversation.
    This is stricter than old transport retry behavior; it is not an exact replay
    claim for that old failure mode.
    """
    def __init__(self, transport, directory, *, forbidden_output_roots=(), maximum_calls=3):
        self.transport, self.root = transport, Path(directory).absolute()
        if type(maximum_calls) is not int or not 1 <= maximum_calls <= 16:
            raise ValueError('bounded request quota required')
        if any(p.is_symlink() for p in [self.root,*self.root.parents]):
            raise ValueError('transport output symlink forbidden')
        if any(self.root.resolve().is_relative_to(Path(p).resolve()) for p in forbidden_output_roots):
            raise ValueError('transport output is inside protected worktree')
        self.maximum_calls = maximum_calls

    def call(self, messages, config, *, turn):
        if type(turn) is not int or not 0 <= turn < self.maximum_calls:
            raise ValueError('request quota exhausted')
        key = request_hash(messages,config)
        receipt = self.root/f'turn_{turn:02d}.json'
        marker = self.root/f'turn_{turn:02d}.started.json'
        if receipt.exists():
            payload = read_envelope(receipt)['payload']
            if payload['request_sha256'] != key:
                raise ValueError('request changed after durable response')
            return payload['answer'], payload['provenance']
        if marker.exists():
            if read_envelope(marker)['payload']['request_sha256'] != key:
                raise ValueError('request changed after dispatch marker')
            raise AmbiguousRequest('started request has no answer; no automatic re-dispatch')
        if self.transport.paid and not getattr(self.transport,'allow_provider',False):
            raise PermissionError('paid provider requires separate explicit authorization')
        self.root.mkdir(parents=True,exist_ok=True)
        write_envelope(marker, {'request_sha256':key,'transport':self.transport.kind,
                               'messages':messages,'config':asdict(config)},exclusive=True)
        answer, provenance = self.transport.call(messages,config)
        write_envelope(receipt, {'request_sha256':key,'answer':answer,'provenance':provenance},exclusive=True)
        return answer, provenance
