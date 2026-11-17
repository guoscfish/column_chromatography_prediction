"""Offline framing and safe-handover checks; no provider calls or source labels."""
from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from contextlib import ExitStack

from scripts.studies import repair_nocw_sse_frames_20261116 as repair


def events(lines):
    data, result = [], []
    for line in lines:
        if line.startswith("data:"):
            data.append(line[5:].lstrip(" "))
        elif line == "" and data:
            result.append(json.loads("\n".join(data)))
            data = []
    if data:
        result.append(json.loads("\n".join(data)))
    return result


class FramingTests(unittest.TestCase):
    def run_lines(self, raw):
        captured, stats = [], {"inserted_blank_separators": 0}
        framed = list(repair.reframe_lines(iter(raw), captured.append, stats))
        self.assertEqual(captured, raw)
        return framed, stats

    def test_normal_events_are_byte_for_byte_unchanged(self):
        raw = ['data: {"type":"response.in_progress"}', '', 'data: {"type":"response.completed"}', '']
        framed, stats = self.run_lines(raw)
        self.assertEqual(framed, raw)
        self.assertEqual(stats["inserted_blank_separators"], 0)

    def test_missing_separator_between_two_full_events(self):
        raw = ['data: {"type":"response.output_text.delta","delta":"x"}',
               'data: {"type":"response.completed","response":{"status":"completed"}}', '']
        with self.assertRaises(json.JSONDecodeError):
            events(raw)
        framed, stats = self.run_lines(raw)
        self.assertEqual([x["type"] for x in events(framed)], ['response.output_text.delta', 'response.completed'])
        self.assertEqual(stats["inserted_blank_separators"], 1)

    def test_provider_error_remains_an_error_event(self):
        raw = ['data: {"type":"response.reasoning_summary_text.done"}',
               'data: {"type":"error","code":"upstream_stream_read_error"}', '']
        framed, stats = self.run_lines(raw)
        self.assertEqual(events(framed)[-1], {"type": "error", "code": "upstream_stream_read_error"})
        self.assertEqual(stats["inserted_blank_separators"], 1)

    def test_multiline_json_stays_multiline(self):
        raw = ['data: {"type":', 'data: "response.in_progress"}', '']
        framed, stats = self.run_lines(raw)
        self.assertEqual(framed, raw)
        self.assertEqual(events(framed), [{"type": "response.in_progress"}])
        self.assertEqual(stats["inserted_blank_separators"], 0)

    def test_complete_event_before_pretty_printed_event(self):
        raw = ['data: {"type":"response.in_progress"}', 'data: {',
               'data: "type":"response.completed"', 'data: }', '']
        framed, stats = self.run_lines(raw)
        self.assertEqual(len(events(framed)), 2)
        self.assertEqual(stats["inserted_blank_separators"], 1)

    def test_truncated_json_not_salvaged(self):
        raw = ['data: {"type":"response.in_progress"', 'data: {"type":"response.completed"}', '']
        framed, stats = self.run_lines(raw)
        with self.assertRaises(json.JSONDecodeError):
            events(framed)
        self.assertEqual(stats["inserted_blank_separators"], 0)

    def test_arbitrary_json_not_reframed(self):
        raw = ['data: {"answer":"not a typed event"}', 'data: {"type":"response.completed"}', '']
        framed, stats = self.run_lines(raw)
        self.assertEqual(framed, raw)
        self.assertEqual(stats["inserted_blank_separators"], 0)

    def test_comments_and_done_preserved(self):
        raw = [': ping', 'data: {"type":"response.in_progress"}', '', 'data: [DONE]', '']
        framed, stats = self.run_lines(raw)
        self.assertEqual(framed, raw)
        self.assertEqual(stats["inserted_blank_separators"], 0)

    def test_no_response_completed_is_created(self):
        raw = ['data: {"type":"response.in_progress"}', 'data: {"type":"keepalive"}', '']
        framed, _ = self.run_lines(raw)
        self.assertNotIn('response.completed', [x['type'] for x in events(framed)])

    def test_client_factory_is_restored_on_failure(self):
        prior = repair.stream.client_for
        with patch.object(repair, 'ORIGINAL_LIVE', side_effect=RuntimeError('offline failure')):
            with self.assertRaises(RuntimeError):
                repair.live_call([], {}, Path('/tmp'), 0, 0)
        self.assertIs(repair.stream.client_for, prior)

    def test_request_arguments_are_unchanged(self):
        messages, config, directory = [{'role': 'user', 'content': 'offline fixture'}], {'model': 'fixture'}, Path('/tmp')
        with patch.object(repair, 'ORIGINAL_LIVE', return_value=('offline answer', {})) as call:
            repair.live_call(messages, config, directory, 2, 7)
        call.assert_called_once_with(messages, config, directory, 2, 7)

    def test_dead_workers_are_never_adopted(self):
        with patch.object(repair.cap, 'alive', return_value=False), patch.object(repair.cap, 'safe_backoff') as check:
            self.assertFalse(repair.safe({'pid': 12, 'worker_script': Path(repair.history.__file__).name}, {}))
            check.assert_not_called()

    def test_training_or_active_request_cannot_be_adopted(self):
        with patch.object(repair.cap, 'alive', return_value=True), patch.object(repair.cap, 'safe_backoff', return_value=False):
            self.assertFalse(repair.safe({'pid': 12, 'worker_script': Path(repair.history.__file__).name}, {'phase': 'REQUESTING'}))

    def test_unexpected_worker_script_is_not_adopted(self):
        with patch.object(repair.cap, 'alive', return_value=True), patch.object(repair.cap, 'safe_backoff') as check:
            self.assertFalse(repair.safe({'pid': 12, 'worker_script': 'unapproved.py'}, {}))
            check.assert_not_called()


class FrozenTransportIntegrationTests(unittest.TestCase):
    def invoke(self, raw, status=200, close_error=False):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        directory = Path(temporary.name)
        plan = directory / 'plan.json'
        plan.write_text('{}')
        calls = []

        class Response:
            status_code = status
            def iter_lines(self):
                yield from raw
            def raise_for_status(self):
                return None

        class Client:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return None
            @contextmanager
            def stream(self, *args, **kwargs):
                calls.append((args, kwargs))
                yield Response()
                if close_error:
                    raise RuntimeError('offline close error')

        messages = [{'role': 'user', 'content': 'offline input'}]
        config = {'base_url': 'https://offline.invalid/v1', 'model': 'offline-model',
                  'reasoning_effort': 'xhigh', 'max_output_tokens': 32768, 'sampling_parameters': {}}
        with ExitStack() as stack:
            stack.enter_context(patch.object(repair, 'PLAN', plan))
            stack.enter_context(patch.object(repair, 'ORIGINAL_CLIENT', return_value=Client()))
            stack.enter_context(patch.object(repair.stream.s, 'require_authorization'))
            stack.enter_context(patch.object(repair.stream.retry, 'WORKER_SEED', None))
            stack.enter_context(patch('src.qgeognn_al.active_learning_v2.scientist_transport.provider_key', return_value='offline-key'))
            stack.enter_context(patch.object(repair.stream, 'completed_result', return_value=('offline answer', {'served_model': 'offline-model'})))
            outcome = repair.live_call(messages, config, directory, 0, 0)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1]['json'], repair.stream.body_for(messages, config))
        return outcome, directory

    def test_complete_event_in_missing_delimiter_frame_is_saved(self):
        outcome, directory = self.invoke([
            'data: {"type":"response.in_progress"}',
            'data: {"type":"response.completed","response":{"id":"offline-response","status":"completed"}}', ''])
        self.assertEqual(outcome[0], 'offline answer')
        self.assertTrue(json.loads((directory / 'stream_00_00.completed.json').read_text())['validated_completed'])
        receipt = json.loads((directory / 'stream_00_00.framing_receipt.json').read_text())
        self.assertEqual(receipt['inserted_blank_separators'], 1)
        self.assertTrue((directory / 'stream_00_00.wire.jsonl').exists())

    def test_upstream_failure_is_not_accepted_as_answer(self):
        import httpx
        with self.assertRaises(httpx.RemoteProtocolError):
            self.invoke(['data: {"type":"response.in_progress"}',
                         'data: {"type":"error","code":"upstream_stream_read_error"}', ''])

    def test_http401_still_fails_immediately(self):
        with self.assertRaises(repair.stream.ProtocolViolation):
            self.invoke([], status=401)

    def test_eof_before_completed_still_fails(self):
        import httpx
        with self.assertRaises(httpx.RemoteProtocolError):
            self.invoke(['data: {"type":"response.in_progress"}', ''])

    def test_close_error_after_completion_does_not_discard_answer(self):
        outcome, _ = self.invoke([
            'data: {"type":"response.completed","response":{"id":"offline-response","status":"completed"}}', ''],
            close_error=True)
        self.assertEqual(outcome[0], 'offline answer')


if __name__ == '__main__':
    unittest.main()
