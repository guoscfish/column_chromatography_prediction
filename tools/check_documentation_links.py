"""Check current Markdown destinations, headings and explicit pinned Git links.

Uses Git's tracked tree to distinguish sparse files from genuinely missing
targets. The archived historical tree is accessed only through explicit pinned links. No network,
training module, data table, or runtime controller is opened.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import posixpath
import re
import subprocess
from urllib.parse import unquote, urlsplit

GITHUB = 'https://github.com/guoscfish/column_chromatography_prediction'


def git(root, *args, check=True):
    return subprocess.run(['git', '--no-optional-locks', '-C', str(root), *args],
                          capture_output=True, check=check,
                          env={**os.environ, 'GIT_NO_LAZY_FETCH': '1', 'GIT_NO_REPLACE_OBJECTS': '1', 'GIT_TERMINAL_PROMPT': '0'})


def mask_code(text, mask_inline=True):
    lines, fence = [], None
    for line in text.splitlines(keepends=True):
        opening = re.match(r'^ {0,3}(`{3,}|~{3,})', line)
        if opening:
            token = opening.group(1)
            if fence is None:
                fence = token
            elif token[0] == fence[0] and len(token) >= len(fence):
                fence = None
            lines.append('\n' if line.endswith('\n') else '')
        elif fence:
            lines.append('\n' if line.endswith('\n') else '')
        else:
            lines.append(re.sub(r'(`+)([^`\n]*?)\1', lambda m: ' ' * len(m[0]), line) if mask_inline else line)
    return ''.join(lines)


def markdown_links(text):
    """Yield actual inline/reference/HTML destinations, ignoring code examples."""
    text = mask_code(text)
    for match in re.finditer(r'!?\[[^\]\n]*\]\(', text):
        start = match.end()
        while start < len(text) and text[start].isspace():
            start += 1
        if start >= len(text):
            continue
        if text[start] == '<':
            end = text.find('>', start + 1)
            if end >= 0:
                yield text[start + 1:end], text.count('\n', 0, match.start()) + 1
            continue
        cursor, depth, escaped = start, 0, False
        while cursor < len(text):
            char = text[cursor]
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == '(':
                depth += 1
            elif char == ')':
                if depth == 0:
                    break
                depth -= 1
            elif char.isspace() and depth == 0:
                break
            cursor += 1
        if cursor > start:
            yield re.sub(r'\\([ ()])', r'\1', text[start:cursor]), text.count('\n', 0, match.start()) + 1
    for match in re.finditer(r'^ {0,3}\[[^\]]+\]:\s*(?:<([^>]+)>|(\S+))', text, re.M):
        yield match[1] or match[2], text.count('\n', 0, match.start()) + 1
    for match in re.finditer(r'\b(?:href|src)=["\']([^"\']+)["\']', text):
        yield match[1], text.count('\n', 0, match.start()) + 1


def heading_ids(text):
    result, counts = set(), Counter()
    for line in mask_code(text, mask_inline=False).splitlines():
        match = re.match(r'^ {0,3}#{1,6}\s+(.+?)\s*#*$', line)
        if not match:
            continue
        value = re.sub(r'<[^>]+>', '', match[1]).strip().lower()
        value = re.sub(r'[^\w\s\-]', '', value, flags=re.UNICODE).replace(' ', '-')
        suffix = '' if counts[value] == 0 else '-' + str(counts[value])
        counts[value] += 1
        result.add(value + suffix)
    result.update(re.findall(r'(?:id|name)=["\']([^"\']+)["\']', text))
    return result


def candidate_paths(root):
    tracked = set(filter(None, git(root, 'ls-files', '-z').stdout.decode().split('\0')))
    deleted = set(filter(None, git(root, 'ls-files', '--deleted', '-z').stdout.decode().split('\0')))
    added = set(filter(None, git(root, 'ls-files', '--others', '--exclude-standard', '-z').stdout.decode().split('\0')))
    return (tracked - deleted) | added


def check(root, *, require_archive_objects=False):
    root = Path(root).resolve()
    paths = candidate_paths(root)
    dirs = {str(parent) for name in paths for parent in Path(name).parents}
    object_cache, records = {}, []

    def object_exists(ref, target):
        key = (ref, target)
        if key not in object_cache:
            object_cache[key] = git(root, 'cat-file', '-e', ref + ':' + target, check=False).returncode == 0
        return object_cache[key]

    def historical_target(ref, target, fragment):
        if not object_exists(ref, target):
            return False
        if fragment and target.endswith('.md'):
            original = git(root, 'show', ref + ':' + target).stdout.decode(errors='replace')
            return unquote(fragment) in heading_ids(original)
        return True

    def read_text(name):
        file = root / name
        if file.is_symlink():
            raise ValueError('documentation symlinks are not read')
        if file.is_file():
            return file.read_text(errors='replace')
        return git(root, 'show', ':' + name).stdout.decode(errors='replace')

    for source in sorted(p for p in paths if p.lower().endswith('.md')):
        if (root/source).is_symlink():
            records.append({'source': source, 'scope': 'CURRENT_NAVIGATION', 'status': 'UNSAFE_DOCUMENT_SYMLINK'})
            continue
        text = read_text(source)
        for destination, line in markdown_links(text):
            record = {'source': source, 'line': line, 'destination': destination,
                      'scope': 'CURRENT_NAVIGATION'}
            url = urlsplit(destination)
            if url.scheme or url.netloc:
                match = re.match(re.escape(GITHUB) + r'/(?:blob|tree)/([0-9a-f]{40})/(.*)', destination)
                if match:
                    ref, target = match[1], unquote(urlsplit(match[2]).path)
                    if git(root, 'cat-file', '-e', ref + '^{commit}', check=False).returncode:
                        status = 'MISSING_ARCHIVE_OBJECT' if require_archive_objects else 'EXTERNAL_PINNED_UNVERIFIED'
                    else:
                        status = 'PINNED_GIT_TARGET' if historical_target(ref, target, url.fragment) else 'BROKEN_PINNED_TARGET'
                    record.update(status=status, target=target, recovery_ref=ref)
                else:
                    record['status'] = 'EXTERNAL_NOT_FETCHED'
                records.append(record)
                continue
            raw = unquote(url.path)
            if raw.startswith('/'):
                record['status'] = 'OUTSIDE_REPOSITORY'
                records.append(record)
                continue
            target = posixpath.normpath(posixpath.join(posixpath.dirname(source), raw)) if raw else source
            record['target'] = target
            if target == '..' or target.startswith('../'):
                record['status'] = 'OUTSIDE_REPOSITORY'
            elif (root/target).is_symlink():
                record['status'] = 'UNSAFE_DOCUMENT_SYMLINK'
            elif target in paths or target in dirs:
                record['status'] = 'LOCAL_TARGET' if (root / target).exists() else 'GIT_TRACKED_SPARSE_TARGET'
                if url.fragment and target.endswith('.md') and unquote(url.fragment) not in heading_ids(read_text(target)):
                    record['status'] = 'BROKEN_ANCHOR'
            else:
                record['status'] = 'BROKEN_LOCAL_TARGET'
            records.append(record)
    bad = {'BROKEN_LOCAL_TARGET', 'BROKEN_ANCHOR', 'BROKEN_PINNED_TARGET',
           'OUTSIDE_REPOSITORY', 'UNSAFE_DOCUMENT_SYMLINK', 'MISSING_ARCHIVE_OBJECT'}
    failures = [r for r in records if r['status'] in bad]
    return {'schema_version': 2, 'candidate_root': str(root),
            'markdown_files_checked': sum(p.lower().endswith('.md') for p in paths),
            'navigation_files_checked': sorted(p for p in paths if p.lower().endswith('.md')),
            'links_checked': len(records), 'status_counts': dict(Counter(r['status'] for r in records)),
            'navigation_passed': not failures, 'navigation_failures': failures, 'records': records,
            'limits': ['All current Markdown is checked; no historical relative-path fallback.',
                       'No HTTP requests or implicit fetch. Pinned own-repository paths are verified if their commit is locally available.',
                       'EXTERNAL_PINNED_UNVERIFIED means the archive must be explicitly fetched to verify it; --require-archive-objects makes this a failure.',
                       'Never interprets code examples or opens label/checkpoint payloads.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--require-archive-objects', action='store_true')
    args = parser.parse_args()
    result = check(args.repository, require_archive_objects=args.require_archive_objects)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({k: result[k] for k in ['markdown_files_checked', 'links_checked', 'navigation_passed', 'status_counts']}, ensure_ascii=False))
    return 0 if result['navigation_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
