"""Check candidate navigation and preserve the context of frozen Markdown.

Uses Git's tracked tree to distinguish sparse files from genuinely missing
targets. Historical documents are never rewritten by this tool. No network,
training module, data table, or runtime controller is opened.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import posixpath
import re
import subprocess
from urllib.parse import unquote, urlsplit

NAVIGATION = {
    'README.md', 'docs/README.md', 'docs/ARCHITECTURE.md',
    'docs/NEXT_STAGE_DECISION.md', 'docs/model/INPUT_SCHEMA.md',
    'docs/repository/README.md', 'tools/README.md', 'experiments/INDEX.md',
    'experiments/HISTORICAL_DECISIONS.md', 'reports/README.md',
    'studies/README.md', 'studies/active_learning/README.md',
    'studies/active_learning/CW_RESEARCH_INDEX.md', 'studies/predictor/README.md',
    'docs/repository/releases/document-cleanup-20261110/CLEANUP_REPORT.md',
}
HISTORICAL_REFS = (
    '0e33a86d3dea3945acb0a638ca78a8f5065980a5',
    '4ee480378b0293f1985e3ee4a09df2c78f77901e',
    '16eca74769f664bc378a58ccbf380a806723f587',
    'dadd2281fba01eb4414128cbf9feb2859c78544b',
    '2bb086dbd2c23a82485f27f3463b1800298a4f27',
)
GITHUB = 'https://github.com/guoscfish/column_chromatography_prediction'


def git(root, *args, check=True):
    return subprocess.run(['git', '--no-optional-locks', '-C', str(root), *args],
                          capture_output=True, check=check)


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


def check(root, historical_refs=HISTORICAL_REFS):
    root = Path(root).resolve()
    paths = candidate_paths(root)
    dirs = {str(parent) for name in paths for parent in Path(name).parents}
    refs = [ref for ref in historical_refs if git(root, 'cat-file', '-e', ref + '^{commit}', check=False).returncode == 0]
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
        if file.is_file():
            return file.read_text(errors='replace')
        return git(root, 'show', 'HEAD:' + name).stdout.decode(errors='replace')

    for source in sorted(p for p in paths if p.lower().endswith('.md')):
        text = read_text(source)
        for destination, line in markdown_links(text):
            record = {'source': source, 'line': line, 'destination': destination,
                      'scope': 'CURRENT_NAVIGATION' if source in NAVIGATION else 'PRESERVED_DOCUMENT'}
            url = urlsplit(destination)
            if url.scheme or url.netloc:
                match = re.match(re.escape(GITHUB) + r'/(?:blob|tree)/([0-9a-f]{40})/(.*)', destination)
                if match:
                    ref, target = match[1], unquote(urlsplit(match[2]).path)
                    record.update(status='PINNED_GIT_TARGET' if historical_target(ref, target, url.fragment) else 'BROKEN_PINNED_TARGET',
                                  target=target, recovery_ref=ref)
                else:
                    record['status'] = 'EXTERNAL_NOT_FETCHED'
                records.append(record)
                continue
            raw = unquote(url.path)
            if raw.startswith('/'):
                record['status'] = 'LOCAL_ABSOLUTE_REFERENCE'
                records.append(record)
                continue
            target = posixpath.normpath(posixpath.join(posixpath.dirname(source), raw)) if raw else source
            record['target'] = target
            if target == '..' or target.startswith('../'):
                record['status'] = 'OUTSIDE_REPOSITORY'
            elif target in paths or target in dirs:
                record['status'] = 'LOCAL_TARGET' if (root / target).exists() else 'GIT_TRACKED_SPARSE_TARGET'
                if url.fragment and target.endswith('.md') and unquote(url.fragment) not in heading_ids(read_text(target)):
                    record['status'] = 'BROKEN_ANCHOR'
            else:
                record['status'] = 'BROKEN_LOCAL_TARGET'
            if record['status'] in ('BROKEN_LOCAL_TARGET', 'BROKEN_ANCHOR') and source not in NAVIGATION:
                restored = next((ref for ref in refs if historical_target(ref, target, url.fragment)), None)
                if restored:
                    record['status'] = 'HISTORICAL_PINNED_RESOLUTION'
                    record['recovery_ref'] = restored
                    record['resolved_url'] = GITHUB + '/blob/' + restored + '/' + target + ('#' + url.fragment if url.fragment else '')
                    record['note'] = 'Frozen original retained byte-for-byte; this is a separate source-context resolution, not a repaired local link/anchor.'
            records.append(record)
    failures = [r for r in records if r['scope'] == 'CURRENT_NAVIGATION' and
                r['status'] in ('BROKEN_LOCAL_TARGET', 'BROKEN_ANCHOR', 'BROKEN_PINNED_TARGET', 'OUTSIDE_REPOSITORY')]
    unresolved = [r for r in records if r['status'].startswith('BROKEN') and r not in failures]
    return {'schema_version': 1, 'candidate_root': str(root),
            'markdown_files_checked': sum(p.lower().endswith('.md') for p in paths),
            'navigation_files_checked': sorted(paths & NAVIGATION), 'links_checked': len(records),
            'status_counts': dict(Counter(r['status'] for r in records)),
            'navigation_passed': not failures, 'navigation_failures': failures,
            'unresolved_preserved_document_links': unresolved, 'records': records,
            'limits': ['No HTTP fetching; pinned own-repository Git paths verified by local immutable objects.',
                       'Historical source resolutions do not rewrite frozen originals or claim their relative links now work in the candidate.',
                       'Never interprets code examples or opens label/checkpoint payloads.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = check(args.repository)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({k: result[k] for k in ['markdown_files_checked', 'links_checked', 'navigation_passed', 'status_counts']}, ensure_ascii=False))
    return 0 if result['navigation_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
