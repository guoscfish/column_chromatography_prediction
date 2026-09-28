#!/usr/bin/env python3
"""V3: explicit actions only. dry-run/stage never invoke an LLM or reveal new labels."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.qgeognn_al.active_learning_v3 import study, protocol
from src.qgeognn_al.active_learning_v3.transport import settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'validate', 'status', 'stage', 'dry-run', 'select', 'advance'))
    parser.add_argument('--study-dir', type=Path, default=study.STUDY)
    parser.add_argument('--source-root', type=Path, default=study.ROOT)
    parser.add_argument('--seed', type=int, choices=study.SEEDS)
    parser.add_argument('--method', choices=study.METHODS)
    parser.add_argument('--round', type=int, choices=range(6), default=0)
    parser.add_argument('--backend', choices=('codex_cli', 'responses'), default='codex_cli')
    parser.add_argument('--model', default='gpt-6-sol')
    parser.add_argument('--base-url', default='https://token4research.cn')
    parser.add_argument('--effort', choices=('low', 'medium', 'high', 'xhigh'), default='high')
    args = parser.parse_args()
    # Prevent accidentally directing V3 actions at an existing V2 study.
    if args.study_dir.resolve().name != 'qgeognn_v2_row_llm_scientist_v3':
        parser.error('--study-dir must be an independent qgeognn_v2_row_llm_scientist_v3 directory')
    if args.action == 'prepare':
        result = study.prepare(args.study_dir, source_root=args.source_root,
            config=settings(args.backend, args.model, args.base_url, args.effort))
    elif args.action == 'validate':
        result = protocol.validate(args.study_dir)
    elif args.action == 'status':
        result = study.status(args.study_dir)
    else:
        if args.seed is None or args.method is None:
            parser.error('--seed and --method required')
        action = 'stage' if args.action == 'dry-run' else args.action
        result = getattr(study, action)(args.seed, args.method, args.round, root=args.study_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
