#!/usr/bin/env python3
"""Scientist V3.2: seed157 Free-LLM32 L333→L365, with no continuation action."""
import argparse
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
# Match the established scientific environment's safe native-library load order.
import pandas
from src.qgeognn_al.active_learning_v3_2 import study, protocol
from src.qgeognn_al.active_learning_v3_2.transport import settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare','stage','select','advance','status','validate'))
    parser.add_argument('--study-dir', type=Path, default=study.STUDY)
    parser.add_argument('--source-root', type=Path, default=study.ROOT)
    parser.add_argument('--api-key-file', type=Path, default=Path.home()/'.config/qgeognn-scientist/token4research.api-key')
    args = parser.parse_args()
    if args.study_dir.resolve().name != study.STUDY.name:
        parser.error('requires independent V3.2 study directory')
    if args.action == 'prepare':
        result = study.prepare(args.study_dir, source_root=args.source_root, config=settings())
    elif args.action == 'validate':
        result = protocol.validate(args.study_dir)
    elif args.action == 'status':
        result = study.status(args.study_dir)
    else:
        if args.action == 'select':
            os.environ['SCIENTIST_API_KEY'] = args.api_key_file.read_text().strip()
        try:
            result = getattr(study, args.action)(157, 'free_llm32_scientist_v3_2', 0, root=args.study_dir)
        finally:
            os.environ.pop('SCIENTIST_API_KEY', None)
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
