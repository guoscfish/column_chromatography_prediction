"""Continue frozen 813 studies, then inherit them into a separate 1005 study."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.qgeognn_al.active_learning_v2 import dual_expert_study as study
from src.qgeognn_al.active_learning_v2.dual_expert_transport import settings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, choices=(157, 6101), required=True)
    parser.add_argument('--rounds', type=int, required=True)
    parser.add_argument('--extended', action='store_true')
    args = parser.parse_args()
    parent = ('sol61_xhigh_seed157_from621_to813_20261014' if args.seed == 157
              else 'sol61_xhigh_seed6101_to813_20261014')
    if args.extended:
        # Only the newly authorized stopping budget changes. Scientific source,
        # prompts, batches, optimizer, truth barrier and checkpoint rule stay fixed.
        study.BUDGETS = tuple(range(333, 1006, 32))
        study.configure_revision(f'sol61_xhigh_seed{args.seed}_from813_to1005_20261021')
        start = 15
        from_revision = None if (study.STUDY / 'protocol.json').exists() else parent
    else:
        study.configure_revision(parent)
        start = 9 if args.seed == 157 else 0
        from_revision = None
    result = study.run(args.seed, study.METHOD, args.rounds,
                       settings('responses', 'gpt-6.1-sol',
                                'https://token4research.cn/v1', 'xhigh'),
                       start_round=start, parent_revision=from_revision)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
