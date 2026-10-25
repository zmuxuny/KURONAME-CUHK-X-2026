#!/usr/bin/env python3
"""Apply the fixed Train-selected option-space fusion to a complete candidate."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from cuhkx.external_candidate_contract import selected_option_target_recipe
from cuhkx.harn_option_fusion import option_fusion
from scripts.run_r34_external_candidate import run


def visual_fusion(qa, r18, old, new, vm, dictionary):
    return option_fusion(qa, [[('r18', r18)], [('old_r34', old), ('aligned', new)],
                             [('videomae', vm)]], dictionary, training_paths=False)[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['config', 'data-dir', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    run(parser.parse_args(), recipe_validator=selected_option_target_recipe,
        raw_script='reuse_paired_target_harn.py', visual_fusion=visual_fusion,
        scope_note='Preselected fixed option-space visual averaging with paired-depth target R34half '
                   'and unchanged temporal-linear parent. Reuse the same final model original-media '
                   '64unit122view evidence; new option consumer and complete SAT, no new raw forwards '
                   'or fitting. Original class-space evidence remains a diagnostic in combined_evidence.npz.')
