"""Fixed hierarchical visual fusion after conditioning each expert on QA options."""
from __future__ import annotations

from cuhkx.harn import constrained_scores
from cuhkx.harn_nested import QA_COLUMNS, normalized_option_scores
from cuhkx.metadata import feature_unit
from cuhkx.submission import available_letters


def option_fusion(qa, groups, bundle, *, training_paths):
    """Average available experts within families, then average available families.

    A group is a list of (name, saved evidence) pairs. The R34 group may have
    the old and new expert; R18 and VideoMAE each remain a single family.
    """
    lookups = {name: {u: i for i, u in enumerate(e['units'])}
               for group in groups for name, e in group}
    result, audit = {}, []
    for _, row in qa[QA_COLUMNS].iterrows():
        unit = feature_unit(row['path'], training=training_paths)
        mapping = bundle['action_mapping' if row['category'] == 'single' else 'object_mapping']
        letters = available_letters(row)
        families = []
        for group in groups:
            experts = []
            for name, evidence in group:
                i = lookups[name][unit]
                classes = evidence['visual_classes']
                scores, _ = constrained_scores(row, evidence['visual_probabilities'][i, classes],
                                                classes, mapping)
                mass = sum(scores.values())
                present = bool(evidence['visual_present'][i])
                audit.append({'qa_id': row['qa_id'], 'expert': name,
                              'present': present, 'option_mass': float(mass)})
                if present and mass > 1e-12:
                    experts.append(normalized_option_scores(scores))
            if experts:
                families.append({k: sum(e[k] for e in experts) / len(experts) for k in letters})
        result[row['qa_id']] = ({k: sum(g[k] for g in families) / len(families) for k in letters}
                                 if families else {k: 0.0 for k in letters})
    return result, audit
