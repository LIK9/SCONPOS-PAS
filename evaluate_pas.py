"""Compute PAS (Persona Appropriateness Score) for a topic-labeled response file.

Standalone entry point: needs only NumPy (no GPU, no word2vec / NLI models).

Usage:
    python evaluate_pas.py --input path/to/labeled_responses.json [--setting self|partner]

Input: a JSON list of records, each with
    agent_persona                       list[str]        speaker's persona attributes
    self_labels                         list[list[int]]  topic ids of each agent_persona entry
    user_persona                        list[str]        partner's persona attributes
    partner_labels                      list[list[int]]  topic ids of each user_persona entry
    query_labeled_topic                 list[int]        topic ids of the partner's last utterance
    generated_response_labeled_persona  list[str]        persona attributes used by the response

This is the format written by processing.py's persona_labeling step, i.e.
./result/<dataset>/<model>/main/response_labeled/<output_path>/processed/<output_path>.json

Output: PAS on the persona-relevant / -irrelevant subsets and their harmonic mean (the
number reported in the paper), for both the "PAS" (speaker persona only) and
"PAS(Both)" (speaker + partner persona) variants -- the same values main_test.py
writes to its .xlsx report with --test_task eval_ovs.
"""

import argparse
import json

from utility.pas import corpus_pas


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="topic-labeled response JSON file")
    parser.add_argument("--setting", default="self", choices=["self", "partner"],
                        help="'self': split relevant/irrelevant by the speaker's persona; "
                             "'partner': by speaker + partner persona (default: self)")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        records = json.load(f)

    print(f"{'':10s} {'relevant':>9s} {'irrelevant':>11s} {'PAS (HM)':>9s}")
    for name, scope in [("PAS", "agent"), ("PAS(Both)", "both")]:
        r = corpus_pas(records, setting=args.setting, persona_scope=scope)
        fmt = lambda v: f"{v:.4f}" if v is not None else "n/a"
        print(f"{name:10s} {fmt(r['relevant']):>9s} {fmt(r['irrelevant']):>11s} {r['pas']:>9.4f}")
    print(f"(n_relevant={r['n_relevant']}, n_irrelevant={r['n_irrelevant']})")


if __name__ == "__main__":
    main()
