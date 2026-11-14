"""PAS (Persona Appropriateness Score).

PAS measures whether a response uses the *right* persona attributes for the current
dialogue context. It compares two topic distributions over the persona topics of the
speaker:

  * the **query** distribution: topics labeled on the partner's last utterance
    (i.e. which persona topics the context actually calls for), and
  * the **response** distribution: topics of the persona attributes the generated
    response uses.

Each distribution is a count vector of length ``1 + |persona topics|``. Slot 0 means
"no persona topic" and is used when the query (or response) touches none of the
persona topics. The vectors are normalized and compared with the Jensen-Shannon
distance (base 2, so it lies in [0, 1]):

    PAS = 1 - JSD(p_query, p_response)        (higher is better, 1 = perfect match)

Because slot 0 is part of the distribution, PAS penalizes both
  * **overuse**  - the query needs no persona, but the response uses some, and
  * **underuse** - the query needs a persona topic, but the response ignores it.

The corpus-level PAS reported in the paper is the harmonic mean of the average PAS
on the two subsets of test samples:
  * *relevant*   - the query matches at least one persona attribute, and
  * *irrelevant* - the query matches none.
so that a model cannot score well by always (or never) using its persona.

This module depends only on NumPy. Quick example::

    from utility.pas import persona_appropriateness_score
    # Persona topics of the speaker: 0 (hobby), 3 (job), 4 (appearance)
    # Query asks about the job (topic 3); response talks about hobby + job.
    persona_appropriateness_score(query_topics=[3], response_topics=[0, 3],
                                  persona_topics=[0, 3, 4])   # -> ~0.44

See ``evaluate_pas.py`` at the repository root for a command-line entry point.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Distribution helpers
# ---------------------------------------------------------------------------

def _kldiv(P, Q):
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)

    eps = 1e-12
    P = np.clip(P, eps, 1)
    Q = np.clip(Q, eps, 1)

    return np.sum(P * np.log2(P / Q))


def js_dist(P, Q):
    """Jensen-Shannon distance (sqrt of the base-2 JS divergence), in [0, 1]."""
    P = np.array(P)
    Q = np.array(Q)
    M = 0.5 * (P + Q)
    return np.sqrt(0.5 * (_kldiv(P, M) + _kldiv(Q, M)))


def _to_prob(vec):
    v = np.asarray(vec, dtype=float)
    s = v.sum()
    if s <= 0:
        v = np.zeros_like(v, dtype=float)
        v[0] = 1.0
        s = 1.0
    return v / s


def _topic_vector(topics, topic_vocab):
    """Count vector of length 1 + |topic_vocab|; slot 0 = "no persona topic"."""
    vec = [0] * (len(topic_vocab) + 1)
    topics = [t for t in topics if t in topic_vocab]
    if len(topics) == 0:
        vec[0] = 1
    else:
        for t in topics:
            vec[topic_vocab.index(t) + 1] += 1
    return vec


# ---------------------------------------------------------------------------
# Sample-level PAS
# ---------------------------------------------------------------------------

def persona_appropriateness_score(query_topics, response_topics, persona_topics):
    """Sample-level PAS = 1 - JSD(query topic dist., response topic dist.).

    Args:
        query_topics:    topic ids labeled on the partner's last utterance.
        response_topics: topic ids of the persona attributes used by the response
                         (one entry per used attribute's topic; duplicates count).
        persona_topics:  topic ids of all persona attributes given to the speaker.
                         Defines the topic vocabulary; topics outside it are ignored.

    Returns:
        float in [0, 1]; higher means more appropriate persona use.
    """
    topic_vocab = sorted(set(persona_topics))
    p_query = _to_prob(_topic_vector(query_topics, topic_vocab))
    p_resp = _to_prob(_topic_vector(response_topics, topic_vocab))
    return 1 - js_dist(p_query, p_resp)


# ---------------------------------------------------------------------------
# Helpers for the labeled-data format used in this repository
# ---------------------------------------------------------------------------

def get_topic(persona_list, agent_persona, user_persona, self_labels, partner_labels):
    """Flatten the topic labels of the given persona attributes."""
    topic_list = []
    for persona in persona_list:
        if persona in agent_persona:
            topic_list.extend(self_labels[agent_persona.index(persona)])
        elif persona in user_persona:
            topic_list.extend(partner_labels[user_persona.index(persona)])
    return topic_list


def get_query_labeled_persona(query_labeled_topic, agent_persona_list, self_labels):
    """Persona attributes whose topic appears in the query's topic labels."""
    query_labeled_persona = []
    for agent_persona in agent_persona_list:
        agent_persona_topic_list = self_labels[agent_persona_list.index(agent_persona)]
        if any(t in query_labeled_topic for t in agent_persona_topic_list):
            query_labeled_persona.append(agent_persona)
    return query_labeled_persona


def is_persona_relevant(record, setting="self"):
    """True if the query matches at least one persona attribute of the record.

    ``setting="self"`` uses only the speaker's persona; ``"partner"`` also uses the
    partner's persona (same split as ``main_test.py``).
    """
    if setting == "self":
        personas, labels = record["agent_persona"], record["self_labels"]
    else:
        personas = record["agent_persona"] + record["user_persona"]
        labels = record["self_labels"] + record["partner_labels"]
    return len(get_query_labeled_persona(record["query_labeled_topic"], personas, labels)) > 0


def pas_from_record(record, persona_scope="agent"):
    """Sample-level PAS for one labeled record.

    Required keys: ``agent_persona``, ``user_persona``, ``self_labels``,
    ``partner_labels``, ``query_labeled_topic``, ``generated_response_labeled_persona``.

    ``persona_scope="agent"`` -> "PAS" column (speaker's persona topics only);
    ``persona_scope="both"``  -> "PAS(Both)" column (speaker + partner persona topics).
    """
    agent_persona, user_persona = record["agent_persona"], record["user_persona"]
    self_labels, partner_labels = record["self_labels"], record["partner_labels"]

    response_topics = get_topic(record["generated_response_labeled_persona"],
                                agent_persona, user_persona, self_labels, partner_labels)
    scope = agent_persona if persona_scope == "agent" else agent_persona + user_persona
    persona_topics = get_topic(scope, agent_persona, user_persona, self_labels, partner_labels)

    return persona_appropriateness_score(record["query_labeled_topic"], response_topics, persona_topics)


# ---------------------------------------------------------------------------
# Corpus-level PAS
# ---------------------------------------------------------------------------

def _harmonic_mean(a, b):
    if a is None or b is None or a <= 0 or b <= 0:
        return 0.0
    return 2 / (1 / a + 1 / b)


def corpus_pas(records, setting="self", persona_scope="agent"):
    """Corpus-level PAS as reported in the paper.

    Records without ``generated_response_labeled_persona`` are skipped (as in
    ``main_test.py``).

    Returns a dict with:
        relevant:   mean PAS over samples whose query matches a persona attribute,
        irrelevant: mean PAS over samples whose query matches none,
        pas:        harmonic mean of the two (the number reported in the paper),
        n_relevant, n_irrelevant: subset sizes.
    """
    scores = {True: [], False: []}
    for r in records:
        if "generated_response_labeled_persona" not in r:
            continue
        scores[is_persona_relevant(r, setting)].append(pas_from_record(r, persona_scope))

    rel = float(np.mean(scores[True])) if scores[True] else None
    irrel = float(np.mean(scores[False])) if scores[False] else None
    return {
        "relevant": rel,
        "irrelevant": irrel,
        "pas": _harmonic_mean(irrel, rel),
        "n_relevant": len(scores[True]),
        "n_irrelevant": len(scores[False]),
    }
