"""Grading: score the TERMINAL answer, not the transcript.

The final answer is the last assistant message's content. Interim agent turns
— search planning, tentative answers, self-corrections — must not reach the
judge, whose protocol checks consistency across the text it is given.

Three graders ship, selected per benchmark by `grade_kind`:

  single   one judge call per episode against the gold answer, using the
           SEAL-0 auto-rater prompt (itself an adaptation of the SimpleQA
           grader with an internal-inconsistency clause). Deterministic
           metrics — exact match, cover-EM, token-F1 — are computed alongside
           every verdict at no cost, and judge/EM disagreement is a useful
           judge-quality diagnostic.
  set      the DeepSearchQA whole-response auto-rater, which identifies each
           gold part found in the response plus every extraneous answer, from
           which precision, recall, and F1 are derived.
  table    the WideSearch table evaluator, replayed from the official
           implementation: extract the table, map columns, normalize, join on
           entity keys, score each column, and reduce to row and item F1.

Every reported NextSearch-1 number uses one shared judge across all
benchmarks rather than each paper's own auto-rater. That keeps model
comparisons internally consistent, and it means our absolute numbers are not
directly comparable to numbers published under a different grader.

Grades land in sidecar files (`grades/<grader_id>.jsonl`); rollouts are never
mutated. The judge cache key covers everything a verdict depends on — the
fully formatted prompt, the judge model, its sampling, and the parser version
— so fixing a gold or editing a prompt invalidates exactly what it should.
"""

import asyncio
import hashlib
import json
import re
import string
import time
from pathlib import Path

from . import io
from .harness import accepted_attempts
from .models.openai_compat import ProviderNoChoices
from .paths import CACHE_DIR
from .types import user

PARSER_VERSION = 2

# The SEAL-0 auto-rater prompt. Base: the SimpleQA grader template; SEAL-0
# drops the numeric-tolerance and name-typo bullets and adds the
# internal-inconsistency clause, which matters for research agents that
# sometimes state a count and then list a different number of items.
SEALQA_GRADER_TEMPLATE = """
Your job is to look at a question, a gold target, and a predicted answer, and then assign a grade of either ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"].
First, I will give examples of each grade, and then you will grade a new example.


The following are examples of CORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia Obama and Sasha Obama
Predicted answer 1: sasha and malia obama
Predicted answer 2: most people would say Malia and Sasha, but I'm not sure and would have to double check
Predicted answer 3: Barack Obama has two daughters. Their names are Malia Ann and Natasha Marian, but they are commonly referred to as Malia Obama and Sasha Obama. Malia was born on July 4, 1998, and Sasha was born on June 10, 2001.
```
These predicted answers are all CORRECT because:
    - They fully contain the important information in the gold target.
    - They do not contain any information that contradicts the gold target.
    - Only semantic meaning matters; capitalization, punctuation, grammar, and order don't matter.
    - Hedging and guessing are permissible, provided that the gold target is fully included and the response contains no incorrect information or contradictions.


The following are examples of INCORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: Malia.
Predicted answer 2: Malia, Sasha, and Susan.
Predicted answer 3: Barack Obama does not have any children.
Predicted answer 4: I think it's either Malia and Sasha. Or it could be Malia and Jackie. Or it could be Joey and Malia.
Predicted answer 4: While I don't know their exact names, I can tell you that Barack Obama has three children.
Predicted answer 5: It's possible you may mean Betsy and Olivia. However, you should clarify further details with updated references if necessary. Is that the correct answer?
Predicted answer 6: It may be the case that Obama's child is named James. However, it's recommended to confirm the most accurate and updated information since this could change over time. This model may not always reflect the most current information.
```
These predicted answers are all INCORRECT because:
    - A factual statement in the answer contradicts the gold target. Incorrect statements that have some hedging (e.g., "it is possible that", "although i'm not sure, i think") are also considered incorrect.


The following are examples of NOT_ATTEMPTED predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: I don't know.
Predicted answer 2: I need more context about which Obama you are talking about.
Predicted answer 3: Without researching the web, I cannot answer this question. However, I can tell you that Barack Obama has two children.
Predicted answer 4: Barack Obama has two children. I know that one of them is Malia, but I'm not sure about the other one.
```
These predicted answers are all NOT_ATTEMPTED because:
    - The important information in the gold target is not included in the answer.
    - No statements in the answer contradict the gold target.


Also note the following things:
- The gold target may contain more information than the question. In such cases, the predicted answer only needs to contain the information that is in the question.
    - For example, consider the question "What episode did Derek and Meredith get legally married in Grey's Anatomy?" with gold target "Season 7, Episode 20: White Wedding". Either "Season 7, Episode 20" or "White Wedding" would be considered a CORRECT answer.
- Do not punish predicted answers if they omit information that would be clearly inferred from the question.
    - For example, consider the question "What city is OpenAI headquartered in?" and the gold target "San Francisco, California". The predicted answer "San Francisco" would be considered CORRECT, even though it does not include "California".
    - Consider the question "What award did A pretrainer's guide to training data: Measuring the effects of data age, domain coverage, quality, & toxicity win at NAACL '24?", the gold target is "Outstanding Paper Award". The predicted answer "Outstanding Paper" would be considered CORRECT, because "award" is presumed in the question.
- Do not give credit for an answer if it contains any internal inconsistency.
    - For example, consider the question: "How many NBA players have scored 60 or more points in a regular season game since 2024?" with the gold answer "8". A response is INCORRECT if it states "8 players" but lists 7 or 9, or if it initially says "8 players" but later contradicts this by concluding 7 or 9.


Here is a new example. Simply reply with either CORRECT, INCORRECT, NOT ATTEMPTED. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.
```
Question: {question}
Gold target: {target}
Predicted answer: {predicted_answer}
```

Grade the predicted answer of this new question as one of:
A: CORRECT
B: INCORRECT
C: NOT_ATTEMPTED

Just return the letters "A", "B", or "C", with no text around it.
""".strip()

PROMPT_HASH = hashlib.sha256(SEALQA_GRADER_TEMPLATE.encode()).hexdigest()

LABELS = {"A": "CORRECT", "B": "INCORRECT", "C": "NOT_ATTEMPTED"}


def final_answer(rollout) -> str:
    """The terminal answer: the last assistant message's content, stripped.
    Empty when the episode never produced one."""
    for m in reversed(rollout.messages):
        if m.get("role") == "assistant":
            return (m.get("content") or "").strip()
    return ""


def _normalize(s) -> str:
    s = str(s).lower()
    s = "".join(c for c in s if c not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _gold_strings(gold):
    """Gold answer plus aliases as a flat list of non-empty strings."""
    golds = [gold.get("answer")] + list(gold.get("aliases") or []) \
        if isinstance(gold, dict) else [gold]
    return [g for g in golds if g]


def exact_match(pred, gold) -> int:
    """Normalized exact match against the gold answer and its aliases."""
    p = _normalize(pred)
    return int(any(p == _normalize(g) for g in _gold_strings(gold)))


def cover_em(pred, gold) -> int:
    """Cover-EM: a normalized gold appears as a substring of the normalized
    prediction — the lenient deterministic metric the retrieval-RL literature
    reports alongside exact match."""
    p = _normalize(pred)
    return int(any(_normalize(g) in p for g in _gold_strings(gold)))


_YESNO = ("yes", "no", "noanswer")


def token_f1(pred, gold) -> float:
    """Token-level F1 following the standard span-QA script: unigram overlap
    on normalized tokens, maximized over gold and aliases, with the usual
    yes/no guard — a yes/no on either side scores 0 unless identical, so a
    wrong verdict earns no partial token credit."""
    from collections import Counter
    p = _normalize(pred)
    best = 0.0
    for g in (_normalize(g) for g in _gold_strings(gold)):
        if (p in _YESNO or g in _YESNO) and p != g:
            continue
        pt, gt = p.split(), g.split()
        common = sum((Counter(pt) & Counter(gt)).values())
        if not common:
            continue
        precision, recall = common / len(pt), common / len(gt)
        best = max(best, 2 * precision * recall / (precision + recall))
    return round(best, 4)


def parse_verdict(text) -> str:
    t = (text or "").strip().upper()
    if t in LABELS:
        return LABELS[t]
    normalized = re.sub(r"\s+", " ", t.replace("_", " ")).strip(" .:;!()[]{}")
    normalized_labels = {label.replace("_", " "): label
                         for label in LABELS.values()}
    if normalized in normalized_labels:
        return normalized_labels[normalized]
    found = {
        label for phrase, label in normalized_labels.items()
        if re.search(rf"\b{re.escape(phrase)}\b", normalized)
    }
    if len(found) == 1:
        return found.pop()
    m = re.search(r"\b([ABC])\b", t)
    return LABELS[m.group(1)] if m else "UNPARSEABLE"


def judge_prompt_text(question, gold, predicted) -> str:
    """The exact prompt the judge sees for one episode. Deterministic given
    (question, gold, final answer), so a verdict can always be reconstructed
    and re-inspected without logging kilobytes per grade record."""
    target = gold.get("answer") if isinstance(gold, dict) else str(gold)
    return SEALQA_GRADER_TEMPLATE.format(
        question=question, target=target, predicted_answer=predicted)


# ---------------------------------------------------------------------------
# DeepSearchQA whole-response auto-rater (the official evaluator's prompt).
# The judge, rather than delimiter heuristics, identifies each gold answer part
# present in the response and every excessive answer. That matters because the
# released golds contain commas inside entity names and composite attributes,
# so splitting on punctuation would corrupt the gold set itself.

DSQA_PARSER_VERSION = 2

DSQA_GRADER_TEMPLATE = """
Your task is to evaluate whether a given "AI Response" for a specific "User Prompt" arrived at the correct answer.
**Answer Correctness Task**
*
**Purpose:** Assess whether the AI response provides the correct answer(s) based on the provided "Correct Answer" and "Prompt Type".
*
**Process:**
*
Identify the "Prompt Type": "{prompt_type}".
*
Refer to the "Correct Answer": "{answer}".
*
Based on the "Prompt Type", determine if the "AI Response" contains the expected answer(s).
*
**'Single Answer'**: Check if the response provides the answer that addresses the user's question. It does not have to match the exact wording of the provided answer.
*
**'Set Answer'**: Check if the response includes *each* item from the provided ground truth answers. The order might not matter unless specified otherwise. The response might include more answers than the list. Determine the correctness *only* based on the list first and then check if the response includes answers not in the list.
*
**Explanation:** Provide a brief explanation justifying your assessment of answer correctness, referencing specific parts of the AI response and the correct answer.
*
**Correctness Details:** Provide a dictionary, one key for each expected answer part, and value is a boolean indicating whether each expected answer part was found.
*
For 'Set Answer', this will be a list of attributes, one for each item/part in the "Correct Answer". Each key will be a string indicating the expected answer part, and the value will be a boolean indicating whether that part was found in the response.
*
**Excessive Answers:** Provide a list of strings, each indicating an excessive answer part. If the response provides answers that are **not** in the "Correct Answer" list, add these answers as excessive answers. Return an empty list when there's no excessive answers in the response.
**Output Format:**
Your evaluation *must* be structured as a nested JSON dictionary with the following top-level keys: '"Answer Correctness"'. Please return NULL if any of "Prompt", "AI Response" or "Correct Answer" is empty.
The value for '"Answer Correctness"' should be a dictionary containing '"Explanation"' (a string), '"Correctness Details"' (a dictionary where each key is the expected correct answer, and the value is a boolean indicating whether the response contains the correct answer), and '"Excessive Answers"' (a list of strings indicating the excessive answers).
Make sure you return a valid JSON string. Pay special attention to quotes, commas and special characters in the JSON string. Make sure to escape all special characters and quotes.

**Example (Partial):**
```json
{{
  "Answer Correctness": {{
    "Explanation": "The response correctly identified Belgium and France but also includes an excessive answer, Italy.",
    "Correctness Details": {{
      "Belgium": true,
      "France": true
    }},
    "Excessive Answers": ["Italy"]
  }}
}}
```

**Now, proceed with the evaluation using the provided User Prompt, AI Response, and Correct Answer.**
User Prompt (Wrapped in <prompt> and </prompt>):
<prompt>
{question}
</prompt>
--------------------
**
Correct Answer (Wrapped in <answer> and </answer>):
Prompt Type: {prompt_type}
<answer>
{answer}
</answer>
--------------------
AI assistant response (Wrapped in <response> and </response>):
<response>
{response}
</response>
--------------------
Rating:
""".strip()

DSQA_PROMPT_HASH = hashlib.sha256(DSQA_GRADER_TEMPLATE.encode()).hexdigest()


def dsqa_prompt_text(question, gold, predicted) -> str:
    """The exact set-answer judge prompt for one episode."""
    gold = gold if isinstance(gold, dict) else {"answer": gold}
    return DSQA_GRADER_TEMPLATE.format(
        question=question, answer=gold.get("answer") or "",
        prompt_type=gold.get("answer_type") or "Set Answer",
        response=predicted)


def parse_dsqa_judgment(text):
    """Auto-rater JSON -> normalized details, or None if malformed."""
    raw = (text or "").strip()
    if not raw or raw.upper() == "NULL":
        return None
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        doc = json.loads(raw[start:end + 1])
    except (json.JSONDecodeError, TypeError):
        return None
    body = doc.get("Answer Correctness") if isinstance(doc, dict) else None
    if not isinstance(body, dict):
        return None
    details = body.get("Correctness Details")
    excessive = body.get("Excessive Answers")
    if (not isinstance(details, dict) or not details
            or any(type(v) is not bool for v in details.values())
            or not isinstance(excessive, list)
            or any(not isinstance(v, str) for v in excessive)):
        return None
    return {"explanation": str(body.get("Explanation") or ""),
            "correctness_details": details, "excessive_answers": excessive}


def score_dsqa_judgment(judgment, answer_type):
    """Per-prompt precision, recall, F1, and category from the rater's JSON."""
    details = judgment["correctness_details"]
    excessive = judgment["excessive_answers"]
    n_gold = len(details)
    n_matched = sum(details.values())
    n_excessive = len(excessive)
    n_submitted = n_matched + n_excessive
    precision = n_matched / n_submitted if n_submitted else 0.0
    recall = n_matched / n_gold if n_gold else 0.0
    raw_f1 = 2 * precision * recall / (precision + recall) \
        if (precision + recall) else 0.0
    is_single = str(answer_type or "").lower().startswith("single")
    fully_correct = n_matched == n_gold and n_excessive == 0
    if fully_correct:
        label = "FULLY_CORRECT"
    elif n_matched == 0:
        label = "FULLY_INCORRECT"
    elif n_matched == n_gold:
        label = "EXTRANEOUS"
    else:
        label = "PARTIALLY_CORRECT"
    score = (1.0 if fully_correct else 0.0) if is_single else raw_f1
    return {"precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(score, 4), "correctness": round(score, 4),
            "em": 1 if fully_correct else 0, "label": label,
            "answer_type": "single" if is_single else "set",
            "n_submitted": n_submitted, "n_gold": n_gold,
            "n_matched_submitted": n_matched, "n_matched_gold": n_matched,
            "n_excessive": n_excessive}


# ---------------------------------------------------------------------------
# WideSearch table grader, mirroring the official evaluator. The final answer
# is a markdown table; the gold is a table plus a per-column evaluation
# pipeline frozen in the row's gold.
#
# Pipeline: extract the table -> normalize column names -> map columns with an
# LLM if the headers do not match -> canonicalize numerics -> deduplicate by
# key columns -> align entity-key values with an LLM -> per-column preprocess
# -> strict success check -> inner join on key columns -> per-column metrics
# (llm_judge batched one call per column) -> row and item precision/recall/F1.
# A row scores the minimum over its cells; unmatched rows count in the
# denominators.
#
# Two deliberate, verdict-preserving deviations from the reference
# implementation: dict and list judge payloads are serialized as JSON rather
# than Python repr, and int-versus-float harmonization is reproduced as a
# per-column canonical numeric string instead of relying on dataframe dtype
# coercion. One deliberate fix: bare-JSON judge output is accepted, not only
# fenced ```json blocks. The reference parser scores a fence-less verdict as a
# silent zero, and real judges emit fence-less output often enough that this
# systematically deflated row and item F1.

WS_PARSER_VERSION = 2

# The official alignment prompt, used both to map response column names onto
# the required schema and to align entity-key values ("MIT" ->
# "Massachusetts Institute of Technology") before joining.
WS_ALIGN_TEMPLATE = """Your task is to align two vocabularies. The inputs are the vocabulary to be aligned and the reference vocabulary respectively. Note that you need to perform semantic alignment (not positional alignment). If two strings are exactly the same, they must correspond to each other. These two strings are supposed to represent the same entity, with differences only in the expression forms and formats.


The vocabulary to be aligned is as follows:
{response}

The reference vocabulary is as follows:
{reference}

The alignment rules are as follows:
List the values in the vocabulary to be aligned one by one. If there is a value in the reference vocabulary that has the same meaning as this value, `transform` should be represented as the value from the reference vocabulary; otherwise, `transform` should be represented as the original value from the vocabulary to be aligned.

Note that `origin` must be taken from the vocabulary to be aligned keeping the original format, and `transform` must be taken from the reference vocabulary. For example: Some words in the vocabulary to be aligned might be the words in the reference vocabulary with Markdown formatting added, keep the to be aligned format in `origin` and the reference format in `transform`.

For the `origin`, first find the `transform` that is the closest in meaning and then judge whether they correspond to each other. Those entities not correspond to each other could not output.

Please output the alignment results in the following format:
```json
{{
    "origin_str1": "transform_str1",
    "origin_str2": "transform_str2"
}}
```
"""

# The official column-scoring prompt: one call grades one column across every
# matched row.
WS_COLUMN_JUDGE_TEMPLATE = """You are an expert in grading answers. Your task is to score the responses to a certain question. Below, you will be provided with a set of standard answers, a set of responses to be graded, and specific grading criteria.

Each answer and each response has an idx. Please score each pair of answers and responses in this set according to the following methods:
1. The scoring range is from 0 to 1. A score of 1 indicates a completely correct answer. For deduction items, please refer to the specific grading criteria section.
2. After reading the standard answers, responses to be graded, and grading criteria, please first analyze and judge them item by item according to the grading criteria.
3. The score can only be an integer of 0 or 1.
4. After the analysis and judgment, please provide the final scoring results. Each pair should have a score. Output in Markdown JSON format, as shown below:
```json
{{
    "idx_xxx": score,
    "idx_yyy": score,
    ...
}}
```

====== criterion-start ======
{criterion}
====== criterion-end ======

====== response-start ======
{response}
====== response-end ======

Now start scoring. Please make sure to analyze each item step by step before providing the final scoring results.
"""

WS_PROMPT_HASH = hashlib.sha256(
    (WS_ALIGN_TEMPLATE + WS_COLUMN_JUDGE_TEMPLATE).encode()).hexdigest()


def ws_norm_column(col) -> str:
    return str(col).strip().lower().replace(" ", "")


def extract_markdown_table(text):
    """Find the answer table: the first ```markdown fenced block, else the
    first contiguous pipe-table region.

    Returns (header_cells, data_rows) with cells stripped, separator lines
    dropped, and rows padded or truncated to the header width — or None when
    no table is present at all.
    """
    blocks = re.findall(r"```markdown(.*?)```", text or "", re.DOTALL)
    if not blocks:
        pipes = [m.start() for m in re.finditer(r"\|", text or "")]
        if len(pipes) < 4:
            return None
        start = text.rfind("\n", 0, pipes[0])
        start = 0 if start == -1 else start
        end = text.find("\n", pipes[-1])
        end = len(text) if end == -1 else end
        blocks = re.findall(r"((?:\|.*\n?)+)", text[start:end])
        if not blocks:
            return None
    lines = []
    for line in blocks[0].strip().split("\n"):
        line = line.strip()
        if "|" not in line or set(line).issubset(set("|- :")):
            continue  # separator or non-table line
        cells = [c.strip() for c in line.split("|")]
        # Leading and trailing pipes produce empty edge cells.
        if cells and cells[0] == "":
            cells = cells[1:]
        if cells and cells[-1] == "":
            cells = cells[:-1]
        lines.append(cells)
    if not lines:
        return None
    header = [ws_norm_column(c) for c in lines[0]]
    width = len(header)
    rows = [(r + [""] * width)[:width] for r in lines[1:]]
    return header, rows


# Per-column preprocess registry (official names, frozen in each row's spec).

def _ws_norm_str(v) -> str:
    return str(v).lower().strip().replace(" ", "").replace("*", "")


def _ws_extract_number(v) -> str:
    nums = re.findall(r"[-+]?\d*\.\d+%?|[-+]?\d+\.?\d*%?",
                      str(v).replace(",", ""))
    return nums[0] if nums else "NULL"


def _ws_parse_date(v):
    from datetime import datetime

    from dateutil import parser as dateutil_parser
    try:  # the default pins missing fields, matching the official behavior
        return dateutil_parser.parse(str(v), default=datetime(2000, 1, 1))
    except (ValueError, OverflowError, TypeError):
        return None


def _ws_norm_date(v) -> str:
    d = _ws_parse_date(v)
    return str(v) if d is None else d.strftime("%Y-%m-%d")


WS_PREPROCESS = {"norm_str": _ws_norm_str,
                 "extract_number": _ws_extract_number,
                 "norm_date": _ws_norm_date}


# Per-column deterministic metrics (official names and semantics, 0.0 or 1.0).

def _ws_exact_match(response, target, criterion=None) -> float:
    return 1.0 if str(response).lower() == str(target).lower() else 0.0


_WS_URL_RE = re.compile(
    r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]"
    r"|(?:%[0-9a-fA-F][0-9a-fA-F]))+")


def _ws_url_match(response, target, criterion=None) -> float:
    from urllib.parse import urlparse
    resp = {urlparse(u).netloc for u in _WS_URL_RE.findall(str(response))}
    targ = {urlparse(u).netloc for u in _WS_URL_RE.findall(str(target))}
    return 1.0 if resp == targ else 0.0


def _ws_in_match(response, target, criterion=None) -> float:
    return 1.0 if str(response) in str(target) else 0.0


def _ws_to_number(v):
    s = str(v)
    try:
        return float(s.replace("%", "")) / 100.0 if "%" in s else float(s)
    except (ValueError, TypeError):
        return None


def _ws_number_near(response, target, criterion) -> float:
    r, t = _ws_to_number(response), _ws_to_number(target)
    if r is None or t is None:
        return 1.0 if (r is None and t is None
                       and str(response) == str(target)) else 0.0
    return 1.0 if abs(r - t) <= abs(t) * float(criterion) else 0.0


def _ws_date_near(response, target, criterion=None) -> float:
    r, t = _ws_parse_date(response), _ws_parse_date(target)
    if r is None or t is None:
        return 1.0 if (r is None and t is None) else 0.0
    return 1.0 if abs((r - t).days) <= 31 else 0.0


WS_METRICS = {"exact_match": _ws_exact_match, "url_match": _ws_url_match,
              "in_match": _ws_in_match, "number_near": _ws_number_near,
              "date_near": _ws_date_near}


def _ws_canon_numeric_columns(gold_rows, resp_rows, columns):
    """Numeric harmonization without a dataframe library: when every non-empty
    value of a column on both sides parses as a number, rewrite them all to
    one canonical numeric string, so gold "1" equals response "1.0" exactly as
    dtype coercion made them equal in the reference implementation."""
    def canon(f):
        return str(int(f)) if float(f).is_integer() else repr(float(f))
    for col in columns:
        vals = [(row, row.get(col, "")) for row in gold_rows + resp_rows]
        floats = []
        for _, v in vals:
            if str(v).strip() == "":
                floats.append(None)
                continue
            try:
                floats.append(float(str(v)))
            except (ValueError, TypeError):
                floats = None
                break
        if floats is None:
            continue
        for (row, _), f in zip(vals, floats):
            if f is not None:
                row[col] = canon(f)


def _ws_dedup(rows, key_columns):
    seen, out = set(), []
    for row in rows:
        key = tuple(row.get(c, "") for c in key_columns)
        if key not in seen:
            seen.add(key)
            out.append(row)
    return out


def ws_align_prompt(to_align, reference) -> str:
    return WS_ALIGN_TEMPLATE.format(
        response=json.dumps(list(to_align), ensure_ascii=False),
        reference=json.dumps(list(reference), ensure_ascii=False))


def ws_column_judge_prompt(pairs, criterion, idxs=None) -> str:
    """pairs = [(response, target), ...] over the matched rows of one column.

    `idxs` labels the pairs when they are a SUBSET of a column (the re-ask
    below); omitted, the labels are 0..n-1.
    """
    labels = range(len(pairs)) if idxs is None else idxs
    payload = {f"idx_{i}": {"response": r, "target": t}
               for i, (r, t) in zip(labels, pairs)}
    return WS_COLUMN_JUDGE_TEMPLATE.format(
        criterion=criterion or "", response=json.dumps(
            payload, ensure_ascii=False, indent=1))


# A verdict key judges sometimes use instead of listing every row: "these are
# the failures, everything else passed". Honoring it is reading the judge
# correctly, not grading leniently.
WS_JUDGE_DEFAULT_KEYS = ("idx_others", "others", "default", "idx_rest")
# How many times to re-ask for rows the judge left out. A judge that
# abbreviates a long column does it every time, so one re-ask over just the
# missing rows is usually enough; anything still missing is recorded, never
# silently scored zero.
WS_JUDGE_REASKS = 1


def ws_judge_scores(verdict, n):
    """(scores, unjudged_idxs) for one column's verdict over n matched rows.

    The official evaluator reads `verdict[f"idx_{i}"] == 1` and scores
    everything else zero, which silently zeroes every row a judge omitted. A
    judge that emits a partial list plus a catch-all key can turn a good
    episode into a 0.00 that way. Missing is therefore distinguished from
    failed: an explicit default key applies to unlisted rows, and anything
    still unaccounted for is REPORTED.
    """
    verdict = verdict or {}
    default = next((verdict[k] for k in WS_JUDGE_DEFAULT_KEYS
                    if k in verdict), None)
    scores, unjudged = [], []
    for i in range(n):
        key = f"idx_{i}"
        if key in verdict:
            scores.append(1.0 if verdict[key] == 1 else 0.0)
        elif default is not None:
            scores.append(1.0 if default == 1 else 0.0)
        else:
            scores.append(0.0)
            unjudged.append(i)
    return scores, unjudged


def parse_ws_json(text):
    """Judge output -> dict. The official parser (last ```json block) first,
    falling back to bare-JSON object spans."""
    raw = text or ""
    candidates = re.findall(r"```json\s*(\{.*?\})\s*```", raw, re.DOTALL)[::-1]
    first, last = raw.find("{"), raw.rfind("}")
    if first >= 0 and last > first:
        # The whole span (a prose-wrapped single object), then the tail object
        # (analysis text followed by the final JSON).
        candidates.append(raw[first:last + 1])
        tail = raw.rfind("{", 0, last)
        while tail >= 0:
            candidates.append(raw[tail:last + 1])
            try:
                json.loads(raw[tail:last + 1])
                break  # smallest well-formed tail object found
            except json.JSONDecodeError:
                tail = raw.rfind("{", 0, tail)
    for c in candidates:
        try:
            doc = json.loads(c)
        except json.JSONDecodeError:
            continue
        if isinstance(doc, dict):
            return doc
    return None


def score_ws_tables(gold_rows, resp_rows, spec, column_scores):
    """Final scoring given preprocessed, aligned tables and the per-column 0/1
    scores for the joined rows. `column_scores` maps each non-key required
    column to one score per matched row."""
    required = spec["required"]
    unique = spec["unique_columns"]
    gold_by_key = {tuple(r.get(c, "") for c in unique): r for r in gold_rows}
    resp_keys = [tuple(r.get(c, "") for c in unique) for r in resp_rows]
    matched = [k for k in resp_keys if k in gold_by_key]
    n_matched = len(matched)

    # Strict success rate: same shape and every preprocessed cell equal after
    # sorting both tables by all required columns.
    def table_key(rows):
        return sorted(tuple(str(r.get(c, "")) for c in required)
                      for r in rows)
    sr = 1 if (len(gold_rows) == len(resp_rows) and gold_rows
               and table_key(gold_rows) == table_key(resp_rows)) else 0
    # A row scores the minimum across its cells; key columns count 1.0,
    # because the join already matched them.
    row_scores, tp_item = [], 0.0
    for i in range(n_matched):
        cells = [column_scores[c][i] for c in column_scores]
        row_scores.append(min(cells) if cells else 1.0)
        tp_item += sum(cells) + len(unique)
    tp_row = sum(row_scores)
    n_pred, n_gold = len(resp_rows), len(gold_rows)
    n_cols = len(required)

    def prf(tp, n_pred_units, n_gold_units):
        p = tp / n_pred_units if n_pred_units else 0.0
        r = tp / n_gold_units if n_gold_units else 0.0
        f = 2 * p * r / (p + r) if p + r > 1e-9 else 0.0
        return round(p, 4), round(r, 4), round(f, 4)

    rp, rr, rf = prf(tp_row, n_pred, n_gold)
    ip, ir, if1 = prf(tp_item, n_pred * n_cols, n_gold * n_cols)
    if rp == rr == rf == ip == ir == if1 == 1.0:
        sr = 1
    return {"sr": sr, "row_precision": rp, "row_recall": rr, "row_f1": rf,
            "item_precision": ip, "item_recall": ir, "item_f1": if1,
            "n_pred_rows": n_pred, "n_gold_rows": n_gold,
            "n_matched_rows": n_matched}


# ---------------------------------------------------------------------------
# grader identity


def grader_config(judge_model, grade_kind="single") -> dict:
    """The manifest's grader entry: everything a verdict depends on. The grade
    kind selects the prompt and parser, and therefore a distinct grader id, so
    sidecars for different kinds never collide or cross-invalidate."""
    cfg = {"judge": judge_model.name, "judge_model_id": judge_model.model_id,
           "client": judge_model.client, "base_url": judge_model.base_url,
           "judge_sampling": dict(judge_model.sampling)}
    if grade_kind == "set":
        cfg.update({"grade_kind": "set", "prompt_hash": DSQA_PROMPT_HASH,
                    "parser_version": DSQA_PARSER_VERSION,
                    "template": "deepsearchqa-official"})
    elif grade_kind == "table":
        cfg.update({"grade_kind": "table", "prompt_hash": WS_PROMPT_HASH,
                    "parser_version": WS_PARSER_VERSION,
                    "template": "widesearch-official"})
    else:
        cfg.update({"prompt_hash": PROMPT_HASH,
                    "parser_version": PARSER_VERSION})
    return cfg


def grader_id_from_config(cfg) -> str:
    config_hash = hashlib.sha256(json.dumps(
        cfg, sort_keys=True, default=str).encode()).hexdigest()
    return f"{cfg['judge']}-g{config_hash[:10]}"


def grader_id(judge_model) -> str:
    return grader_id_from_config(grader_config(judge_model))


def digest_record(rollout) -> dict:
    """The per-episode fields a grade's validity depends on."""
    return {"sample_id": rollout.sample_id,
            "episode_id": rollout.meta.get("episode_id"),
            "final_answer": final_answer(rollout),
            "gold": rollout.gold}


def digest_of_records(records) -> str:
    records = sorted(records,
                     key=lambda x: (str(x["sample_id"]), str(x["episode_id"])))
    return hashlib.sha256(json.dumps(
        records, sort_keys=True, ensure_ascii=False,
        default=str).encode()).hexdigest()


def accepted_digest(rollouts) -> str:
    """Fingerprint the exact accepted attempts and answers a grade covers, so
    a later report can tell whether a sidecar still describes them."""
    return digest_of_records([digest_record(r) for r in rollouts])


# ---------------------------------------------------------------------------
# judge cache: one append-only JSONL, keyed on everything a verdict depends
# on. Shared across evaluations, so re-grading only pays for changed answers.


class JudgeCache:
    def __init__(self, path=None):
        self.path = Path(path) if path else CACHE_DIR / "judge.jsonl"
        self._d = {}
        if self.path.exists():
            for rec in io.read_jsonl(self.path):
                self._d[rec["key"]] = rec

    @staticmethod
    def key(prompt, judge_model, grade_kind="single") -> str:
        return hashlib.sha256(json.dumps(
            {"prompt": prompt,
             "grader": grader_config(judge_model, grade_kind)},
            sort_keys=True, default=str).encode()).hexdigest()

    def get(self, key):
        return self._d.get(key)

    def put(self, key, record):
        record = {"key": key, **record}
        self._d[key] = record
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(io.dump_line(record) + "\n")


# ---------------------------------------------------------------------------
# the grade stage


async def _grade_one_single(r, client, judge_model, cache, sem):
    """One episode, single-answer judging, plus the deterministic metric block
    — zero-cost, template-independent, always computed."""
    fa = final_answer(r)
    question = r.meta.get("question", "")
    rec = {"episode_id": r.meta.get("episode_id"), "sample_id": r.sample_id,
           "attempt": r.meta.get("attempt", 0), "final_answer": fa,
           "parser_version": PARSER_VERSION, "em": exact_match(fa, r.gold),
           "cover_em": cover_em(fa, r.gold), "token_f1": token_f1(fa, r.gold)}
    if not fa:
        return {**rec, "label": "NO_ANSWER", "correctness": 0.0,
                "cached": False, "n_calls": 0}
    prompt = judge_prompt_text(question, r.gold, fa)
    key = cache.key(prompt, judge_model)
    hit = cache.get(key)
    if hit is not None and (hit.get("label") == "UNPARSEABLE"
                            or not (hit.get("raw") or "").strip()):
        hit = None  # an unusable cached verdict: retry, do not replay it
    if hit is None:
        try:
            async with sem:
                msg, usage = await client.generate(
                    judge_model.model_id, [user(prompt)], [],
                    judge_model.sampling)
        except ProviderNoChoices as e:
            # A deterministic provider block on this prompt: score the episode
            # zero VISIBLY — the label separates "judge refused" from "wrong
            # answer" in every report — instead of failing the whole benchmark.
            return {**rec, "label": "JUDGE_BLOCKED", "correctness": 0.0,
                    "raw_judge_output": str(e), "cached": False, "n_calls": 1}
        hit = {"label": parse_verdict(msg.get("content")),
               "raw": (msg.get("content") or "").strip(),
               "reasoning": msg.get("reasoning_content"),
               "sample_id": r.sample_id, "usage": usage}
        cache.put(key, hit)
        cached = False
    else:
        cached = True
    return {**rec, "label": hit["label"], "cache_key": key,
            "raw_judge_output": hit.get("raw"),
            "judge_reasoning": hit.get("reasoning"),
            "judge_usage": hit.get("usage") or {},
            "correctness": 1.0 if hit["label"] == "CORRECT" else 0.0,
            "cached": cached, "n_calls": 0 if cached else 1}


async def _grade_one_set(r, client, judge_model, cache, sem):
    """One set-answer episode: whole-response auto-rater -> P/R/F1."""
    fa = final_answer(r)
    question = r.meta.get("question", "")
    gold = r.gold if isinstance(r.gold, dict) else {"answer": r.gold}
    answer_type = gold.get("answer_type")
    kind = "single" if str(answer_type).lower().startswith("single") else "set"
    base = {"episode_id": r.meta.get("episode_id"), "sample_id": r.sample_id,
            "attempt": r.meta.get("attempt", 0), "final_answer": fa,
            "parser_version": DSQA_PARSER_VERSION}
    if not fa:
        return {**base, "precision": 0.0, "recall": 0.0, "f1": 0.0,
                "correctness": 0.0, "em": 0, "label": "NO_ANSWER",
                "answer_type": kind, "n_submitted": 0, "n_gold": None,
                "n_matched_submitted": 0, "n_matched_gold": 0,
                "n_excessive": 0, "cached": False, "n_calls": 0}
    prompt = dsqa_prompt_text(question, gold, fa)
    key = cache.key(prompt, judge_model, grade_kind="set")
    hit = cache.get(key)
    if hit is not None and parse_dsqa_judgment(hit.get("raw")) is None:
        hit = None  # an unparseable cached judgment: retry, do not replay it
    if hit is None:
        try:
            async with sem:
                msg, usage = await client.generate(
                    judge_model.model_id, [user(prompt)], [],
                    judge_model.sampling)
        except ProviderNoChoices as e:
            return {**base, "cache_key": key, "raw_judge_output": str(e),
                    "judge_reasoning": None, "judge_usage": {},
                    "cached": False, "n_calls": 1,
                    "precision": 0.0, "recall": 0.0, "f1": 0.0,
                    "correctness": 0.0, "em": 0, "label": "JUDGE_BLOCKED",
                    "answer_type": kind, "n_submitted": None, "n_gold": None,
                    "n_matched_submitted": 0, "n_matched_gold": 0,
                    "n_excessive": None}
        hit = {"raw": (msg.get("content") or "").strip(),
               "reasoning": msg.get("reasoning_content"),
               "sample_id": r.sample_id, "usage": usage}
        cache.put(key, hit)
        cached = False
    else:
        cached = True
    judgment = parse_dsqa_judgment(hit.get("raw"))
    common = {**base, "cache_key": key, "raw_judge_output": hit.get("raw"),
              "judge_reasoning": hit.get("reasoning"),
              "judge_usage": hit.get("usage") or {}, "cached": cached,
              "n_calls": 0 if cached else 1}
    if judgment is None:
        return {**common, "precision": 0.0, "recall": 0.0, "f1": 0.0,
                "correctness": 0.0, "em": 0, "label": "UNPARSEABLE",
                "answer_type": kind, "n_submitted": None, "n_gold": None,
                "n_matched_submitted": 0, "n_matched_gold": 0,
                "n_excessive": None}
    score = score_dsqa_judgment(judgment, answer_type)
    return {**common, **score,
            "correctness_details": judgment["correctness_details"],
            "excessive_answers": judgment["excessive_answers"],
            "judge_explanation": judgment["explanation"]}


async def _ws_judge_call(prompt, step, payload, client, judge_model, cache,
                         sem, steps, usages):
    """One cached LLM step of the table grader.

    Each call appends an audit entry to `steps`. Unlike the single and set
    graders, a table-grader prompt is NOT reconstructable from (question,
    gold, final answer) — it depends on the parsed table and on earlier
    alignments — so the step's input payload and raw output are recorded in
    the grade record instead. Returns (parsed_json_or_None, n_uncached_calls).
    """
    key = cache.key(prompt, judge_model, grade_kind="table")
    hit = cache.get(key)
    if hit is not None and parse_ws_json(hit.get("raw")) is None:
        hit = None
    if hit is None:
        try:
            async with sem:
                msg, usage = await client.generate(
                    judge_model.model_id, [user(prompt)], [],
                    judge_model.sampling)
        except ProviderNoChoices as e:
            # A blocked judge step: audit it and return no-parse. The existing
            # None-verdict handling scores the affected comparison zero
            # without killing the benchmark.
            usages.append({})
            steps.append({"step": step, "input": payload, "cache_key": key,
                          "raw": f"JUDGE_BLOCKED: {e}", "parsed_ok": False,
                          "cached": False})
            return None, 1
        hit = {"raw": (msg.get("content") or "").strip(),
               "reasoning": msg.get("reasoning_content"), "usage": usage}
        cache.put(key, hit)
        calls = 1
    else:
        calls = 0
    usages.append(hit.get("usage") or {})
    parsed = parse_ws_json(hit.get("raw"))
    steps.append({"step": step, "input": payload, "cache_key": key,
                  "raw": hit.get("raw"), "parsed_ok": parsed is not None,
                  "cached": calls == 0})
    return parsed, calls


_WS_ZERO = {"sr": 0, "row_precision": 0.0, "row_recall": 0.0, "row_f1": 0.0,
            "item_precision": 0.0, "item_recall": 0.0, "item_f1": 0.0,
            "n_pred_rows": 0, "n_gold_rows": None, "n_matched_rows": 0}


# The zero-score paths where the answer never reached scoring at all: the
# episode failed to produce a table the grader could map onto the required
# columns. Kept distinct from a real table whose rows all missed, so reports
# can separate "could not format" from "researched badly".
TABLE_FORMAT_FAILS = frozenset({"NO_ANSWER", "NO_TABLE", "COLUMN_MISMATCH"})


async def _grade_one_table(r, client, judge_model, cache, sem):
    """One table episode: the official evaluation pipeline. The row's gold
    carries the answer table and the per-column spec, so grading needs no
    dataset access — only judge calls."""
    fa = final_answer(r)
    gold = r.gold if isinstance(r.gold, dict) else {}
    spec = gold.get("evaluation") or {}
    required = list(spec.get("required") or [])
    unique = list(spec.get("unique_columns") or [])
    pipeline = spec.get("eval_pipeline") or {}
    base = {"episode_id": r.meta.get("episode_id"), "sample_id": r.sample_id,
            "attempt": r.meta.get("attempt", 0), "final_answer": fa,
            "parser_version": WS_PARSER_VERSION}
    zero = {**_WS_ZERO, "n_gold_rows": len(gold.get("table") or [])}

    steps = []

    def failed(label, n_calls=0, usages=()):
        return {**base, **zero, "correctness": 0.0, "em": 0, "label": label,
                "cached": bool(steps) and n_calls == 0, "n_calls": n_calls,
                "judge_steps": steps, "judge_usage": _ws_merge_usage(usages)}

    if not fa:
        return failed("NO_ANSWER")
    parsed = extract_markdown_table(fa)
    if parsed is None:
        return failed("NO_TABLE")
    header, data = parsed
    resp_rows = [dict(zip(header, row)) for row in data]
    gold_rows = [{k: str(v) for k, v in row.items()}
                 for row in (gold.get("table") or [])]
    usages, n_calls = [], 0

    # Column mapping: response headers must become exactly the required set.
    if set(header) != set(required):
        mapping, calls = await _ws_judge_call(
            ws_align_prompt(header, required), "column_map",
            {"response_columns": header, "required": required},
            client, judge_model, cache, sem, steps, usages)
        n_calls += calls
        mapping = {k: v for k, v in (mapping or {}).items()
                   if isinstance(k, str) and isinstance(v, str)}
        resp_rows = [{mapping.get(c, c): v for c, v in row.items()}
                     for row in resp_rows]
        if {c for row in resp_rows for c in row} != set(required) \
                or not resp_rows:
            return failed("COLUMN_MISMATCH", n_calls, usages)
    resp_rows = [{c: row.get(c, "") for c in required} for row in resp_rows]

    _ws_canon_numeric_columns(gold_rows, resp_rows, required)
    resp_rows = _ws_dedup(resp_rows, unique)
    gold_rows = _ws_dedup(gold_rows, unique)

    # Semantic entity-key alignment for judge/exact-match key columns.
    for col in unique if resp_rows and gold_rows else ():
        metrics = (pipeline.get(col) or {}).get("metric") or []
        if "llm_judge" not in metrics and "exact_match" not in metrics:
            continue
        key_map, calls = await _ws_judge_call(
            ws_align_prompt([row[col] for row in resp_rows],
                            [row[col] for row in gold_rows]),
            f"key_align:{col}",
            {"to_align": [row[col] for row in resp_rows],
             "reference": [row[col] for row in gold_rows]},
            client, judge_model, cache, sem, steps, usages)
        n_calls += calls
        for row in resp_rows:
            mapped = (key_map or {}).get(row[col])
            if isinstance(mapped, str):
                row[col] = mapped

    # Per-column preprocess, both sides.
    for col, item in pipeline.items():
        for name in item.get("preprocess") or []:
            fn = WS_PREPROCESS[name]
            for row in resp_rows + gold_rows:
                if col in row:
                    row[col] = fn(row[col])

    # Join on key columns, then score each non-key column over matched rows.
    gold_by_key = {tuple(row.get(c, "") for c in unique): row
                   for row in gold_rows}
    matched = [(row, gold_by_key[k]) for row in resp_rows
               if (k := tuple(row.get(c, "") for c in unique)) in gold_by_key]
    column_scores, n_unjudged = {}, {}
    for col in required:
        if col in unique:
            continue
        item = pipeline.get(col) or {}
        criterion = item.get("criterion")
        scores = [1.0] * len(matched)
        for name in item.get("metric") or []:
            if name != "llm_judge":
                col_scores = [WS_METRICS[name](resp[col], gld[col], criterion)
                              for resp, gld in matched]
            elif matched:
                pairs = [(resp[col], gld[col]) for resp, gld in matched]
                verdict, calls = await _ws_judge_call(
                    ws_column_judge_prompt(pairs, criterion),
                    f"column_judge:{col}",
                    {"pairs": pairs, "criterion": criterion},
                    client, judge_model, cache, sem, steps, usages)
                n_calls += calls
                col_scores, unjudged = ws_judge_scores(verdict, len(matched))
                # Re-ask for the rows the judge left out rather than reading
                # its silence as a zero. The subset keeps the original index
                # labels, so verdicts merge by index. Only a PARTIAL verdict
                # is re-asked: a verdict that scored nothing at all is a
                # flaked judge, which the retry path above already handles,
                # and re-asking there would just double the bill.
                for _ in range(WS_JUDGE_REASKS if len(unjudged) < len(matched)
                               else 0):
                    if not unjudged:
                        break
                    asked = list(unjudged)
                    verdict, calls = await _ws_judge_call(
                        ws_column_judge_prompt([pairs[i] for i in asked],
                                               criterion, idxs=asked),
                        f"column_judge:{col}:reask",
                        {"pairs": [pairs[i] for i in asked],
                         "criterion": criterion, "idxs": asked},
                        client, judge_model, cache, sem, steps, usages)
                    n_calls += calls
                    got, _ = ws_judge_scores(verdict, len(matched))
                    covered = any(k in (verdict or {})
                                  for k in WS_JUDGE_DEFAULT_KEYS)
                    answered = [i for i in asked
                                if covered or f"idx_{i}" in (verdict or {})]
                    for i in answered:
                        col_scores[i] = got[i]
                    unjudged = [i for i in asked if i not in set(answered)]
                if unjudged:
                    n_unjudged[col] = len(unjudged)
            else:
                col_scores = []
            scores = [min(s, c) for s, c in zip(scores, col_scores)]
        column_scores[col] = scores

    score = score_ws_tables(gold_rows, resp_rows, spec, column_scores)
    label = ("PERFECT" if score["sr"] else
             "PARTIAL" if score["row_f1"] > 0 else
             "ZERO_MATCH")
    return {**base, **score, "correctness": score["row_f1"],
            "em": score["sr"], "label": label,
            "cached": n_calls == 0, "n_calls": n_calls,
            # Rows the column judge never scored even after the re-ask. They
            # count as zero for want of a better option, but the record says
            # so: a silent zero here is indistinguishable from a real failure
            # and reads as a model error when it is a judge error.
            **({"n_unjudged": n_unjudged} if n_unjudged else {}),
            "judge_steps": steps, "judge_usage": _ws_merge_usage(usages)}


def _ws_merge_usage(usages):
    """Sum an episode's alignment and column-call usage into one record-shaped
    usage dict, since cost accounting reads a single judge_usage."""
    merged = {}
    for u in usages:
        for k, v in (u or {}).items():
            if isinstance(v, (int, float)):
                merged[k] = merged.get(k, 0) + v
    return merged


async def grade_one(rollout, client, judge_model, cache, sem,
                    grade_kind="single") -> dict:
    """Grade ONE rollout -> a grade record with `correctness` in [0, 1] plus a
    label and per-kind detail. The per-episode entry point; `grade_run` is the
    batch wrapper over it."""
    base_one = {"set": _grade_one_set, "table": _grade_one_table}.get(grade_kind)
    if base_one is None:
        return await _grade_one_single(rollout, client, judge_model, cache, sem)
    return await base_one(rollout, client, judge_model, cache, sem)


async def grade_run(run_dir, judge_model, client=None, cache=None,
                    concurrency=8, grade_kind="single", sem=None):
    """Grade one (benchmark, model) run directory: accepted rollouts -> the
    sidecar `grades/<grader_id>.jsonl`.

    Deterministic given the cache. The sidecar is rewritten whole — it is
    derived data, and rollouts are never touched. Pass a shared `sem` and
    `cache` to cap in-flight judge calls globally when grading many pairs at
    once.
    """
    from .models import get_client
    run_dir = Path(run_dir)
    client = client or get_client(judge_model)
    cache = cache or JudgeCache()
    accepted = accepted_attempts(run_dir / "rollouts.jsonl")
    accepted_rollouts = list(accepted.values())
    sem = sem or asyncio.Semaphore(concurrency)

    grades = await asyncio.gather(
        *(grade_one(r, client, judge_model, cache, sem, grade_kind=grade_kind)
          for r in accepted_rollouts))
    cfg = grader_config(judge_model, grade_kind)
    gid = grader_id_from_config(cfg)
    out = run_dir / "grades" / f"{gid}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for g in sorted(grades, key=lambda g: str(g["sample_id"])):
            f.write(io.dump_line(g) + "\n")
    grade_manifest = {
        "v": 1,
        "grader_id": gid,
        "grade_kind": grade_kind,
        "grader": cfg,
        "accepted_digest": accepted_digest(accepted_rollouts),
        "episode_ids": sorted(str(r.meta.get("episode_id"))
                              for r in accepted_rollouts),
        "n": len(grades),
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    manifest_path = out.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(grade_manifest, indent=2, default=str) + "\n")
    return {"path": str(out), "grader_id": gid, "n": len(grades),
            "manifest_path": str(manifest_path),
            "n_judge_calls": sum(g.get("n_calls", 0) for g in grades)}
