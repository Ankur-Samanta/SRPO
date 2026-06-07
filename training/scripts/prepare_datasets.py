"""Prepare 400/100 train/test parquet splits for VERL training.

Supports: math500, gpqa, csqa, mathqa, numinamath_olympiads, numinamath_aops,
          numinamath_amc, openmath2.

Uses TREE's dataset_loaders for gpqa/csqa (pulls from HF), local JSON for
math500/mathqa, and HuggingFace datasets for NuminaMath/OpenMath2.

Usage:
    python training/scripts/prepare_datasets.py [--datasets math500 gpqa csqa mathqa] [--n-train 400] [--n-test 100] [--seed 42]
    python training/scripts/prepare_datasets.py --datasets numinamath_olympiads numinamath_aops numinamath_amc openmath2
    python training/scripts/prepare_datasets.py --datasets all
"""

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

# dataset_loaders is vendored under SRPO/vendor/ (see vendor/README.md).
# DATA_DIR is an optional local JSON cache for non-paper datasets, read only when
# the file exists; absent that cache, loaders fall through to HuggingFace.
TREE_DIR = Path(__file__).parent.parent.parent / "vendor"
DATA_DIR = TREE_DIR / "data"
OUTPUT_BASE = Path.home() / "data" / "rlhf"

ALL_DATASETS = [
    "math500", "gpqa", "csqa", "mathqa",
    "numinamath_olympiads", "numinamath_aops", "numinamath_amc", "openmath2",
    "polaris",
    "mmlu_pro", "ifeval", "sciknoweval_l3",
    "sciknoweval_chemistry", "sciknoweval_physics",
    "sciknoweval_biology", "sciknoweval_materials",
    "livecodebench", "livecodebench_medium", "livecodebench_hard",
    "retrosynthesis_uspto50k",
]


def extract_boxed_answer(text: str) -> str:
    """Extract the last \\boxed{...} answer from a string."""
    matches = list(re.finditer(r"\\boxed\{", text))
    if not matches:
        return ""
    start_pos = matches[-1].end()
    brace_count = 1
    i = start_pos
    while i < len(text) and brace_count > 0:
        if text[i] == "{":
            brace_count += 1
        elif text[i] == "}":
            brace_count -= 1
        i += 1
    if brace_count == 0:
        return text[start_pos : i - 1]
    return ""


def load_problems(dataset_name: str) -> list[dict]:
    """Load problems as list of dicts with 'problem' and 'answer' keys."""

    # --- Local JSON (math500, mathqa) ---
    local_json = DATA_DIR / f"{dataset_name}.json"
    if local_json.exists():
        with open(local_json) as f:
            return json.load(f)

    # --- NuminaMath (filter by source) ---
    numina_sources = {
        "numinamath_olympiads": "olympiads",
        "numinamath_aops": "aops_forum",
        "numinamath_amc": "synthetic_amc",
    }
    if dataset_name in numina_sources:
        return _load_numinamath(numina_sources[dataset_name])

    # --- OpenMath2 ---
    if dataset_name == "openmath2":
        return _load_openmath2()

    # --- Polaris (handled separately due to difficulty filtering) ---
    if dataset_name == "polaris":
        raise ValueError("Use --datasets polaris --polaris-levels 6 7 to load Polaris")

    # --- SDPO paper datasets ---
    if dataset_name == "mmlu_pro":
        return _load_mmlu_pro()
    if dataset_name == "ifeval":
        return _load_ifeval()
    if dataset_name == "sciknoweval_l3":
        return _load_sciknoweval_l3()
    _SCIKNOWEVAL_DOMAINS = {
        "sciknoweval_chemistry": "Chemistry",
        "sciknoweval_physics": "Physics",
        "sciknoweval_biology": "Biology",
        "sciknoweval_materials": "Material",  # HF uses "Material" not "Materials"
    }
    if dataset_name in _SCIKNOWEVAL_DOMAINS:
        return _load_sciknoweval_l3(domain=_SCIKNOWEVAL_DOMAINS[dataset_name])
    if dataset_name == "livecodebench":
        return _load_livecodebench()
    if dataset_name == "livecodebench_medium":
        return _load_livecodebench(difficulties=("medium",))
    if dataset_name == "livecodebench_hard":
        return _load_livecodebench(difficulties=("hard",))
    if dataset_name == "retrosynthesis_uspto50k":
        return _load_retrosynthesis_uspto50k()

    # --- Fall back to TREE loaders (gpqa, csqa) ---
    sys.path.insert(0, str(TREE_DIR))
    from dataset_loaders import load_dataset_by_name
    return load_dataset_by_name(dataset_name)


def _load_mmlu_pro() -> list[dict]:
    """Load MMLU-Pro 10-choice MCQ from TIGER-Lab/MMLU-Pro."""
    from datasets import load_dataset

    print("  Loading TIGER-Lab/MMLU-Pro...")
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")

    problems = []
    for row in ds:
        # Format options as lettered list
        options = row["options"]
        option_letters = "ABCDEFGHIJ"
        option_lines = []
        for i, opt in enumerate(options):
            if i < len(option_letters):
                option_lines.append(f"{option_letters[i]}. {opt}")
        options_text = "\n".join(option_lines)
        problem_text = f"{row['question']}\n\n{options_text}"

        problems.append({
            "problem": problem_text,
            "answer": row["answer"],
            "unique_id": f"mmlu_pro_{row['question_id']}",
            "subject": row.get("category", ""),
            "level": "",
            "options": options,
            "category": row.get("category", ""),
        })

    print(f"  {len(problems)} problems loaded")
    return problems


def _load_ifeval() -> list[dict]:
    """Load IFEval instruction-following prompts from google/IFEval.

    The HF repo stores data as a JSONL file, not parquet, so we download
    and parse it directly.
    """
    from huggingface_hub import hf_hub_download

    print("  Loading google/IFEval...")
    path = hf_hub_download(
        repo_id="google/IFEval",
        filename="ifeval_input_data.jsonl",
        repo_type="dataset",
    )

    problems = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            problems.append({
                "problem": row["prompt"],
                "answer": "",  # No single answer; scoring is rule-based
                "unique_id": f"ifeval_{row['key']}",
                "subject": "instruction_following",
                "level": "",
                "instruction_id_list": row["instruction_id_list"],
                "kwargs": row["kwargs"],
            })

    print(f"  {len(problems)} problems loaded")
    return problems


def _load_sciknoweval_l3(domain: str = None) -> list[dict]:
    """Load SciKnowEval L3 (knowledge reasoning) from hicai-zju/SciKnowEval.

    Args:
        domain: If set, filter to a specific domain (Chemistry, Physics,
                Biology, Material). None loads all L3 problems.
    """
    from datasets import load_dataset

    label = f"L3/{domain}" if domain else "L3"
    print(f"  Loading hicai-zju/SciKnowEval v2, filtering to {label}...")
    ds = load_dataset("hicai-zju/SciKnowEval", "v2", split="test")

    problems = []
    for row in ds:
        details = row.get("details", {})
        if details.get("level") != "L3":
            continue
        if domain and row.get("domain") != domain:
            continue

        qtype = row.get("type", "")
        answer_key = row.get("answerKey", "")
        answer = row.get("answer", "")

        # For MCQ, build options text
        problem_text = row["question"]
        choices = row.get("choices", {})
        if "mcq" in qtype.lower() and choices:
            labels = choices.get("label", [])
            texts = choices.get("text", [])
            option_lines = [f"{l}. {t}" for l, t in zip(labels, texts)]
            if option_lines:
                problem_text = f"{problem_text}\n\n" + "\n".join(option_lines)

        # Use answerKey for MCQ, answer for open-ended
        gt = answer_key if ("mcq" in qtype.lower() and answer_key) else answer

        problems.append({
            "problem": problem_text,
            "answer": gt,
            "unique_id": f"sciknoweval_l3_{len(problems)}",
            "subject": row.get("domain", ""),
            "level": "L3",
            "type": qtype,
            "answerKey": answer_key,
            "domain": row.get("domain", ""),
        })

    print(f"  {len(problems)} L3 problems loaded")
    return problems


def _decode_lcb_test_cases(raw: str) -> list[dict]:
    """Decode LiveCodeBench test cases.

    Public tests are plain JSON. Private tests are base64 → zlib → pickle → JSON.
    """
    if not raw:
        return []
    # Try plain JSON first (public test cases)
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass
    # Try base64 → zlib → pickle → JSON (private test cases)
    try:
        import base64, pickle, zlib
        decoded = pickle.loads(zlib.decompress(base64.b64decode(raw)))
        parsed = json.loads(decoded)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    return []


def _load_livecodebench(
    seed: int = 42,
    difficulties: tuple = ("medium", "hard"),
) -> list[dict]:
    """Load LiveCodeBench code generation problems.

    Following SDPO protocol: randomly split private test cases 50/50 into
    train_tests (reward signal during training) and eval_tests (held-out
    for evaluation).

    Args:
        seed: Random seed for test case splitting.
        difficulties: Tuple of difficulty levels to include.

    Private tests are compressed (base64 → zlib → pickle → JSON).
    """
    import random
    from datasets import load_dataset

    rng = random.Random(seed)

    diff_label = "+".join(difficulties)
    print(f"  Loading livecodebench/code_generation_lite ({diff_label})...")
    ds = load_dataset("livecodebench/code_generation_lite",
                      version_tag="release_v6", split="test",
                      trust_remote_code=True)

    n_skipped = 0
    problems = []
    for row in ds:
        if row.get("difficulty") not in difficulties:
            n_skipped += 1
            continue

        private_tests = _decode_lcb_test_cases(row.get("private_test_cases", ""))

        # 50/50 split of private tests (SDPO protocol)
        shuffled = list(private_tests)
        rng.shuffle(shuffled)
        mid = len(shuffled) // 2
        train_tests = shuffled[:mid]
        eval_tests = shuffled[mid:]

        starter_code = row.get("starter_code", "")
        problem_text = row.get("question_content", "")
        if starter_code:
            problem_text += f"\n\nStarter code:\n```python\n{starter_code}\n```"

        problems.append({
            "problem": problem_text,
            "answer": "",  # No single answer; scoring via test execution
            "unique_id": f"lcb_{row.get('question_id', len(problems))}",
            "subject": row.get("platform", ""),
            "level": row.get("difficulty", ""),
            "train_tests": json.dumps(train_tests),
            "eval_tests": json.dumps(eval_tests),
            "test_cases": json.dumps(train_tests),  # default: train reward uses train_tests
            "metadata": row.get("metadata", "{}"),
        })

    print(f"  {len(problems)} {diff_label} problems loaded ({n_skipped} skipped)")
    return problems


_RETRO_PROMPT_TEMPLATE = (
    "You are an expert chemist solving a single-step retrosynthesis problem. "
    "Given a target product molecule (SMILES), propose the reactants for one "
    "retrosynthetic disconnection.\n\n"
    "Reason step by step about the product structure, key functional groups, "
    "and likely bond disconnections. Then output the reactants as a single "
    "SMILES string with components separated by '.'.\n\n"
    "Put your final answer in \\boxed{{...}}.\n\n"
    "Product: {product}\n"
)


def _load_retrosynthesis_uspto50k(seed: int = 42) -> list[dict]:
    """Load USPTO-50K single-step retrosynthesis problems.

    For each row: product SMILES is the prompt input; gold reactants are
    extracted from the atom-mapped rxn_smiles (left side of '>>') with atom
    maps stripped and canonicalized via RDKit.

    Train and validation splits are merged into one list (the standard
    to_verl_parquet path samples train/test from a shuffled pool).
    """
    from datasets import concatenate_datasets, load_dataset
    from rdkit import Chem

    print("  Loading pingzhili/uspto-50k...")
    ds_all = load_dataset("pingzhili/uspto-50k")
    ds = concatenate_datasets([ds_all["train"], ds_all["validation"]])

    def strip_amap(smi):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        return Chem.MolToSmiles(mol)

    def canon(smi):
        mol = Chem.MolFromSmiles(smi)
        return Chem.MolToSmiles(mol) if mol else None

    problems = []
    n_skipped = 0
    for i, row in enumerate(ds):
        rxn = row.get("rxn_smiles", "")
        prod_raw = row.get("prod_smiles", "")
        if ">>" not in rxn:
            n_skipped += 1
            continue
        reactants = strip_amap(rxn.split(">>")[0])
        product = canon(prod_raw)
        if reactants is None or product is None:
            n_skipped += 1
            continue
        problems.append({
            "problem": _RETRO_PROMPT_TEMPLATE.format(product=product),
            "answer": reactants,
            "product_smiles": product,
            "unique_id": row.get("id", f"uspto50k_{i}"),
            "subject": "retrosynthesis",
            "level": str(row.get("class", "")),
        })

    print(f"  {len(problems)} problems loaded ({n_skipped} skipped)")
    return problems


def _clean_amc_answer(answer: str) -> str:
    """Normalize AMC-style boxed answers to bare value.

    Handles: \\textbf{(D)}\\ 7 -> 7, (C) -> C, \\text{B} -> B,
             C) 23 -> 23, bare '6' -> 6, bare 'B' -> B
    """
    # Strip \\( \\) wrappers
    answer = re.sub(r'^\\\(|\\\)$', '', answer).strip()
    # \textbf{(X)} VALUE or \text{(X)} VALUE -> VALUE
    m = re.match(r'\\text(?:bf)?\{?\(?([A-E])\)?\}?[\s\\:]*(.+)', answer)
    if m:
        value = m.group(2).strip()
        return value if value else m.group(1)
    # (X) or X) with trailing value
    m = re.match(r'\(?([A-E])\)?[\s)]*(.+)', answer)
    if m:
        value = m.group(2).strip()
        return value if value else m.group(1)
    # \text{X} -> X
    m = re.match(r'\\text\{([A-E])\}', answer)
    if m:
        return m.group(1)
    # Already clean
    return answer.strip()


def _is_verifiable_answer(answer: str) -> bool:
    """Whitelist: accept only answers with reliably verifiable rewards.

    Five principled rules:
    1. <=60 chars (reject complex expressions)
    2. Balanced {} and () (reject broken extraction)
    3. No \\text/\\textbf/\\mathrm wrappers (reject formatting noise)
    4. No English words >=3 chars after stripping LaTeX commands (reject prose/proofs)
    5. Must contain a digit or be a single MC letter A-E (reject symbolic relations)

    Plus minor cleanup: leading punctuation, percentages, units, markdown artifacts.
    """
    if not answer or len(answer) > 60:
        return False
    # Balanced braces
    if answer.count('{') != answer.count('}'):
        return False
    if answer.count('(') != answer.count(')'):
        return False
    # Single MC letter
    if re.match(r'^[A-E]$', answer):
        return True
    # Strip LaTeX commands (\frac, \sqrt, \pi, \circ, \cdot, etc.)
    stripped = re.sub(r'\\[a-zA-Z]+', '', answer)
    # After stripping, reject if any English word >= 3 chars remains
    if re.search(r'[a-zA-Z]{3,}', stripped):
        return False
    # Must contain at least one digit
    if not re.search(r'\d', answer):
        return False
    # No formatting artifacts
    if re.search(r'\\text|\\mathrm|\\textbf', answer):
        return False
    # No leading junk from broken extraction
    if re.match(r'^[};.,;:~\-\*]\s?', answer):
        return False
    # Percentages, units, markdown, tildes
    if re.search(r'%|~|\*\*|\bor\b', answer):
        return False
    if re.search(r'\b(sq|cm|yd|km|kg|lb|oz|ft|mi)\b|cm[²³]|m[²³]', answer):
        return False
    return True


def _fix_answer(answer: str) -> str:
    """Fix common LaTeX issues in extracted answers."""
    # Missing backslash on frac, sqrt, log, ln, pi, circ
    answer = re.sub(r'^(frac|sqrt|log|ln)\{', r'\\\1{', answer)
    answer = re.sub(r'(?<=\s)(frac|sqrt)\{', r'\\\1{', answer)
    return answer


def _load_numinamath(source_filter: str, max_scan: int = 0) -> list[dict]:
    """Load NuminaMath-CoT problems filtered by source, keeping only verifiable answers."""
    from datasets import load_dataset

    print(f"  Streaming AI-MO/NuminaMath-CoT (source={source_filter})...")
    ds = load_dataset("AI-MO/NuminaMath-CoT", split="train", streaming=True)

    problems = []
    n_seen = 0
    n_no_boxed = 0
    n_not_verifiable = 0
    for row in ds:
        if row["source"] != source_filter:
            continue
        n_seen += 1
        if max_scan and n_seen > max_scan:
            break

        answer = extract_boxed_answer(row["solution"])
        if not answer:
            n_no_boxed += 1
            continue

        # Clean AMC-style formatted answers
        if source_filter == "synthetic_amc":
            answer = _clean_amc_answer(answer)

        # Fix common LaTeX issues
        answer = _fix_answer(answer)

        if not _is_verifiable_answer(answer):
            n_not_verifiable += 1
            continue

        problems.append({
            "problem": row["problem"],
            "answer": answer,
            "unique_id": f"numina_{source_filter}_{len(problems)}",
            "subject": source_filter,
            "level": "",
        })

    print(f"  Scanned {n_seen} {source_filter} rows, {len(problems)} verifiable, dropped {n_no_boxed} (no boxed) + {n_not_verifiable} (not verifiable)")
    return problems


def _load_openmath2(max_scan: int = 200000) -> list[dict]:
    """Load OpenMathInstruct-2 problems with expected_answer field."""
    from datasets import load_dataset

    print(f"  Streaming nvidia/OpenMathInstruct-2 (max_scan={max_scan})...")
    ds = load_dataset("nvidia/OpenMathInstruct-2", split="train", streaming=True)

    problems = []
    n_seen = 0
    for row in ds:
        n_seen += 1
        if max_scan and n_seen > max_scan:
            break

        answer = row.get("expected_answer", "")
        if not answer:
            continue

        problems.append({
            "problem": row["problem"],
            "answer": str(answer),
            "unique_id": f"openmath2_{len(problems)}",
            "subject": row.get("problem_source", ""),
            "level": "",
        })

    print(f"  Scanned {n_seen} rows, {len(problems)} with answers")
    return problems


def _load_polaris(levels: list[int]) -> list[dict]:
    """Load Polaris-Dataset-53K filtered by difficulty levels.

    Args:
        levels: List of difficulty levels (0-7). E.g. [6, 7] loads '6/8' and '7/8'.
    """
    from datasets import load_dataset

    level_strs = {f"{l}/8" for l in levels}
    print(f"  Loading POLARIS-Project/Polaris-Dataset-53K (levels={sorted(levels)})...")
    ds = load_dataset("POLARIS-Project/Polaris-Dataset-53K", split="train")

    problems = []
    n_skipped_level = 0
    n_not_verifiable = 0
    for row in ds:
        if row["difficulty"] not in level_strs:
            n_skipped_level += 1
            continue

        answer = str(row["answer"])
        answer = _fix_answer(answer)

        if not _is_verifiable_answer(answer):
            n_not_verifiable += 1
            continue

        problems.append({
            "problem": row["problem"],
            "answer": answer,
            "unique_id": f"polaris_{len(problems)}",
            "subject": "",
            "level": row["difficulty"],
        })

    print(f"  {len(ds)} total, {n_skipped_level} wrong level, {n_not_verifiable} not verifiable, {len(problems)} kept")
    return problems


def to_verl_parquet(
    problems: list[dict],
    dataset_name: str,
    n_train: int,
    n_test: int,
    seed: int,
):
    """Convert problems to verl format and write train/test parquets."""
    test_only = (n_train == 0)
    total_needed = n_train + n_test
    if total_needed > len(problems):
        if test_only:
            # Test-only: cap test size to available data
            n_test = min(n_test, len(problems))
            print(f"  NOTE: only {len(problems)} available, using {n_test} test")
        else:
            # Reduce train size to fit, keep test size
            n_train = len(problems) - n_test
            if n_train <= 0:
                raise ValueError(
                    f"{dataset_name}: only {len(problems)} available, can't even fill test={n_test}"
                )
            print(f"  NOTE: only {len(problems)} available, using {n_train} train / {n_test} test")

    # For MC datasets, normalize answer to uppercase to match model output
    mc_datasets = {"gpqa", "csqa", "mathqa", "mmlu_pro", "sciknoweval_l3",
                    "sciknoweval_chemistry", "sciknoweval_physics",
                    "sciknoweval_biology", "sciknoweval_materials"}
    is_mc = dataset_name in mc_datasets

    # Fields that go into extra_info beyond the standard ones
    _EXTRA_FIELDS = {
        "instruction_id_list", "kwargs",  # IFEval
        "test_cases", "train_tests", "eval_tests", "metadata",  # LiveCodeBench
        "type", "answerKey", "domain",    # SciKnowEval
        "options", "category",            # MMLU-Pro
        "product_smiles",                 # Retrosynthesis (target product for retro_score)
    }

    rows = []
    for i, p in enumerate(problems):
        answer = p["answer"].upper() if is_mc else p["answer"]
        extra = {
            "unique_id": p.get("unique_id", f"{dataset_name}_{i}"),
            "subject": p.get("subject", ""),
            "level": p.get("level", ""),
            "index": i,
        }
        # Pass through dataset-specific fields
        for field in _EXTRA_FIELDS:
            if field in p:
                extra[field] = p[field]

        rows.append({
            "prompt": [{"role": "user", "content": p["problem"]}],
            "data_source": dataset_name,
            "reward_model": {"ground_truth": answer},
            "extra_info": extra,
        })

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    out_dir = OUTPUT_BASE / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if test_only:
        test_df = df.iloc[:n_test]
        test_path = out_dir / "test.parquet"
        test_df.to_parquet(test_path)
        print(f"{dataset_name}: test={len(test_df)} -> {test_path}")
    else:
        train_df = df.iloc[:n_train]
        test_df = df.iloc[n_train : n_train + n_test]
        train_path = out_dir / "train.parquet"
        test_path = out_dir / "test.parquet"
        train_df.to_parquet(train_path)
        test_df.to_parquet(test_path)
        print(f"{dataset_name}: train={len(train_df)} -> {train_path}")
        print(f"{' ' * len(dataset_name)}  test={len(test_df)}  -> {test_path}")


def to_verl_parquet_lcb(problems: list[dict], dataset_name: str, seed: int):
    """Write LCB train/test parquets with test-case-level split.

    All problems appear in both train and test. Test cases are stored
    separately in JSON files keyed by problem ID, not in the parquet.
    The parquet extra_info just contains the problem ID for lookup.

    Layout:
        ~/data/rlhf/<dataset>/train.parquet
        ~/data/rlhf/<dataset>/test.parquet
        ~/data/rlhf/<dataset>/train_tests.json   {id: [test_cases...]}
        ~/data/rlhf/<dataset>/eval_tests.json    {id: [test_cases...]}
    """
    out_dir = OUTPUT_BASE / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write evaluation sample files keyed by unique_id.
    # Each entry is an LCB-format evaluation sample dict:
    #   {"input_output": json.dumps({"inputs": [...], "outputs": [...], "fn_name": ...})}
    # This matches CodeGenerationProblem.get_evaluation_sample() exactly.
    train_samples_map = {}
    eval_samples_map = {}
    for p in problems:
        uid = p.get("unique_id", "")
        meta = p.get("metadata", "{}")
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except json.JSONDecodeError:
                meta = {}
        fn_name = meta.get("func_name", None) if isinstance(meta, dict) else None

        train_tcs = json.loads(p["train_tests"])
        eval_tcs = json.loads(p["eval_tests"])

        train_samples_map[uid] = {
            "input_output": json.dumps({
                "inputs": [tc["input"] for tc in train_tcs],
                "outputs": [tc["output"] for tc in train_tcs],
                "fn_name": fn_name,
            }),
        }
        eval_samples_map[uid] = {
            "input_output": json.dumps({
                "inputs": [tc["input"] for tc in eval_tcs],
                "outputs": [tc["output"] for tc in eval_tcs],
                "fn_name": fn_name,
            }),
        }

    train_samples_path = out_dir / "train_samples.json"
    eval_samples_path = out_dir / "eval_samples.json"
    with open(train_samples_path, "w") as f:
        json.dump(train_samples_map, f)
    with open(eval_samples_path, "w") as f:
        json.dump(eval_samples_map, f)

    # Build parquet rows (no test cases, just IDs)
    rows = []
    for i, p in enumerate(problems):
        rows.append({
            "prompt": [{"role": "user", "content": p["problem"]}],
            "data_source": dataset_name,
            "reward_model": {"ground_truth": p["answer"]},
            "extra_info": {
                "unique_id": p.get("unique_id", f"{dataset_name}_{i}"),
                "subject": p.get("subject", ""),
                "level": p.get("level", ""),
                "index": i,
                "metadata": p.get("metadata", "{}"),
            },
        })

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Same problems in both train and test parquets
    train_path = out_dir / "train.parquet"
    test_path = out_dir / "test.parquet"
    df.to_parquet(train_path)
    df.to_parquet(test_path)

    tt_mb = train_samples_path.stat().st_size / 1e6
    et_mb = eval_samples_path.stat().st_size / 1e6
    print(f"{dataset_name}: train={len(df)} -> {train_path}")
    print(f"{' ' * len(dataset_name)}  test={len(df)}  -> {test_path}")
    print(f"{' ' * len(dataset_name)}  train_samples.json ({tt_mb:.1f} MB)")
    print(f"{' ' * len(dataset_name)}  eval_samples.json  ({et_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["math500", "gpqa", "csqa", "mathqa"],
    )
    parser.add_argument("--n-train", type=int, default=400)
    parser.add_argument("--n-test", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--polaris-levels",
        nargs="+",
        type=int,
        help="Difficulty levels for Polaris (0-7). E.g. --polaris-levels 6 7",
    )
    args = parser.parse_args()

    if args.datasets == ["all"]:
        args.datasets = ALL_DATASETS

    # Test-only datasets (evaluation benchmarks, no train split)
    TEST_ONLY_DATASETS = {
        "sciknoweval_chemistry", "sciknoweval_physics",
        "sciknoweval_biology", "sciknoweval_materials",
    }
    TEST_ONLY_N_TEST = 500

    # LCB datasets: all problems in both splits, test-case-level split
    LCB_DATASETS = {"livecodebench", "livecodebench_medium", "livecodebench_hard"}

    for ds in args.datasets:
        if ds == "polaris":
            if not args.polaris_levels:
                parser.error("--polaris-levels required when using polaris dataset")
            levels = args.polaris_levels
            level_tag = "_".join(str(l) for l in sorted(levels))
            ds_name = f"polaris_d{level_tag}"
            print(f"\nLoading {ds_name}...")
            problems = _load_polaris(levels)
            print(f"  {len(problems)} problems loaded")
            to_verl_parquet(problems, ds_name, args.n_train, args.n_test, args.seed)
        elif ds in TEST_ONLY_DATASETS:
            print(f"\nLoading {ds} (test-only)...")
            problems = load_problems(ds)
            print(f"  {len(problems)} problems loaded")
            to_verl_parquet(problems, ds, 0, TEST_ONLY_N_TEST, args.seed)
        elif ds in LCB_DATASETS:
            print(f"\nLoading {ds} (test-case split)...")
            problems = load_problems(ds)
            print(f"  {len(problems)} problems loaded")
            to_verl_parquet_lcb(problems, ds, args.seed)
        else:
            print(f"\nLoading {ds}...")
            problems = load_problems(ds)
            print(f"  {len(problems)} problems loaded")
            to_verl_parquet(problems, ds, args.n_train, args.n_test, args.seed)


if __name__ == "__main__":
    main()
