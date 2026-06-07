"""Reward scoring functions for non-math datasets.

Provides scoring for:
- Multiple-choice (MMLU-Pro, GPQA, CSQA, MathQA, AGIEval, SciKnowEval MCQ)
- IFEval (instruction-following rule checks)
- SciKnowEval L3 (MCQ + string match)
- LiveCodeBench (sandboxed code execution)
- Yes/No (StrategyQA)
- QA short-answer (HotpotQA): exact-match after answer extraction
- Python assert tests (HumanEval+, MBPP+)
"""

import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Multiple-Choice Scoring
# =============================================================================

# Regex patterns for extracting MC answer from model output, ordered by priority
_MC_PATTERNS = [
    r"\\boxed\{([A-Ja-j])\}",              # \boxed{A}
    r"[Aa]nswer\s*(?:is|:)\s*\(?([A-Ja-j])\)?",  # "answer is (A)" / "Answer: B"
    r"\b([A-J])\s*[\.\)]\s*$",              # trailing "A." or "A)" at end of line
]


def extract_mc_answer(solution_str: str) -> str:
    """Extract multiple-choice answer letter from model output.

    Tries patterns in priority order: \\boxed{}, "answer is", trailing letter.
    Falls back to the last standalone capital letter A-J.

    Returns:
        Uppercase letter A-J, or empty string if no answer found.
    """
    for pattern in _MC_PATTERNS:
        matches = re.findall(pattern, solution_str, re.MULTILINE | re.IGNORECASE)
        if matches:
            return matches[-1].upper()

    # Fallback: last standalone A-J on a line by itself or after whitespace
    fallback = re.findall(r"(?:^|\s)([A-J])(?:\s|$|\.)", solution_str, re.MULTILINE)
    if fallback:
        return fallback[-1].upper()

    return ""


def mc_score(solution_str: str, ground_truth: str) -> float:
    """Score a multiple-choice response.

    Args:
        solution_str: Full model output.
        ground_truth: Correct answer letter (A-J).

    Returns:
        1.0 if correct, 0.0 otherwise.
    """
    predicted = extract_mc_answer(solution_str)
    return 1.0 if predicted == ground_truth.upper() else 0.0


# =============================================================================
# IFEval Scoring
# =============================================================================

def ifeval_score(solution_str: str, extra_info: Optional[dict] = None) -> float:
    """Score an IFEval response by checking instruction-following constraints.

    Uses lm-evaluation-harness IFEval checkers (strict mode). Returns the
    fraction of instructions satisfied.

    Requires: pip install lm-eval  (optional dependency)

    Args:
        solution_str: Full model output.
        extra_info: Must contain 'instruction_id_list' and 'kwargs'.

    Returns:
        Fraction of instructions followed (0.0 to 1.0).
    """
    if not extra_info:
        logger.warning("ifeval_score called without extra_info, returning 0.0")
        return 0.0

    instruction_id_list = extra_info.get("instruction_id_list", [])
    kwargs_list = extra_info.get("kwargs", [])

    if not instruction_id_list:
        logger.warning("ifeval_score: no instruction_id_list in extra_info")
        return 0.0

    try:
        from lm_eval.tasks.ifeval import instructions_registry
    except ImportError:
        raise ImportError(
            "IFEval scoring requires lm-evaluation-harness. "
            "Install with: pip install lm-eval"
        )

    n_followed = 0
    for idx, instruction_id in enumerate(instruction_id_list):
        instruction_cls = instructions_registry.INSTRUCTION_DICT.get(instruction_id)
        if instruction_cls is None:
            logger.warning(f"Unknown IFEval instruction: {instruction_id}")
            continue

        instruction = instruction_cls(instruction_id)
        kwargs = {k: v for k, v in kwargs_list[idx].items() if v is not None}
        instruction.build_description(**kwargs)
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            prompt = extra_info.get("prompt", "")
            instruction.build_description(prompt=prompt)

        if solution_str.strip() and instruction.check_following(solution_str):
            n_followed += 1

    return n_followed / len(instruction_id_list)


# =============================================================================
# SciKnowEval L3 Scoring
# =============================================================================

def _normalize_string(s: str) -> str:
    """Normalize a string for comparison: lowercase, strip, remove punctuation."""
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def sciknoweval_score(
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict] = None,
) -> float:
    """Score a SciKnowEval L3 response.

    For MCQ types: extract letter and compare.
    For open-ended: normalized string match.

    Args:
        solution_str: Full model output.
        ground_truth: Correct answer (letter for MCQ, string for open-ended).
        extra_info: Should contain 'type' (e.g., 'mcq-4-choices') and 'answerKey'.

    Returns:
        1.0 if correct, 0.0 otherwise.
    """
    qtype = (extra_info or {}).get("type", "")
    answer_key = (extra_info or {}).get("answerKey", "")

    if "mcq" in qtype.lower():
        # MCQ: use answerKey if available, else ground_truth
        gt = answer_key if answer_key else ground_truth
        return mc_score(solution_str, gt)
    else:
        # Open-ended: normalized string match
        if not ground_truth:
            return 0.0
        pred = _normalize_string(solution_str)
        gt = _normalize_string(ground_truth)
        # Check if ground truth appears in the response
        return 1.0 if gt in pred else 0.0


# =============================================================================
# LiveCodeBench Scoring
# =============================================================================

def _extract_code_block(solution_str: str) -> str:
    """Extract code from model output, matching LiveCodeBench's extraction.

    Takes content between the last pair of ``` markers. Returns empty string
    if no code block found (will score 0 on all test cases).

    Reference: lcb_runner/utils/extraction_utils.py::extract_code()
    """
    lines = solution_str.split("\n")
    fence_indices = [i for i, line in enumerate(lines) if "```" in line]
    if len(fence_indices) < 2:
        return ""
    return "\n".join(lines[fence_indices[-2] + 1 : fence_indices[-1]])


# Cache for loaded LCB evaluation sample files (keyed by file path)
_lcb_sample_cache: dict = {}


def _load_lcb_sample(
    data_source: str,
    unique_id: str,
    split: str = "train",
) -> Optional[dict]:
    """Load a pre-built LCB evaluation sample from the sidecar JSON file.

    Returns a dict matching CodeGenerationProblem.get_evaluation_sample():
        {"input_output": json.dumps({"inputs": [...], "outputs": [...], "fn_name": ...})}

    Args:
        data_source: Dataset name (e.g., 'livecodebench_medium').
        unique_id: Problem ID to look up.
        split: 'train' for train_samples.json, 'eval' for eval_samples.json.
    """
    from pathlib import Path
    filename = f"{split}_samples.json"
    json_path = Path.home() / "data" / "rlhf" / data_source / filename

    path_str = str(json_path)
    if path_str not in _lcb_sample_cache:
        if not json_path.exists():
            logger.warning(f"LCB sample file not found: {json_path}")
            return None
        with open(json_path) as f:
            _lcb_sample_cache[path_str] = json.load(f)

    return _lcb_sample_cache[path_str].get(unique_id)


def code_score(
    solution_str: str,
    extra_info: Optional[dict] = None,
    data_source: str = "",
    split: str = "train",
) -> float:
    """Score a code generation response using LCB's check_correctness.

    Loads the pre-built LCB evaluation sample (with fn_name for call-based
    problems) from a sidecar JSON, extracts code from the model output,
    and delegates entirely to LCB's check_correctness().

    Requires: lcb_runner (pip install from LiveCodeBench repo)

    Args:
        solution_str: Full model output containing code in ``` blocks.
        extra_info: Must contain 'unique_id' for sample lookup.
        data_source: Dataset name for locating the sidecar JSON file.
        split: 'train' or 'eval' — determines which test cases to use.

    Returns:
        Fraction of test cases passed (0.0 to 1.0).
    """
    if not extra_info:
        logger.warning("code_score called without extra_info, returning 0.0")
        return 0.0

    unique_id = extra_info.get("unique_id", "")
    if not unique_id or not data_source:
        logger.warning("code_score: missing unique_id or data_source")
        return 0.0

    sample = _load_lcb_sample(data_source, unique_id, split)
    if not sample:
        logger.warning(f"code_score: no LCB sample for {unique_id}")
        return 0.0

    code = _extract_code_block(solution_str)
    if not code:
        return 0.0

    try:
        from lcb_runner.evaluation.compute_code_generation_metrics import (
            check_correctness,
        )
    except ImportError:
        raise ImportError(
            "LiveCodeBench scoring requires lcb_runner. "
            "Install from: https://github.com/LiveCodeBench/LiveCodeBench"
        )

    try:
        result, metadata = check_correctness(
            sample=sample,
            generation=code,
            timeout=10,
            debug=False,
        )
        # result is a list of per-test-case outcomes (1=pass, negative=fail)
        if isinstance(result, list) and result:
            n_passed = sum(1 for r in result if r == 1)
            return n_passed / len(result)
        return 0.0
    except Exception as e:
        logger.debug(f"code_score: execution failed with {e}")
        return 0.0


# =============================================================================
# Short-Answer Extraction
# =============================================================================

def extract_short_answer(solution_str: str) -> str:
    """Extract a short final answer from chain-of-thought output.

    Tries: \\boxed{...}, "answer is/: X", last non-empty line.
    """
    m = re.search(r"\\boxed\{([^{}]*)\}", solution_str)
    if m:
        return m.group(1).strip()
    m = re.search(r"[Aa]nswer\s*(?:is|:)\s*(.+?)(?:\.|\n|$)", solution_str)
    if m:
        return m.group(1).strip()
    lines = [l.strip() for l in solution_str.split("\n") if l.strip()]
    return lines[-1] if lines else ""


# =============================================================================
# Yes/No Scoring (StrategyQA)
# =============================================================================

def yes_no_score(solution_str: str, ground_truth: str) -> float:
    """Find the last yes/no token in the solution and compare to ground truth."""
    matches = re.findall(r"\b(yes|no)\b", solution_str.lower())
    if not matches:
        return 0.0
    return 1.0 if matches[-1] == ground_truth.lower().strip() else 0.0


# =============================================================================
# QA Short-Answer Scoring (HotpotQA)
# =============================================================================

def qa_em_score(solution_str: str, ground_truth: str, exact: bool = False) -> float:
    """Extract a short answer and check via VERL's EM (or subEM if exact=False)."""
    from verl.utils.reward_score.search_r1_like_qa_em import em_check, subem_check
    pred = extract_short_answer(solution_str) or solution_str
    check = em_check if exact else subem_check
    return float(check(pred, ground_truth))


# =============================================================================
# Python Assert Tests (HumanEval+, MBPP+)
# =============================================================================

def python_assert_score(
    solution_str: str,
    extra_info: Optional[dict] = None,
    timeout: float = 10.0,
) -> float:
    """Run extracted code against a Python test harness in extra_info['test'].

    For HumanEval+ the harness defines a `check(candidate)` function; we
    append a call `check(<entry_point>)` so the asserts execute. For MBPP+
    the harness contains assert statements at module level, so concatenation
    alone is enough.

    Returns 1.0 if the program exits cleanly, 0.0 otherwise.
    """
    if not extra_info or not extra_info.get("test"):
        return 0.0
    code = _extract_code_block(solution_str)
    if not code:
        return 0.0
    test_str = extra_info["test"]
    entry_point = extra_info.get("entry_point", "")
    program = code + "\n\n" + test_str
    if entry_point and "def check(" in test_str:
        program = program + f"\n\ncheck({entry_point})\n"
    try:
        result = subprocess.run(
            ["python", "-c", program],
            capture_output=True,
            timeout=timeout,
        )
        return 1.0 if result.returncode == 0 else 0.0
    except subprocess.TimeoutExpired:
        return 0.0
    except Exception as e:
        logger.debug(f"python_assert_score: execution failed with {e}")
        return 0.0


# =============================================================================
# Retrosynthesis Scoring (USPTO-50K, template-based round-trip + Tanimoto)
# =============================================================================

# Caches keyed by template-file path.
_retro_template_cache: dict = {}      # path -> [{"smarts", "count", "rxn_obj", "prod_pat"}]
_retro_per_product_cache: dict = {}   # (path, target_canon) -> set of valid reactant-canonical strings


def _retro_canonicalize(smi):
    """Canonical SMILES via RDKit. Returns None on parse failure."""
    try:
        from rdkit import Chem
    except ImportError:
        raise ImportError("Retrosynthesis scoring requires rdkit. pip install rdkit")
    if not smi:
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def _retro_atom_balance_score(reactants_canon, target_canon):
    """Soft atom-balance: fraction of product heavy atoms (per element) covered
    by reactants. Returns a score in [0, 1] (1.0 = all product atoms present)."""
    from rdkit import Chem
    rmol = Chem.MolFromSmiles(reactants_canon)
    pmol = Chem.MolFromSmiles(target_canon)
    if rmol is None or pmol is None:
        return 0.0
    from collections import Counter
    rcnt = Counter(a.GetSymbol() for a in rmol.GetAtoms() if a.GetSymbol() != "H")
    pcnt = Counter(a.GetSymbol() for a in pmol.GetAtoms() if a.GetSymbol() != "H")
    total = sum(pcnt.values())
    if total == 0:
        return 0.0
    matched = sum(min(n, rcnt.get(sym, 0)) for sym, n in pcnt.items())
    return matched / total


def _retro_tanimoto(smi_a, smi_b):
    """Morgan-fingerprint Tanimoto similarity in [0, 1]. 0 on parse failure."""
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    ma, mb = Chem.MolFromSmiles(smi_a), Chem.MolFromSmiles(smi_b)
    if ma is None or mb is None:
        return 0.0
    fa = AllChem.GetMorganFingerprintAsBitVect(ma, 2, nBits=2048)
    fb = AllChem.GetMorganFingerprintAsBitVect(mb, 2, nBits=2048)
    return float(DataStructs.TanimotoSimilarity(fa, fb))


def _retro_mcs_score(smi_a, smi_b, timeout: int = 1):
    """Fragment-aware MCS atom fraction in [0, 1].

    rdFMCS only finds connected substructures, so multi-fragment SMILES
    (e.g. 'CCC(=O)O.NCC') would match poorly with itself. We greedily pair
    each fragment in `a` with its best-matching fragment in `b`, sum the
    per-pair MCS atom counts, and normalize by max(total_atoms_a, total_atoms_b).
    """
    from rdkit import Chem
    from rdkit.Chem import rdFMCS
    ma, mb = Chem.MolFromSmiles(smi_a), Chem.MolFromSmiles(smi_b)
    if ma is None or mb is None:
        return 0.0
    frags_a = list(Chem.GetMolFrags(ma, asMols=True))
    frags_b = list(Chem.GetMolFrags(mb, asMols=True))
    total_a = sum(f.GetNumHeavyAtoms() for f in frags_a)
    total_b = sum(f.GetNumHeavyAtoms() for f in frags_b)
    if total_a == 0 or total_b == 0:
        return 0.0

    # Greedy matching: for each fragment in larger side, find best partner
    primary, partners = (frags_a, frags_b) if total_a >= total_b else (frags_b, frags_a)
    primary = sorted(primary, key=lambda m: -m.GetNumHeavyAtoms())
    used = [False] * len(partners)
    matched_atoms = 0
    for p in primary:
        best_n, best_idx = 0, -1
        for j, q in enumerate(partners):
            if used[j]:
                continue
            try:
                r = rdFMCS.FindMCS(
                    [p, q], timeout=timeout,
                    matchValences=False,
                    ringMatchesRingOnly=True,
                    completeRingsOnly=True,
                )
                if not r.canceled and r.numAtoms > best_n:
                    best_n, best_idx = r.numAtoms, j
            except Exception:
                continue
        if best_idx >= 0:
            used[best_idx] = True
            matched_atoms += best_n
    return matched_atoms / max(total_a, total_b)


# ~15 functional-group SMARTS, biased toward USPTO-50K reaction classes
_FG_SMARTS_RAW = [
    ("amide",        "[NX3;H1,H2][CX3](=[OX1])[#6]"),
    ("ester",        "[#6][CX3](=O)[OX2H0][#6]"),
    ("carboxylic",   "[CX3](=O)[OX2H1]"),
    ("alcohol",      "[OX2H][CX4]"),
    ("amine_1",      "[NX3;H2;!$(NC=O)][CX4]"),
    ("amine_2",      "[NX3;H1;!$(NC=O)]([CX4])[CX4]"),
    ("amine_3",      "[NX3;H0;!$(NC=O)]([CX4])([CX4])[CX4]"),
    ("aryl_halide",  "[c;$(c[F,Cl,Br,I])]"),
    ("alkyl_halide", "[CX4][F,Cl,Br,I]"),
    ("nitro",        "[$([NX3](=O)=O),$([N+](=O)[O-])]"),
    ("nitrile",      "[CX2]#N"),
    ("ketone",       "[#6][CX3](=O)[#6]"),
    ("aldehyde",     "[CX3H1](=O)[#6]"),
    ("ether",        "[OD2]([#6])[#6]"),
    ("sulfonyl",     "[#16](=O)(=O)"),
    ("boronic",      "[#5]([OX2])[OX2]"),
    ("aromatic_ring", "c1ccccc1"),
]
_FG_PATTERNS_CACHE = None


def _retro_fg_patterns():
    """Lazy-compile functional-group SMARTS patterns once."""
    global _FG_PATTERNS_CACHE
    if _FG_PATTERNS_CACHE is None:
        from rdkit import Chem
        out = []
        for name, smarts in _FG_SMARTS_RAW:
            patt = Chem.MolFromSmarts(smarts)
            if patt is not None:
                out.append((name, patt))
        _FG_PATTERNS_CACHE = out
    return _FG_PATTERNS_CACHE


def _retro_fg_profile(smi):
    """Vector of functional-group occurrence counts."""
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return [len(mol.GetSubstructMatches(patt)) for _, patt in _retro_fg_patterns()]


def _retro_fg_score(smi_a, smi_b):
    """Cosine similarity of functional-group count vectors. [0, 1]."""
    va, vb = _retro_fg_profile(smi_a), _retro_fg_profile(smi_b)
    if va is None or vb is None:
        return 0.0
    import math
    dot = sum(x * y for x, y in zip(va, vb))
    na = math.sqrt(sum(x * x for x in va))
    nb = math.sqrt(sum(y * y for y in vb))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _retro_fragment_count(smi):
    """Number of '.'-separated fragments in a SMILES."""
    return smi.count(".") + 1 if smi else 0


def _retro_chemistry_score(pred_canon, gold_canon):
    """Composite graded chemistry similarity in [0, 1]:
        0.40 * MCS + 0.25 * Tanimoto + 0.25 * FG-cosine + 0.10 * fragment-match
    """
    if not gold_canon:
        return 0.0
    mcs = _retro_mcs_score(pred_canon, gold_canon)
    tan = _retro_tanimoto(pred_canon, gold_canon)
    fg = _retro_fg_score(pred_canon, gold_canon)
    frag = 1.0 if _retro_fragment_count(pred_canon) == _retro_fragment_count(gold_canon) else 0.0
    return 0.40 * mcs + 0.25 * tan + 0.25 * fg + 0.10 * frag


def _retro_load_templates(template_path: str):
    """Load templates and pre-build RDKit reaction objects + product-side patterns."""
    if template_path in _retro_template_cache:
        return _retro_template_cache[template_path]
    from rdkit import Chem
    from rdkit.Chem import AllChem
    with open(template_path) as f:
        raw = json.load(f)
    out = []
    for entry in raw:
        smarts = entry["smarts"]
        try:
            rxn = AllChem.ReactionFromSmarts(smarts)
            # Retro template: products>>reactants in rdchiral form, so RDKit's
            # "reactant template" (the LHS) is actually the PRODUCT side.
            prod_pat = rxn.GetReactantTemplate(0) if rxn.GetNumReactantTemplates() > 0 else None
        except Exception:
            continue
        if prod_pat is None:
            continue
        out.append({
            "smarts": smarts,
            "count": entry.get("count", 1),
            "rxn_obj": rxn,
            "prod_pat": prod_pat,
        })
    _retro_template_cache[template_path] = out
    return out


def _retro_valid_disconnections(target_canon: str, template_path: str, max_templates: int = 200):
    """For a given target product, return the set of canonical reactant strings
    produced by applying any matching template. Cached per target.

    Filters templates by HasSubstructMatch on the product-side pattern, then
    runs the top-N most-common matches via RDKit RunReactants.
    """
    from rdkit import Chem
    cache_key = (template_path, target_canon)
    if cache_key in _retro_per_product_cache:
        return _retro_per_product_cache[cache_key]

    target_mol = Chem.MolFromSmiles(target_canon)
    if target_mol is None:
        _retro_per_product_cache[cache_key] = set()
        return set()

    templates = _retro_load_templates(template_path)
    matching = []
    for t in templates:
        try:
            if target_mol.HasSubstructMatch(t["prod_pat"]):
                matching.append(t)
        except Exception:
            continue
    matching.sort(key=lambda t: -t["count"])
    matching = matching[:max_templates]

    valid_reactants: set = set()
    for t in matching:
        try:
            outcomes = t["rxn_obj"].RunReactants((target_mol,))
        except Exception:
            continue
        for outset in outcomes:
            try:
                fragments = []
                for m in outset:
                    Chem.SanitizeMol(m)
                    fragments.append(Chem.MolToSmiles(m))
                joined = ".".join(fragments)
                canon = _retro_canonicalize(joined)
                if canon:
                    valid_reactants.add(canon)
            except Exception:
                continue

    _retro_per_product_cache[cache_key] = valid_reactants
    return valid_reactants


_BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")


def _retro_extract_boxed(text: str):
    matches = _BOXED_RE.findall(text or "")
    return matches[-1].strip() if matches else None


def retro_score(
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict] = None,
    data_source: str = "",
) -> float:
    """Layered retrosynthesis reward with diversity-enhancing chemistry signals.

    Tiers (highest wins):
      0.0       - invalid SMILES or hacking (predicted == product)
      0..0.5    - graded chemistry score (MCS + Tanimoto + FG + frag-count)
                  multiplied by soft atom-balance score
      0.8       - valid template-based disconnection (algorithmic round-trip)
      1.0       - exact canonical match to gold

    extra_info must contain 'product_smiles' (target product, canonical or not)
    and optionally 'template_path' (defaults to the per-dataset templates.json).
    """
    extra_info = extra_info or {}
    pred_raw = _retro_extract_boxed(solution_str)
    if not pred_raw:
        return 0.0

    pred_canon = _retro_canonicalize(pred_raw)
    if pred_canon is None:
        return 0.0  # invalid SMILES

    target_smiles = extra_info.get("product_smiles", "")
    target_canon = _retro_canonicalize(target_smiles)
    if target_canon is None:
        gold_canon = _retro_canonicalize(ground_truth)
        return 1.0 if (gold_canon and pred_canon == gold_canon) else 0.0

    # Anti-hack: predicting the target itself is not a disconnection
    if pred_canon == target_canon:
        return 0.0

    gold_canon = _retro_canonicalize(ground_truth)
    if gold_canon and pred_canon == gold_canon:
        return 1.0

    # Template-based round-trip
    template_path = extra_info.get(
        "template_path",
        str(Path.home() / "data" / "rlhf" / "retrosynthesis_uspto50k" / "templates.json"),
    )
    try:
        if Path(template_path).exists():
            valid = _retro_valid_disconnections(target_canon, template_path)
            if pred_canon in valid:
                return 0.8
    except Exception as e:
        logger.debug(f"retro_score template check failed: {e}")

    # Graded chemistry score (multiplicative soft atom-balance gate)
    if gold_canon:
        chem = _retro_chemistry_score(pred_canon, gold_canon)
        balance = _retro_atom_balance_score(pred_canon, target_canon)
        return 0.5 * chem * balance
    return 0.0
