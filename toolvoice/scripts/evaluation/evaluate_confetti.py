"""
AST-based Function Call Parser and Evaluator

Parses function calls into tree structures and performs binary matching
at each node level: function name, parameter names, and parameter values.

Based on CONFETTI paper (Alkhouli et al., 2025):
- AST Soft: Uses AlignScore for soft matching of string parameter values
- Non-string values are matched in binary fashion
- Function names and parameter names are matched in binary fashion

Reference: https://github.com/amazon-science/confetti
"""

import ast
import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

# Try to import AlignScore for soft string matching (as per CONFETTI paper).
# AlignScore pins transformers<5 and pytorch_lightning<2; these two lines patch
# the two APIs it relies on that were removed/changed in newer releases of each.
try:
    import torch as _torch
    import transformers as _transformers
    if not hasattr(_transformers, 'AdamW'):
        _transformers.AdamW = _torch.optim.AdamW

    import pytorch_lightning as _pl
    from alignscore.model import BERTAlignModel as _BERTAlignModel
    _BERTAlignModel.load_from_checkpoint = classmethod(_pl.LightningModule.load_from_checkpoint.__func__)

    from alignscore import AlignScore
    ALIGNSCORE_AVAILABLE = True
except ImportError:
    ALIGNSCORE_AVAILABLE = False
    AlignScore = None

# Global AlignScore model (lazy loaded)
_alignscore_model = None


def _get_device():
    """Get the best available device (CUDA if available, else CPU)."""
    try:
        import torch
        if torch.cuda.is_available():
            return 'cuda'
    except ImportError:
        pass
    return 'cpu'


def get_alignscore_model():
    """Lazy load AlignScore model with GPU support and optimized batch size."""
    global _alignscore_model
    if _alignscore_model is None and ALIGNSCORE_AVAILABLE:
        device = _get_device()
        # Use larger batch size for GPU (A100 80GB can handle 512+ easily)
        batch_size = 512 if device == 'cuda' else 32
        _alignscore_model = AlignScore(
            model='roberta-base',
            batch_size=batch_size,
            device=device,
            ckpt_path='https://huggingface.co/yzha/AlignScore/resolve/main/AlignScore-base.ckpt',
            evaluation_mode='nli_sp'
        )
        print(f"AlignScore loaded on {device} with batch_size={batch_size}")
    return _alignscore_model


# Pairs are scored in chunks rather than one giant call so that a single bad pair
# cannot take down a whole file's scores (see _score_chunked).
ALIGNSCORE_CHUNK = 256


def batch_alignscore(contexts: list[str], claims: list[str]) -> list[float]:
    """
    Score multiple context-claim pairs in batches for efficiency.

    An empty claim is scored 0.0 without calling the model: AlignScore's 'nli_sp'
    mode splits the claim into sentences, and an empty claim yields zero chunks,
    which makes its internal torch.cat() raise. Scoring the rest in chunks (and
    falling back to per-pair only for a chunk that fails) keeps one degenerate
    pair from zeroing every other pair in the file.

    Args:
        contexts: List of reference/gold strings
        claims: List of predicted strings

    Returns:
        List of alignment scores between 0 and 1
    """
    if not contexts or not claims:
        return []

    if len(contexts) != len(claims):
        raise ValueError("contexts and claims must have the same length")

    model = get_alignscore_model()
    if model is None:
        # No model: every string pair silently scores 0, which looks like a very
        # bad model rather than a broken setup, so say so loudly.
        print("AlignScore model unavailable - returning 0.0 for all string pairs. "
              "Scores will understate soft matching; pass --binary for honest binary scoring.")
        return [0.0] * len(contexts)

    scores: list[float | None] = [None] * len(contexts)
    scorable = []
    for i, claim in enumerate(claims):
        if not str(claim).strip():
            scores[i] = 0.0  # an empty prediction earns no credit anyway
        else:
            scorable.append(i)

    n_failed = 0
    for start in range(0, len(scorable), ALIGNSCORE_CHUNK):
        idxs = scorable[start:start + ALIGNSCORE_CHUNK]
        try:
            out = model.score(contexts=[contexts[i] for i in idxs],
                              claims=[claims[i] for i in idxs])
            for i, s in zip(idxs, out):
                scores[i] = float(s)
        except Exception as e:
            print(f"AlignScore chunk error ({e}) - retrying {len(idxs)} pairs individually")
            for i in idxs:
                try:
                    scores[i] = float(model.score(contexts=[contexts[i]], claims=[claims[i]])[0])
                except Exception:
                    scores[i] = 0.0
                    n_failed += 1

    if n_failed:
        print(f"AlignScore: {n_failed}/{len(contexts)} pairs could not be scored (counted as 0.0)")

    return [0.0 if s is None else s for s in scores]


@dataclass
class ParameterNode:
    """Represents a parameter in the function call tree."""
    name: str
    value: Any

    def __repr__(self):
        return f"ParameterNode(name={self.name!r}, value={self.value!r})"


@dataclass
class FunctionCallNode:
    """Represents a function call as a tree node."""
    name: str
    parameters: list[ParameterNode] = field(default_factory=list)

    def __repr__(self):
        return f"FunctionCallNode(name={self.name!r}, parameters={self.parameters})"


@dataclass
class MatchResult:
    """Match results for each tree node (supports both binary and soft scoring)."""
    function_name_match: int  # Always binary: 1 or 0
    parameter_name_matches: dict[str, int]  # param_name -> 1 or 0 (always binary)
    parameter_value_matches: dict[str, float]  # param_name -> 0.0 to 1.0 (soft) or 0/1 (binary)

    @property
    def all_match(self) -> bool:
        """Returns True if all nodes match perfectly (score >= 1.0)."""
        if self.function_name_match == 0:
            return False
        if any(v == 0 for v in self.parameter_name_matches.values()):
            return False
        if any(v < 1.0 for v in self.parameter_value_matches.values()):
            return False
        return True

    @property
    def score(self) -> float:
        """
        Returns overall match score as fraction (AST Soft score).

        Following CONFETTI: Average of all node scores where:
        - Function name: 0 or 1
        - Parameter names: 0 or 1
        - Parameter values: 0.0 to 1.0 (soft) or 0/1 (binary)
        """
        total = 1 + len(self.parameter_name_matches) + len(self.parameter_value_matches)
        matched = self.function_name_match + sum(self.parameter_name_matches.values()) + sum(self.parameter_value_matches.values())
        return matched / total if total > 0 else 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'function_name_match': self.function_name_match,
            'parameter_name_matches': self.parameter_name_matches,
            'parameter_value_matches': {k: round(v, 4) for k, v in self.parameter_value_matches.items()},
            'all_match': self.all_match,
            'score': round(self.score, 4)
        }


def normalize_value(value: Any) -> Any:
    """Normalize a value for comparison."""
    if isinstance(value, str):
        return value.strip().lower()
    if isinstance(value, list):
        if len(value) == 1:
            return normalize_value(value[0])
        return [normalize_value(v) for v in value]
    if isinstance(value, dict):
        return {k: normalize_value(v) for k, v in value.items()}
    return value


def is_string_type(value: Any) -> bool:
    """Check if a value is string type (for soft matching)."""
    if isinstance(value, str):
        return True
    if isinstance(value, list) and len(value) > 0:
        return all(isinstance(v, str) for v in value)
    return False


def values_match_soft(predicted: Any, gold: Any, use_alignscore: bool = False) -> float:
    """
    Check if two values match, returning a score between 0 and 1.

    Following CONFETTI paper:
    - String values: Use AlignScore for soft matching (if available)
    - Non-string values: Binary matching (0 or 1)

    Args:
        predicted: Predicted parameter value
        gold: Gold/reference parameter value
        use_alignscore: Whether to use AlignScore for string matching

    Returns:
        Float score between 0.0 and 1.0
    """
    pred_norm = normalize_value(predicted)
    gold_norm = normalize_value(gold)

    # Handle list vs single value - extract single values
    if isinstance(gold_norm, list) and len(gold_norm) == 1:
        gold_norm = gold_norm[0]
    if isinstance(pred_norm, list) and len(pred_norm) == 1:
        pred_norm = pred_norm[0]

    # Exact match
    if pred_norm == gold_norm:
        return 1.0

    # String comparison
    if str(pred_norm) == str(gold_norm):
        return 1.0

    # For string types, use soft matching
    if is_string_type(predicted) and is_string_type(gold):
        pred_str = str(pred_norm) if not isinstance(pred_norm, str) else pred_norm
        gold_str = str(gold_norm) if not isinstance(gold_norm, str) else gold_norm

        # Try AlignScore if available and requested
        if use_alignscore and ALIGNSCORE_AVAILABLE:
            try:
                model = get_alignscore_model()
                if model is not None:
                    score = model.score(contexts=[gold_str], claims=[pred_str])[0]
                    return float(score)
            except Exception:
                pass  # Fall back to fuzzy matching

        # Fuzzy string matching fallback
        pred_clean = re.sub(r'\s+', ' ', pred_str.strip())
        gold_clean = re.sub(r'\s+', ' ', gold_str.strip())

        if pred_clean == gold_clean:
            return 1.0
        # Containment check (partial match)
        if pred_clean in gold_clean or gold_clean in pred_clean:
            return 1.0

        return 0.0

    # Non-string types: binary matching only
    return 0.0


def values_match(predicted: Any, gold: Any) -> bool:
    """Check if two values match (binary). Wrapper for backwards compatibility."""
    return values_match_soft(predicted, gold, use_alignscore=False) >= 1.0


def parse_model_response(response_str: str) -> list[FunctionCallNode]:
    """
    Parse model response formats:
    1. GPT format: [{"id": "...", "name": "function_name", "args_json": "{\"param\":\"value\"}"}]
    2. Qwen format: ['text <tool_call>{"name": "...", "arguments": {...}}</tool_call>']
    """
    function_calls = []

    if not response_str or response_str == 'nan':
        return function_calls

    # First, try to parse as Python literal if it looks like one
    parsed_data = None
    if response_str.strip().startswith('['):
        try:
            parsed_data = ast.literal_eval(response_str)
        except (ValueError, SyntaxError):
            try:
                parsed_data = json.loads(response_str)
            except json.JSONDecodeError:
                pass

    # If parsed as a list, check each element
    if isinstance(parsed_data, list):
        for item in parsed_data:
            if isinstance(item, str) and '<tool_call>' in item:
                # Qwen format: string with tool_call tags
                function_calls.extend(parse_qwen_tool_calls(item))
            elif isinstance(item, dict):
                # GPT format
                func_name = item.get('name', '')
                args_json = item.get('args_json', '{}')

                try:
                    if isinstance(args_json, str):
                        args = json.loads(args_json)
                    else:
                        args = args_json
                except json.JSONDecodeError:
                    args = {}

                parameters = [ParameterNode(name=k, value=v) for k, v in args.items()]
                function_calls.append(FunctionCallNode(name=func_name, parameters=parameters))
        return function_calls

    # Check for Qwen format with <tool_call> tags in raw string
    if '<tool_call>' in response_str:
        return parse_qwen_tool_calls(response_str)

    return function_calls


def parse_qwen_tool_calls(text: str) -> list[FunctionCallNode]:
    """
    Parse Qwen format tool calls with <tool_call> tags.
    Format: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
    """
    function_calls = []

    # Find all tool_call tags
    pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
    matches = re.findall(pattern, text, re.DOTALL)

    for match in matches:
        try:
            tool_data = json.loads(match)
            func_name = tool_data.get('name', '')
            args = tool_data.get('arguments', {})

            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}

            parameters = [ParameterNode(name=k, value=v) for k, v in args.items()]
            function_calls.append(FunctionCallNode(name=func_name, parameters=parameters))
        except json.JSONDecodeError:
            continue

    return function_calls


def parse_gold_answer(gold_str: str) -> list[FunctionCallNode]:
    """
    Parse gold answer format:
    [{'function_name': {'param': ['value']}}]
    """
    function_calls = []

    if not gold_str or gold_str == 'nan':
        return function_calls

    try:
        # Try parsing as Python literal (gold answers often use single quotes)
        gold_data = ast.literal_eval(gold_str)
    except (ValueError, SyntaxError):
        try:
            gold_data = json.loads(gold_str)
        except json.JSONDecodeError:
            return function_calls

    if not isinstance(gold_data, list):
        gold_data = [gold_data]

    for item in gold_data:
        if isinstance(item, dict):
            for func_name, params in item.items():
                if isinstance(params, dict):
                    parameters = [ParameterNode(name=k, value=v) for k, v in params.items()]
                else:
                    parameters = []
                function_calls.append(FunctionCallNode(name=func_name, parameters=parameters))

    return function_calls


def match_function_calls(predicted: FunctionCallNode, gold: FunctionCallNode,
                         use_soft_matching: bool = False) -> MatchResult:
    """
    Compare predicted and gold function calls, producing match scores for each node.

    Args:
        predicted: Predicted function call
        gold: Gold/reference function call
        use_soft_matching: If True, use AlignScore for soft string matching (AST Soft).
                          If False, use binary matching only.

    Following CONFETTI paper Section 4.1:
    - Function names: Binary match (0 or 1)
    - Parameter names: Binary match (0 or 1)
    - Parameter values (string): Soft match using AlignScore (0.0 to 1.0) if use_soft_matching=True
    - Parameter values (non-string): Binary match (0 or 1)
    """
    # Match function name (always binary)
    func_match = 1 if predicted.name == gold.name else 0

    # Build parameter lookup for gold
    gold_params = {p.name: p.value for p in gold.parameters}
    pred_params = {p.name: p.value for p in predicted.parameters}

    param_name_matches = {}
    param_value_matches = {}

    for param_name in gold_params.keys():
        # Check if parameter name exists in prediction (always binary)
        param_name_matches[param_name] = 1 if param_name in pred_params else 0

        # Check if parameter value matches
        if param_name in pred_params:
            if use_soft_matching:
                # AST Soft: Use AlignScore for string values
                score = values_match_soft(pred_params[param_name], gold_params[param_name],
                                         use_alignscore=True)
                param_value_matches[param_name] = score
            else:
                # Binary matching
                param_value_matches[param_name] = 1 if values_match(pred_params[param_name],
                                                                     gold_params[param_name]) else 0
        else:
            param_value_matches[param_name] = 0

    # Track extra parameters in prediction (not in gold) - penalize
    for param_name in pred_params.keys():
        if param_name not in gold_params:
            param_name_matches[f"extra:{param_name}"] = 0
            param_value_matches[f"extra:{param_name}"] = 0

    return MatchResult(
        function_name_match=func_match,
        parameter_name_matches=param_name_matches,
        parameter_value_matches=param_value_matches
    )


def evaluate_single_turn(model_response: str, gold_answer: str,
                         use_soft_matching: bool = False) -> dict:
    """
    Evaluate a single turn's function call prediction against ground truth.

    Args:
        model_response: Model's predicted function call(s)
        gold_answer: Gold/reference function call(s)
        use_soft_matching: If True, use AlignScore for soft string matching (AST Soft)

    Returns:
        Dictionary with match results for each function call pair.
    """
    predicted_calls = parse_model_response(model_response)
    gold_calls = parse_gold_answer(gold_answer)

    results = {
        'predicted_calls': [repr(c) for c in predicted_calls],
        'gold_calls': [repr(c) for c in gold_calls],
        'num_predicted': len(predicted_calls),
        'num_gold': len(gold_calls),
        'matches': [],
        'overall_binary': 0,
        'overall_score': 0.0
    }

    if len(predicted_calls) == 0 and len(gold_calls) == 0:
        # Both empty - correct
        results['overall_binary'] = 1
        results['overall_score'] = 1.0
        return results

    if len(predicted_calls) == 0 or len(gold_calls) == 0:
        # One empty, one not - mismatch
        results['overall_binary'] = 0
        results['overall_score'] = 0.0
        return results

    # Match predicted calls to gold calls (one-to-one, in order)
    total_score = 0.0
    all_match = True

    for i, (pred, gold) in enumerate(zip(predicted_calls, gold_calls)):
        match_result = match_function_calls(pred, gold, use_soft_matching=use_soft_matching)
        results['matches'].append({
            'index': i,
            'predicted': repr(pred),
            'gold': repr(gold),
            **match_result.to_dict()
        })
        total_score += match_result.score
        if not match_result.all_match:
            all_match = False

    # Handle mismatched counts
    if len(predicted_calls) != len(gold_calls):
        all_match = False

    results['overall_binary'] = 1 if all_match else 0
    results['overall_score'] = total_score / max(len(predicted_calls), len(gold_calls))

    return results


def evaluate_dataset_batched(input_csv: str, output_csv: str = None,
                             response_column: str = 'model_response') -> dict:
    """
    Evaluate all rows using batched AlignScore for efficiency on GPU.

    This function collects all string parameter pairs first, scores them
    in a single batch using AlignScore, then assembles the results.
    Much more efficient than per-sample scoring on GPU.

    Args:
        input_csv: Path to input CSV with model_response/tool_calls and gold_answer columns
        output_csv: Optional path for output CSV with evaluation results
        response_column: Column name for model responses (default: 'model_response', use 'tool_calls' for Gemini)

    Returns:
        Dictionary with aggregate statistics
    """
    import pandas as pd

    df = pd.read_csv(input_csv)

    # First pass: parse all function calls and collect string pairs for batched scoring
    parsed_data = []  # List of (idx, predicted_calls, gold_calls)
    string_pairs = []  # List of (idx, call_idx, param_name, pred_str, gold_str)

    for idx, row in df.iterrows():
        model_response = str(row.get(response_column, ''))
        # Fall back to assistant_text if tool_calls is empty (Gemini text responses)
        if not model_response or model_response == 'nan' or model_response.strip() == '':
            model_response = str(row.get('assistant_text', ''))
        gold_answer = str(row.get('gold_answer', ''))

        predicted_calls = parse_model_response(model_response)
        gold_calls = parse_gold_answer(gold_answer)
        parsed_data.append((idx, predicted_calls, gold_calls))

        # Collect string pairs that need AlignScore
        for call_idx, (pred, gold) in enumerate(zip(predicted_calls, gold_calls)):
            gold_params = {p.name: p.value for p in gold.parameters}
            pred_params = {p.name: p.value for p in pred.parameters}

            for param_name, gold_value in gold_params.items():
                if param_name in pred_params:
                    pred_value = pred_params[param_name]
                    # Check if both are strings (need soft matching)
                    if is_string_type(pred_value) and is_string_type(gold_value):
                        pred_norm = normalize_value(pred_value)
                        gold_norm = normalize_value(gold_value)
                        # Only score non-exact matches
                        if pred_norm != gold_norm and str(pred_norm) != str(gold_norm):
                            pred_str = str(pred_norm) if not isinstance(pred_norm, str) else pred_norm
                            gold_str = str(gold_norm) if not isinstance(gold_norm, str) else gold_norm
                            string_pairs.append((idx, call_idx, param_name, pred_str, gold_str))

    # Batch score all string pairs at once
    alignscore_results = {}
    if string_pairs and ALIGNSCORE_AVAILABLE:
        contexts = [pair[4] for pair in string_pairs]  # gold strings
        claims = [pair[3] for pair in string_pairs]    # pred strings
        scores = batch_alignscore(contexts, claims)
        for pair, score in zip(string_pairs, scores):
            idx, call_idx, param_name, _, _ = pair
            alignscore_results[(idx, call_idx, param_name)] = score

    # Second pass: assemble results using precomputed scores
    results = []
    total_binary_correct = 0
    total_score = 0.0
    func_name_correct = 0
    param_name_correct = 0
    param_value_correct = 0.0
    total_params = 0

    for idx, predicted_calls, gold_calls in parsed_data:
        result = {
            'predicted_calls': [repr(c) for c in predicted_calls],
            'gold_calls': [repr(c) for c in gold_calls],
            'num_predicted': len(predicted_calls),
            'num_gold': len(gold_calls),
            'matches': [],
            'overall_binary': 0,
            'overall_score': 0.0,
            'index': idx
        }

        if len(predicted_calls) == 0 and len(gold_calls) == 0:
            result['overall_binary'] = 1
            result['overall_score'] = 1.0
            results.append(result)
            total_binary_correct += 1
            total_score += 1.0
            continue

        if len(predicted_calls) == 0 or len(gold_calls) == 0:
            results.append(result)
            continue

        call_total_score = 0.0
        all_match = True

        for call_idx, (pred, gold) in enumerate(zip(predicted_calls, gold_calls)):
            func_match = 1 if pred.name == gold.name else 0
            func_name_correct += func_match

            gold_params = {p.name: p.value for p in gold.parameters}
            pred_params = {p.name: p.value for p in pred.parameters}

            param_name_matches = {}
            param_value_matches = {}

            for param_name, gold_value in gold_params.items():
                param_name_matches[param_name] = 1 if param_name in pred_params else 0

                if param_name in pred_params:
                    pred_value = pred_params[param_name]
                    # Check precomputed AlignScore
                    key = (idx, call_idx, param_name)
                    if key in alignscore_results:
                        score = alignscore_results[key]
                    else:
                        # Exact match or non-string: use binary
                        score = 1.0 if values_match_soft(pred_value, gold_value, use_alignscore=False) >= 1.0 else 0.0
                    param_value_matches[param_name] = score
                else:
                    param_value_matches[param_name] = 0.0

            for param_name in pred_params:
                if param_name not in gold_params:
                    param_name_matches[f"extra:{param_name}"] = 0
                    param_value_matches[f"extra:{param_name}"] = 0.0

            # Calculate match result score
            total_nodes = 1 + len(param_name_matches) + len(param_value_matches)
            matched = func_match + sum(param_name_matches.values()) + sum(param_value_matches.values())
            match_score = matched / total_nodes if total_nodes > 0 else 0.0

            match_all = (func_match == 1 and
                        all(v == 1 for v in param_name_matches.values()) and
                        all(v >= 1.0 for v in param_value_matches.values()))

            result['matches'].append({
                'index': call_idx,
                'predicted': repr(pred),
                'gold': repr(gold),
                'function_name_match': func_match,
                'parameter_name_matches': param_name_matches,
                'parameter_value_matches': {k: round(v, 4) for k, v in param_value_matches.items()},
                'all_match': match_all,
                'score': round(match_score, 4)
            })

            call_total_score += match_score
            if not match_all:
                all_match = False

            # Aggregate stats
            for param, val in param_name_matches.items():
                if not param.startswith('extra:'):
                    param_name_correct += val
                    total_params += 1
            for param, val in param_value_matches.items():
                if not param.startswith('extra:'):
                    param_value_correct += val

        if len(predicted_calls) != len(gold_calls):
            all_match = False

        result['overall_binary'] = 1 if all_match else 0
        result['overall_score'] = call_total_score / max(len(predicted_calls), len(gold_calls))

        results.append(result)
        total_binary_correct += result['overall_binary']
        total_score += result['overall_score']

    num_samples = len(df)
    stats = {
        'total_samples': num_samples,
        'binary_accuracy': total_binary_correct / num_samples if num_samples > 0 else 0,
        'average_score': total_score / num_samples if num_samples > 0 else 0,
        'function_name_accuracy': func_name_correct / num_samples if num_samples > 0 else 0,
        'parameter_name_accuracy': param_name_correct / total_params if total_params > 0 else 0,
        'parameter_value_accuracy': param_value_correct / total_params if total_params > 0 else 0,
    }

    if output_csv:
        df['ast_binary_match'] = [r['overall_binary'] for r in results]
        df['ast_score'] = [r['overall_score'] for r in results]
        df['ast_eval_details'] = [json.dumps(r['matches']) for r in results]
        df.to_csv(output_csv, index=False)
        print(f"Results saved to {output_csv}")

    return stats, results


def evaluate_dataset(input_csv: str, output_csv: str = None,
                     use_soft_matching: bool = False,
                     response_column: str = 'model_response') -> dict:
    """
    Evaluate all rows in a confetti output CSV file.

    Args:
        input_csv: Path to input CSV with model_response/tool_calls and gold_answer columns
        output_csv: Optional path for output CSV with evaluation results
        use_soft_matching: If True, use AlignScore for soft string matching (AST Soft).
                          When True, uses batched evaluation for GPU efficiency.
        response_column: Column name for model responses (default: 'model_response', use 'tool_calls' for Gemini)

    Returns:
        Dictionary with aggregate statistics
    """
    # Use batched evaluation for soft matching (much more efficient on GPU)
    if use_soft_matching and ALIGNSCORE_AVAILABLE:
        return evaluate_dataset_batched(input_csv, output_csv, response_column=response_column)

    import pandas as pd

    df = pd.read_csv(input_csv)

    results = []
    total_binary_correct = 0
    total_score = 0.0

    # Detailed breakdown
    func_name_correct = 0
    param_name_correct = 0
    param_value_correct = 0
    total_params = 0

    for idx, row in df.iterrows():
        model_response = str(row.get(response_column, ''))
        # Fall back to assistant_text if tool_calls is empty (Gemini text responses)
        if not model_response or model_response == 'nan' or model_response.strip() == '':
            model_response = str(row.get('assistant_text', ''))
        gold_answer = str(row.get('gold_answer', ''))

        eval_result = evaluate_single_turn(model_response, gold_answer,
                                           use_soft_matching=use_soft_matching)
        eval_result['index'] = idx
        results.append(eval_result)

        total_binary_correct += eval_result['overall_binary']
        total_score += eval_result['overall_score']

        # Aggregate match statistics
        for match in eval_result['matches']:
            func_name_correct += match['function_name_match']
            for param, val in match['parameter_name_matches'].items():
                if not param.startswith('extra:'):
                    param_name_correct += val
                    total_params += 1
            for param, val in match['parameter_value_matches'].items():
                if not param.startswith('extra:'):
                    param_value_correct += val

    num_samples = len(df)

    stats = {
        'total_samples': num_samples,
        'binary_accuracy': total_binary_correct / num_samples if num_samples > 0 else 0,
        'average_score': total_score / num_samples if num_samples > 0 else 0,
        'function_name_accuracy': func_name_correct / num_samples if num_samples > 0 else 0,
        'parameter_name_accuracy': param_name_correct / total_params if total_params > 0 else 0,
        'parameter_value_accuracy': param_value_correct / total_params if total_params > 0 else 0,
    }

    if output_csv:
        # Create output dataframe with evaluation results
        df['ast_binary_match'] = [r['overall_binary'] for r in results]
        df['ast_score'] = [r['overall_score'] for r in results]
        df['ast_eval_details'] = [json.dumps(r['matches']) for r in results]
        df.to_csv(output_csv, index=False)
        print(f"Results saved to {output_csv}")

    return stats, results


def print_tree(func_call: FunctionCallNode, indent: int = 0) -> str:
    """Pretty print a function call as a tree."""
    lines = []
    prefix = "  " * indent
    lines.append(f"{prefix}FunctionCall")
    lines.append(f"{prefix}├── name: {func_call.name!r}")
    lines.append(f"{prefix}└── parameters")
    for i, param in enumerate(func_call.parameters):
        is_last = i == len(func_call.parameters) - 1
        branch = "└──" if is_last else "├──"
        lines.append(f"{prefix}    {branch} {param.name}: {param.value!r}")
    return "\n".join(lines)


if __name__ == "__main__":
    import pandas as pd
    import argparse
    import os

    parser = argparse.ArgumentParser(description='AST-based function call evaluation (CONFETTI paper)')
    parser.add_argument('--input', '-i', type=str, help='Input CSV file path')
    parser.add_argument('--output', '-o', type=str, help='Output CSV file path (optional)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Print detailed results')
    parser.add_argument('--eval-all', action='store_true',
                        help='Evaluate every Confetti response CSV under model_responses/ at the repo root.')
    parser.add_argument('--soft', dest='soft', action='store_true', default=True,
                        help='Use AST Soft scoring (AlignScore for string params) per CONFETTI paper. '
                             'This is the default; requires the AlignScore package and checkpoint.')
    parser.add_argument('--binary', dest='soft', action='store_false',
                        help='Use AST Binary scoring (exact string match) instead of the AlignScore-based default.')
    parser.add_argument('--response-column', type=str, default=None,
                        help="Column holding model responses. Default: auto-detect per file "
                             "('tool_calls' if present, else 'model_response').")
    args = parser.parse_args()

    # Check AlignScore availability if soft mode requested
    if args.soft and not ALIGNSCORE_AVAILABLE:
        print("WARNING: AST Soft scoring requires AlignScore. Install with: pip install alignscore")
        print("         Falling back to fuzzy string matching. Pass --binary for exact-match binary scoring.")

    def detect_response_column(filepath):
        """'tool_calls' if the file has one (realtime/live and Nemotron outputs),
        else 'model_response' (Qwen-Omni and other vLLM outputs)."""
        columns = pd.read_csv(filepath, nrows=0).columns
        return 'tool_calls' if 'tool_calls' in columns else 'model_response'

    if args.eval_all:
        # Evaluate every Confetti response file the inference scripts produced.
        # Layout: model_responses/<model>_responses/BFCL_v2_conversations_clean_with_context_<model>_<tts>_<voice>.csv
        base_dir = os.path.dirname(os.path.abspath(__file__))
        responses_root = os.path.join(os.path.dirname(os.path.dirname(base_dir)), 'model_responses')
        if not os.path.isdir(responses_root):
            raise SystemExit(f"No model_responses/ directory found at {responses_root} — run inference first.")

        mode_str = "AST Soft (AlignScore)" if args.soft else "AST Binary"
        print(f"{mode_str} Confetti Benchmark Evaluation")
        print("=" * 110)
        print(f"{'Model':<32} {'TTS Model':<22} {'Voice':<8} {'Binary Acc':<12} {'Avg Score':<12} {'Func Name':<12} {'Param Val':<12}")
        print("-" * 110)

        all_results = []
        for dir_name in sorted(os.listdir(responses_root)):
            dir_path = os.path.join(responses_root, dir_name)
            if not os.path.isdir(dir_path):
                continue
            model_name = dir_name.replace('_responses', '')

            for filename in sorted(os.listdir(dir_path)):
                if not filename.endswith('.csv') or 'bfcl' not in filename.lower():
                    continue

                filepath = os.path.join(dir_path, filename)
                response_col = args.response_column or detect_response_column(filepath)
                stats, _ = evaluate_dataset(filepath, use_soft_matching=args.soft,
                                            response_column=response_col)

                # Filenames end in ..._<tts_model>_<voice>.csv
                parts = filename[:-4].split('_')
                tts_model, voice = (parts[-2], parts[-1]) if len(parts) >= 2 else ('N/A', 'N/A')

                print(f"{model_name:<32} {tts_model:<22} {voice:<8} "
                      f"{stats['binary_accuracy']*100:>10.2f}% "
                      f"{stats['average_score']*100:>10.2f}% "
                      f"{stats['function_name_accuracy']*100:>10.2f}% "
                      f"{stats['parameter_value_accuracy']*100:>10.2f}%")

                all_results.append({
                    'model': model_name,
                    'tts_model': tts_model,
                    'voice': voice,
                    'total_samples': stats['total_samples'],
                    'binary_accuracy': round(stats['binary_accuracy'] * 100, 2),
                    'average_score': round(stats['average_score'] * 100, 2),
                    'function_name_accuracy': round(stats['function_name_accuracy'] * 100, 2),
                    'parameter_name_accuracy': round(stats['parameter_name_accuracy'] * 100, 2),
                    'parameter_value_accuracy': round(stats['parameter_value_accuracy'] * 100, 2),
                    'source_file': filename,
                })

        # Save results to CSV
        mode_suffix = "soft" if args.soft else "binary"
        output_csv = os.path.join(base_dir, f'ast_eval_results_{mode_suffix}.csv')
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(output_csv, index=False)
        print(f"\nResults saved to {output_csv}")

    elif args.input:
        response_col = args.response_column or detect_response_column(args.input)
        stats, results = evaluate_dataset(args.input, args.output,
                                          use_soft_matching=args.soft,
                                          response_column=response_col)

        print("\nAST Evaluation Results")
        print("=" * 50)
        print(f"Total Samples: {stats['total_samples']}")
        print(f"Binary Accuracy (all nodes match): {stats['binary_accuracy']*100:.2f}%")
        print(f"Average Score: {stats['average_score']*100:.2f}%")
        print(f"Function Name Accuracy: {stats['function_name_accuracy']*100:.2f}%")
        print(f"Parameter Name Accuracy: {stats['parameter_name_accuracy']*100:.2f}%")
        print(f"Parameter Value Accuracy: {stats['parameter_value_accuracy']*100:.2f}%")

        if args.verbose:
            print("\n" + "=" * 50)
            print("Detailed Results (first 10 samples)")
            print("=" * 50)
            for result in results[:10]:
                print(f"\nSample {result['index']}:")
                print(f"  Predicted: {result['predicted_calls']}")
                print(f"  Gold: {result['gold_calls']}")
                print(f"  Binary Match: {result['overall_binary']}")
                print(f"  Score: {result['overall_score']:.2f}")
                for match in result['matches']:
                    print(f"  - Function Name Match: {match['function_name_match']}")
                    print(f"    Param Name Matches: {match['parameter_name_matches']}")
                    print(f"    Param Value Matches: {match['parameter_value_matches']}")
    else:
        # Demo with example data
        print("AST Function Call Parser Demo")
        print("=" * 50)

        # Example from actual data
        model_response = '[{"id": "item_xxx", "name": "HRCompanyDirectory_employees", "args_json": "{\\"name\\":\\"Shalissa Valentino\\"}"}]'
        gold_answer = "[{'HRCompanyDirectory_employees': {'name': ['Shalissa Valentino']}}]"

        print("\nModel Response:")
        print(model_response)
        print("\nGold Answer:")
        print(gold_answer)

        predicted = parse_model_response(model_response)
        gold = parse_gold_answer(gold_answer)

        print("\n--- Parsed Trees ---")
        print("\nPredicted:")
        for call in predicted:
            print(print_tree(call))

        print("\nGold:")
        for call in gold:
            print(print_tree(call))

        print("\n--- Match Results ---")
        result = evaluate_single_turn(model_response, gold_answer)
        print(f"Binary Match: {result['overall_binary']}")
        print(f"Score: {result['overall_score']:.2f}")
        for match in result['matches']:
            print(f"Function Name Match: {match['function_name_match']}")
            print(f"Parameter Name Matches: {match['parameter_name_matches']}")
            print(f"Parameter Value Matches: {match['parameter_value_matches']}")

        # Example with mismatch
        print("\n" + "=" * 50)
        print("Example with Partial Mismatch")
        print("=" * 50)

        model_response2 = '[{"id": "item_xxx", "name": "BookFlight_get_airport_code", "args_json": "{\\"query\\":\\"San Diego Airport\\"}"}]'
        gold_answer2 = "[{'BookFlight_get_airport_code': {'query': ['San Diego']}}]"

        print("\nModel Response:")
        print(model_response2)
        print("\nGold Answer:")
        print(gold_answer2)

        result2 = evaluate_single_turn(model_response2, gold_answer2)
        print(f"\nBinary Match: {result2['overall_binary']}")
        print(f"Score: {result2['overall_score']:.2f}")
        for match in result2['matches']:
            print(f"Function Name Match: {match['function_name_match']}")
            print(f"Parameter Name Matches: {match['parameter_name_matches']}")
            print(f"Parameter Value Matches: {match['parameter_value_matches']}")
