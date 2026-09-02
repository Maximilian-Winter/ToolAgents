"""Sandboxed, serializable conditions for pipeline flow control.

Pipeline JSON may come from disk, a database, or a user-facing editor, so a
condition must never be able to run arbitrary Python. Conditions here are
compiled from a deliberately small expression language: a whitelisted subset of
Python's own grammar, evaluated against the pipeline results mapping.

What is allowed:
    comparisons, ``and`` / ``or`` / ``not``, ``in`` / ``not in``, arithmetic
    (except ``**``), indexing, list/tuple/dict/set literals, conditional
    expressions, and calls to a fixed table of safe helper functions.

What is not:
    attribute access, imports, lambdas, comprehensions, assignment
    expressions, generators, ``**``, f-strings, or any name that is neither a
    pipeline result nor a whitelisted helper.

Because attribute access is rejected outright, no value reachable from the
results mapping can be used to escape the sandbox via ``__class__`` or friends.
"""

from __future__ import annotations

import abc
import ast
from typing import Any, Callable, Mapping

__all__ = [
    "Condition",
    "ExpressionCondition",
    "PipelineConditionError",
    "SafeExpression",
    "condition_from_config",
    "condition_to_config",
    "register_condition_kind",
]


class PipelineConditionError(ValueError):
    """Raised when a condition cannot be compiled or evaluated."""


# --------------------------------------------------------------------------
# The sandbox
# --------------------------------------------------------------------------

#: Upper bound on source length, as a cheap guard against pathological input.
MAX_EXPRESSION_LENGTH = 2000

#: Upper bound on AST node count, guarding against deeply nested literals.
MAX_EXPRESSION_NODES = 250

#: Largest integer literal permitted as a multiplication operand. ``**`` is
#: rejected outright, but ``draft * 100000000`` is the same memory blowup by
#: another door, so repetition counts are capped as well.
MAX_SEQUENCE_REPEAT = 10_000

_ALLOWED_NODES: tuple[type[ast.AST], ...] = (
    ast.Expression,
    # boolean / unary / binary operators
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.UnaryOp,
    ast.Not,
    ast.USub,
    ast.UAdd,
    ast.BinOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    # comparisons
    ast.Compare,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.In,
    ast.NotIn,
    ast.Is,
    ast.IsNot,
    # data
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.List,
    ast.Tuple,
    ast.Dict,
    ast.Set,
    ast.Subscript,
    ast.Slice,
    ast.IfExp,
    # calls (the callee is separately restricted to whitelisted plain names)
    ast.Call,
)

# Named explicitly so the error can explain *why*, not merely "not allowed".
_REJECTED_NODES: dict[type[ast.AST], str] = {
    ast.Attribute: "attribute access is not allowed",
    ast.Lambda: "lambdas are not allowed",
    ast.ListComp: "comprehensions are not allowed",
    ast.SetComp: "comprehensions are not allowed",
    ast.DictComp: "comprehensions are not allowed",
    ast.GeneratorExp: "generator expressions are not allowed",
    ast.Await: "await is not allowed",
    ast.NamedExpr: "assignment expressions (:=) are not allowed",
    ast.Starred: "argument unpacking is not allowed",
    ast.JoinedStr: "f-strings are not allowed",
    ast.FormattedValue: "f-strings are not allowed",
    ast.Pow: "the ** operator is not allowed",
}


def _fn_contains(haystack: Any, needle: Any) -> bool:
    """Return whether ``needle`` occurs in ``haystack``."""

    return needle in haystack


def _fn_startswith(value: Any, prefix: Any) -> bool:
    """Return whether the string form of ``value`` starts with ``prefix``."""

    return str(value).startswith(str(prefix))


def _fn_endswith(value: Any, suffix: Any) -> bool:
    """Return whether the string form of ``value`` ends with ``suffix``."""

    return str(value).endswith(str(suffix))


def _fn_is_empty(value: Any) -> bool:
    """Return whether ``value`` is ``None`` or an empty container/string."""

    if value is None:
        return True
    try:
        return len(value) == 0
    except TypeError:
        return False


def _fn_default(value: Any, fallback: Any) -> Any:
    """Return ``fallback`` when ``value`` is ``None``, otherwise ``value``.

    This handles a result that *exists* but is ``None``. For a result that may
    not exist at all — a key written only inside a branch that did not run —
    use ``defined('name')``, which is bound per evaluation.
    """

    return fallback if value is None else value


def _fn_defined_placeholder(name: Any) -> bool:
    """Stand-in so validation accepts ``defined``; rebound per evaluation."""

    raise PipelineConditionError(
        "defined() is only available while evaluating a condition."
    )


class _LazyNamespace(dict):
    """Resolve result names on demand so boolean operators short-circuit.

    Pre-checking every name in an expression makes ``or`` and conditional
    expressions useless: ``"has_score and score > 3"`` would fail before it
    could short-circuit. Resolving lazily means a name is only required if the
    expression actually reaches it. Used as ``eval``'s locals mapping, so a
    miss surfaces through ``__missing__``.
    """

    def __init__(
        self,
        source: str,
        results: Mapping[str, Any],
        helpers: Mapping[str, Any],
    ) -> None:
        super().__init__(results)
        self._source = source
        self._helpers = helpers
        self._results = results

    def __missing__(self, key: str) -> Any:
        # Helper functions live in globals; name lookup consults locals first,
        # so they arrive here and must be handed back rather than reported
        # missing.
        if key in self._helpers:
            return self._helpers[key]

        # A sectioned results mapping exposes section names that its flat view
        # does not, so that a condition can say outputs['draft'] explicitly.
        try:
            return self._results[key]
        except KeyError:
            pass
        available = ", ".join(sorted(k for k in self if not k.startswith("_")))
        raise PipelineConditionError(
            f"Condition expression {self._source!r} references unknown "
            f"result: {key}. Available results: {available or '<none>'}. "
            f"Use defined('{key}') to test for a result that may not exist yet."
        )


#: Names callable from within an expression. Everything else is rejected.
SAFE_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "len": len,
    "abs": abs,
    "min": min,
    "max": max,
    "sum": sum,
    "round": round,
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
    "lower": lambda value: str(value).lower(),
    "upper": lambda value: str(value).upper(),
    "strip": lambda value: str(value).strip(),
    "contains": _fn_contains,
    "startswith": _fn_startswith,
    "endswith": _fn_endswith,
    "is_empty": _fn_is_empty,
    "default": _fn_default,
    "defined": _fn_defined_placeholder,
}


class SafeExpression:
    """A compiled, sandboxed expression evaluated against a results mapping.

    The expression is validated once at construction time, so an invalid or
    unsafe expression fails when the pipeline is loaded rather than midway
    through an expensive run.
    """

    def __init__(self, source: str) -> None:
        if not isinstance(source, str) or not source.strip():
            raise PipelineConditionError(
                "Condition expression must be a non-empty string."
            )
        if len(source) > MAX_EXPRESSION_LENGTH:
            raise PipelineConditionError(
                f"Condition expression is too long "
                f"({len(source)} > {MAX_EXPRESSION_LENGTH} characters)."
            )

        self.source = source
        self._called_names: set[str] = set()
        try:
            parsed = ast.parse(source, mode="eval")
        except SyntaxError as exc:
            raise PipelineConditionError(
                f"Could not parse condition expression {source!r}: {exc.msg}"
            ) from exc

        self._validate(parsed)
        self._tree = parsed
        self._code = compile(parsed, filename="<pipeline-condition>", mode="eval")

    # -- validation --------------------------------------------------------

    def _validate(self, tree: ast.Expression) -> None:
        node_count = 0
        mult_count = 0
        for node in ast.walk(tree):
            node_count += 1
            if node_count > MAX_EXPRESSION_NODES:
                raise PipelineConditionError(
                    f"Condition expression is too complex "
                    f"(over {MAX_EXPRESSION_NODES} nodes)."
                )

            reason = _REJECTED_NODES.get(type(node))
            if reason is not None:
                raise PipelineConditionError(
                    f"Condition expression {self.source!r} is not allowed: {reason}."
                )

            if not isinstance(node, _ALLOWED_NODES):
                raise PipelineConditionError(
                    f"Condition expression {self.source!r} is not allowed: "
                    f"{type(node).__name__} is not a permitted construct."
                )

            if isinstance(node, ast.Name) and node.id.startswith("__"):
                # Nothing legitimate is named this way, and rejecting it here
                # means the sandbox does not have to rely on the evaluation
                # namespace happening to shadow it.
                raise PipelineConditionError(
                    f"Condition expression {self.source!r} is not allowed: "
                    f"'{node.id}' is a reserved name."
                )

            if isinstance(node, ast.Call):
                self._validate_call(node)
            elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
                mult_count += 1
                if mult_count > 1:
                    # A per-operand cap is trivially chained around:
                    # ``'a' * 9999 * 9999`` allocates 100 MB while every
                    # individual literal stays under the limit. A condition
                    # has no legitimate need to multiply twice.
                    raise PipelineConditionError(
                        f"Condition expression {self.source!r} is not allowed: "
                        "at most one multiplication is permitted, because "
                        "chained multiplication can be used to exhaust memory."
                    )
                self._validate_multiplication(node)

    def _validate_multiplication(self, node: ast.BinOp) -> None:
        """Reject ``sequence * huge_int``, which would blow up memory."""

        for operand in (node.left, node.right):
            if (
                isinstance(operand, ast.Constant)
                and isinstance(operand.value, int)
                and not isinstance(operand.value, bool)
                and abs(operand.value) > MAX_SEQUENCE_REPEAT
            ):
                raise PipelineConditionError(
                    f"Condition expression {self.source!r} is not allowed: "
                    f"multiplication by {operand.value} exceeds the "
                    f"{MAX_SEQUENCE_REPEAT} repetition limit."
                )

    def _validate_call(self, node: ast.Call) -> None:
        if not isinstance(node.func, ast.Name):
            raise PipelineConditionError(
                f"Condition expression {self.source!r} may only call plain "
                "whitelisted functions."
            )
        if node.keywords:
            raise PipelineConditionError(
                f"Condition expression {self.source!r} may not use keyword arguments."
            )
        if node.func.id not in SAFE_FUNCTIONS:
            allowed = ", ".join(sorted(SAFE_FUNCTIONS))
            raise PipelineConditionError(
                f"Condition expression {self.source!r} calls unknown function "
                f"'{node.func.id}'. Allowed functions: {allowed}."
            )
        self._called_names.add(node.func.id)

    # -- introspection -----------------------------------------------------

    def referenced_names(self) -> set[str]:
        """Return the result keys this expression reads.

        Only names used as *values* count. A name used solely as a call target
        (``len`` in ``len(draft)``) is a helper, not a result key.
        """

        return {
            node.id
            for node in ast.walk(self._tree)
            if isinstance(node, ast.Name) and node.id not in self._called_names
        }

    # -- evaluation --------------------------------------------------------

    def evaluate(self, results: Mapping[str, Any]) -> Any:
        """Evaluate the expression against ``results`` and return its value.

        Names resolve lazily, so ``and`` / ``or`` and conditional expressions
        short-circuit properly: ``"has_score and score > 3"`` is well defined
        even when ``score`` does not exist. Reaching a name that is genuinely
        absent raises an error naming the results that do exist.
        """

        shadowed = sorted(
            set(results)
            & set(SAFE_FUNCTIONS)
            & (self.referenced_names() | self._called_names)
        )
        if shadowed:
            raise PipelineConditionError(
                f"Condition expression {self.source!r} refers to "
                f"{', '.join(shadowed)}, which is both a pipeline result and a "
                "built-in condition helper. Rename the step so the two do not "
                "collide."
            )

        helpers: dict[str, Any] = dict(SAFE_FUNCTIONS)
        helpers["defined"] = lambda name: str(name) in results

        namespace = _LazyNamespace(self.source, results, helpers)
        # ``__builtins__`` is emptied so that no builtin is reachable even if a
        # name were somehow to slip past validation.
        globals_: dict[str, Any] = dict(helpers)
        globals_["__builtins__"] = {}

        try:
            return eval(  # noqa: S307 - AST whitelisted above
                self._code, globals_, namespace
            )
        except PipelineConditionError:
            raise
        except Exception as exc:
            raise PipelineConditionError(
                f"Condition expression {self.source!r} failed to evaluate: {exc}"
            ) from exc

    def evaluate_bool(self, results: Mapping[str, Any]) -> bool:
        """Evaluate the expression and coerce the result to ``bool``."""

        return bool(self.evaluate(results))

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"SafeExpression({self.source!r})"


# --------------------------------------------------------------------------
# Conditions
# --------------------------------------------------------------------------


class Condition(abc.ABC):
    """A serializable boolean test over the pipeline results mapping."""

    #: The value written to, and dispatched on, the JSON ``kind`` field.
    kind: str = ""

    @abc.abstractmethod
    def evaluate(self, results: Mapping[str, Any]) -> bool:
        """Return the truth value of this condition for ``results``."""

    @abc.abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation of this condition."""

    @classmethod
    @abc.abstractmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Condition":
        """Restore a condition of this kind from its JSON representation."""

    def describe(self) -> str:
        """Return a short human-readable form, used in error messages."""

        return f"{type(self).__name__}()"


class ExpressionCondition(Condition):
    """A condition backed by a sandboxed expression over the results mapping.

    Example::

        ExpressionCondition("score > 0.8 and not is_empty(draft)")
    """

    kind = "expression"

    def __init__(self, expression: str | SafeExpression) -> None:
        self.expression = (
            expression
            if isinstance(expression, SafeExpression)
            else SafeExpression(expression)
        )

    @property
    def source(self) -> str:
        """Return the original expression source."""

        return self.expression.source

    def evaluate(self, results: Mapping[str, Any]) -> bool:
        return self.expression.evaluate_bool(results)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "expression": self.expression.source}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExpressionCondition":
        expression = data.get("expression", data.get("value"))
        if expression is None:
            raise PipelineConditionError(
                "Expression conditions require an 'expression' field."
            )
        return cls(str(expression))

    def describe(self) -> str:
        return self.expression.source

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"ExpressionCondition({self.expression.source!r})"


# --------------------------------------------------------------------------
# Registry and (de)serialization
# --------------------------------------------------------------------------

_CONDITION_KINDS: dict[str, type[Condition]] = {}


def register_condition_kind(condition_cls: type[Condition]) -> type[Condition]:
    """Register a ``Condition`` subclass so JSON can dispatch to it.

    Usable as a decorator. The class must set a non-empty ``kind``.
    """

    kind = getattr(condition_cls, "kind", "")
    if not kind:
        raise ValueError(
            f"{condition_cls.__name__} must define a non-empty 'kind' to be registered."
        )
    _CONDITION_KINDS[kind] = condition_cls
    return condition_cls


register_condition_kind(ExpressionCondition)


def condition_from_config(config: Condition | Mapping[str, Any] | str) -> Condition:
    """Build a ``Condition`` from JSON, a bare expression string, or itself.

    A plain string is treated as an expression, so ``"score > 0.8"`` and
    ``{"kind": "expression", "expression": "score > 0.8"}`` are equivalent.
    """

    if isinstance(config, Condition):
        return config
    if isinstance(config, str):
        return ExpressionCondition(config)
    if not isinstance(config, Mapping):
        raise PipelineConditionError(
            f"Condition config must be a string or object, got {type(config).__name__}."
        )

    kind = str(config.get("kind", ExpressionCondition.kind))
    condition_cls = _CONDITION_KINDS.get(kind)
    if condition_cls is None:
        known = ", ".join(sorted(_CONDITION_KINDS)) or "<none>"
        raise PipelineConditionError(
            f"Unknown condition kind: '{kind}'. Registered kinds: {known}."
        )
    return condition_cls.from_dict(config)


def condition_to_config(condition: Condition | None) -> dict[str, Any] | None:
    """Return the JSON form of ``condition``, or ``None`` when absent."""

    return None if condition is None else condition.to_dict()
