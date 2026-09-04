"""The prompt template engine: the Liquid subset PromptOn allows, and nothing else.

Two engines exist. ``liquid`` parses and renders; ``raw`` returns the source verbatim, for prompts
whose text genuinely contains ``{{`` or ``{%``.

The allowed subset is small on purpose, because every SDK has to reproduce it byte for byte:

===============  =========================================================================
Output           ``{{ var }}``, ``{{ a.b }}``, ``{{ a[0] }}``, with filters
Tags             ``for`` (``else``/``break``/``continue``/``forloop.*``), ``if``/``elsif``/
                 ``else``, ``unless``, ``assign``
Filters          ``size``, ``join``, ``default``
Rejected         ``include``, ``capture``, ``case``, ``raw``, ``comment``, ``cycle``,
                 ``render``, ``tablerow``, ``increment``, ``liquid`` - all parse errors
===============  =========================================================================

A variable is *missing* when its key is absent from the variables map. A key present with a
``None`` value is not missing: it renders as the empty string and ``default`` replaces it. Missing
is an error at output positions, in a ``for`` enumerable, in an ``unless`` condition and as an
``assign`` source. It is not checked in a branch that does not execute, and - matching the
reference implementation - not inside ``if``/``elsif`` conditions.

One Liquid rule that surprises people and that every SDK must reproduce: a **blank block body**
renders as nothing. When every entry of a block's body is blank (whitespace-only text, or an
``assign``), the text is dropped, so ``{% unless forloop.last %} {% endunless %}`` emits no space
at all while ``{% unless forloop.last %},{% endunless %}`` emits the comma.
"""

from __future__ import annotations

import contextlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from ._json import canonical_json
from .errors import MissingVariableError, RenderError, TemplateSyntaxError

__all__ = [
    "ALLOWED_FILTERS",
    "ALLOWED_TAGS",
    "Engine",
    "LintReason",
    "Template",
    "lint",
    "parse",
    "render",
    "render_messages",
    "to_output_string",
    "variables_of",
]

Engine = Literal["liquid", "raw"]

ALLOWED_TAGS = ("assign", "break", "continue", "for", "if", "unless")
ALLOWED_FILTERS = ("size", "join", "default")

_BLOCK_TAGS = frozenset(
    {
        "for",
        "endfor",
        "if",
        "endif",
        "elsif",
        "else",
        "unless",
        "endunless",
        "assign",
        "break",
        "continue",
    }
)

_BUILTIN_VARIABLES = frozenset({"forloop"})

_TOKEN_RE = re.compile(r"(\{\{-?.*?-?\}\}|\{%-?.*?-?%\})", re.DOTALL)
_TAG_NAME_RE = re.compile(r"\{%-?\s*([A-Za-z_][A-Za-z0-9_]*)")
_WS_CONTROL_RE = re.compile(r"\{\{-|\{%-|-\}\}|-%\}")


class _BreakSignal(Exception):
    """Raised by ``{% break %}``; caught by the enclosing ``for``."""


class _ContinueSignal(Exception):
    """Raised by ``{% continue %}``; caught by the enclosing ``for``."""


# ---------------------------------------------------------------------------
# expression tokens


@dataclass(frozen=True)
class _Tok:
    kind: str
    value: Any


_EXPR_SPEC = [
    ("SPACE", r"[ \t\r\n]+"),
    ("STRING", r'"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\''),
    ("NUMBER", r"-?\d+\.\d+|-?\d+"),
    ("OP", r"==|!=|>=|<=|>|<"),
    ("IDENT", r"[A-Za-z_][A-Za-z0-9_]*"),
    ("DOT", r"\."),
    ("LBRACKET", r"\["),
    ("RBRACKET", r"\]"),
    ("PIPE", r"\|"),
    ("COLON", r":"),
    ("COMMA", r","),
    ("EQUALS", r"="),
    ("OTHER", r"."),
]
_EXPR_RE = re.compile("|".join(f"(?P<{name}>{pattern})" for name, pattern in _EXPR_SPEC))


def _tokenize_expression(text: str) -> list[_Tok]:
    tokens: list[_Tok] = []
    for match in _EXPR_RE.finditer(text):
        kind = match.lastgroup or "OTHER"
        raw = match.group()
        if kind == "SPACE":
            continue
        if kind == "STRING":
            tokens.append(_Tok("STRING", _unquote(raw)))
        elif kind == "NUMBER":
            value: Any = float(raw) if "." in raw else int(raw)
            tokens.append(_Tok("NUMBER", value))
        else:
            tokens.append(_Tok(kind, raw))
    return tokens


def _unquote(raw: str) -> str:
    return re.sub(r"\\(.)", r"\1", raw[1:-1])


# ---------------------------------------------------------------------------
# expressions


class _Expr:
    def evaluate(self, ctx: _Context, strict: bool) -> Any:  # pragma: no cover - interface
        raise NotImplementedError

    def roots(self) -> list[str]:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass(frozen=True)
class _Literal(_Expr):
    value: Any

    def evaluate(self, ctx: _Context, strict: bool) -> Any:
        return self.value

    def roots(self) -> list[str]:
        return []


@dataclass(frozen=True)
class _Var(_Expr):
    root: str
    path: tuple[tuple[str, Any], ...] = ()

    def name_upto(self, depth: int) -> str:
        name = self.root
        for kind, key in self.path[:depth]:
            name += f".{key}" if kind == "key" else f"[{key}]"
        return name

    def evaluate(self, ctx: _Context, strict: bool) -> Any:
        if not ctx.has(self.root):
            if strict:
                raise MissingVariableError(self.name_upto(len(self.path)))
            return None
        value = ctx.get(self.root)
        for depth, (kind, key) in enumerate(self.path):
            value, found = _step(value, kind, key)
            if not found:
                if strict:
                    raise MissingVariableError(self.name_upto(depth + 1))
                return None
        return value

    def roots(self) -> list[str]:
        return [self.root]


@dataclass(frozen=True)
class _Filtered(_Expr):
    base: _Expr
    filters: tuple[tuple[str, tuple[_Expr, ...]], ...]

    def evaluate(self, ctx: _Context, strict: bool) -> Any:
        value = self.base.evaluate(ctx, strict)
        for name, args in self.filters:
            value = _apply_filter(name, value, [arg.evaluate(ctx, strict) for arg in args])
        return value

    def roots(self) -> list[str]:
        found = list(self.base.roots())
        for _name, args in self.filters:
            for arg in args:
                found.extend(arg.roots())
        return found


@dataclass(frozen=True)
class _Binary(_Expr):
    op: str
    left: _Expr
    right: _Expr

    def evaluate(self, ctx: _Context, strict: bool) -> Any:
        if self.op == "and":
            return _truthy(self.left.evaluate(ctx, strict)) and _truthy(
                self.right.evaluate(ctx, strict)
            )
        if self.op == "or":
            return _truthy(self.left.evaluate(ctx, strict)) or _truthy(
                self.right.evaluate(ctx, strict)
            )
        return _compare(self.op, self.left.evaluate(ctx, strict), self.right.evaluate(ctx, strict))

    def roots(self) -> list[str]:
        return self.left.roots() + self.right.roots()


@dataclass(frozen=True)
class _Truthiness(_Expr):
    """Wraps an expression so ``unless`` can negate its Liquid truthiness."""

    inner: _Expr

    def evaluate(self, ctx: _Context, strict: bool) -> Any:
        return _truthy(self.inner.evaluate(ctx, strict))

    def roots(self) -> list[str]:
        return self.inner.roots()


def _step(value: Any, kind: str, key: Any) -> tuple[Any, bool]:
    """One path segment. ``found`` False means the key is absent: a missing variable."""
    if isinstance(value, Mapping):
        return (value[key], True) if key in value else (None, False)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if kind == "index" and isinstance(key, int):
            return (value[key], True) if -len(value) <= key < len(value) else (None, True)
        if key == "size":
            return len(value), True
        if key == "first":
            return (value[0] if value else None), True
        if key == "last":
            return (value[-1] if value else None), True
        return None, True
    if isinstance(value, str) and key == "size":
        return len(value), True
    return None, True


def _compare(op: str, left: Any, right: Any) -> bool:
    if op == "==":
        return _eq(left, right)
    if op == "!=":
        return not _eq(left, right)
    if op == "contains":
        try:
            return right in left
        except TypeError:
            return False
    try:
        if op == ">":
            return bool(left > right)
        if op == "<":
            return bool(left < right)
        if op == ">=":
            return bool(left >= right)
        if op == "<=":
            return bool(left <= right)
    except TypeError:
        return False
    raise RenderError(f"unknown operator: {op}")


def _eq(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    return bool(left == right)


def _truthy(value: Any) -> bool:
    """Liquid truthiness: only ``nil`` and ``false`` are falsy.

    ``""``, ``0`` and ``[]`` are all true.
    """
    return not (value is None or value is False)


def _blank(value: Any) -> bool:
    return value is None or value is False or value == "" or value == [] or value == {}


def to_output_string(value: Any) -> str:
    """Render a value into an output position, following Liquid's rules.

    Integers lose their decimal point and floats keep one (``2.0`` renders as ``"2.0"``); ``None``
    renders as the empty string; a list renders as its elements concatenated with no separator,
    which is the Liquid rule and almost never what you want - use ``join``.
    """
    if value is None:
        return ""
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, Mapping):
        return canonical_json(value).decode("utf-8")
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return "".join(to_output_string(item) for item in value)
    return str(value)


def _apply_filter(name: str, value: Any, args: list[Any]) -> Any:
    if name == "size":
        if value is None:
            return 0
        if isinstance(value, (str, bytes, Mapping, Sequence)):
            return len(value)
        return 0
    if name == "join":
        separator = to_output_string(args[0]) if args else " "
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return separator.join(to_output_string(item) for item in value)
        return to_output_string(value)
    if name == "default":
        fallback = args[0] if args else None
        return fallback if _blank(value) else value
    raise RenderError(f"filter not allowed: {name} (allowed: {', '.join(ALLOWED_FILTERS)})")


# ---------------------------------------------------------------------------
# nodes


class _Node:
    def render(self, ctx: _Context, out: list[str]) -> None:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass
class _Text(_Node):
    text: str

    def render(self, ctx: _Context, out: list[str]) -> None:
        out.append(self.text)


@dataclass
class _Output(_Node):
    expr: _Expr

    def render(self, ctx: _Context, out: list[str]) -> None:
        out.append(to_output_string(self.expr.evaluate(ctx, strict=True)))


@dataclass
class _If(_Node):
    branches: list[tuple[_Expr, list[_Node]]]
    otherwise: list[_Node] = field(default_factory=list)
    strict_condition: bool = False

    def render(self, ctx: _Context, out: list[str]) -> None:
        for condition, body in self.branches:
            if _truthy(condition.evaluate(ctx, strict=self.strict_condition)):
                _render_all(body, ctx, out)
                return
        _render_all(self.otherwise, ctx, out)


@dataclass
class _For(_Node):
    variable: str
    enumerable: _Expr
    body: list[_Node]
    otherwise: list[_Node] = field(default_factory=list)

    def render(self, ctx: _Context, out: list[str]) -> None:
        items = _as_iterable(self.enumerable.evaluate(ctx, strict=True))
        if not items:
            _render_all(self.otherwise, ctx, out)
            return
        total = len(items)
        for index, item in enumerate(items):
            ctx.push(
                {
                    self.variable: item,
                    "forloop": {
                        "index": index + 1,
                        "index0": index,
                        "rindex": total - index,
                        "rindex0": total - index - 1,
                        "first": index == 0,
                        "last": index == total - 1,
                        "length": total,
                    },
                }
            )
            try:
                _render_all(self.body, ctx, out)
            except _ContinueSignal:
                continue
            except _BreakSignal:
                return
            finally:
                ctx.pop()


@dataclass
class _Assign(_Node):
    target: str
    expr: _Expr

    def render(self, ctx: _Context, out: list[str]) -> None:
        ctx.assign(self.target, self.expr.evaluate(ctx, strict=True))


@dataclass
class _BreakNode(_Node):
    def render(self, ctx: _Context, out: list[str]) -> None:
        raise _BreakSignal


@dataclass
class _ContinueNode(_Node):
    def render(self, ctx: _Context, out: list[str]) -> None:
        raise _ContinueSignal


def _as_iterable(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        return [[key, item] for key, item in value.items()]
    if isinstance(value, (str, bytes)):
        return [value]
    if isinstance(value, Sequence):
        return list(value)
    return []


def _render_all(nodes: list[_Node], ctx: _Context, out: list[str]) -> None:
    for node in nodes:
        node.render(ctx, out)


def _is_blank_node(node: _Node) -> bool:
    if isinstance(node, _Text):
        return node.text.strip() == ""
    return isinstance(node, _Assign)


def _strip_blank_body(nodes: list[_Node]) -> list[_Node]:
    """Liquid's "blank body" rule: when every entry is blank, the text is dropped."""
    if nodes and all(_is_blank_node(node) for node in nodes):
        return [node for node in nodes if not isinstance(node, _Text)]
    return nodes


# ---------------------------------------------------------------------------
# context


class _Context:
    __slots__ = ("scopes",)

    def __init__(self, variables: Mapping[str, Any] | None) -> None:
        self.scopes: list[dict[str, Any]] = [dict(variables or {}), {}]

    def has(self, name: str) -> bool:
        return any(name in scope for scope in reversed(self.scopes))

    def get(self, name: str) -> Any:
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        return None

    def assign(self, name: str, value: Any) -> None:
        self.scopes[1][name] = value

    def push(self, scope: dict[str, Any]) -> None:
        self.scopes.append(scope)

    def pop(self) -> None:
        self.scopes.pop()


# ---------------------------------------------------------------------------
# chunking


@dataclass
class _Chunk:
    kind: str  # text | output | tag
    body: str
    left_trim: bool = False
    right_trim: bool = False


def _chunks(source: str) -> list[_Chunk]:
    chunks: list[_Chunk] = []
    for piece in _TOKEN_RE.split(source):
        if not piece:
            continue
        if piece.startswith("{{") and piece.endswith("}}"):
            inner = piece[2:-2]
            left, right = inner.startswith("-"), inner.endswith("-")
            chunks.append(_Chunk("output", inner.strip("-").strip(), left, right))
        elif piece.startswith("{%") and piece.endswith("%}"):
            inner = piece[2:-2]
            left, right = inner.startswith("-"), inner.endswith("-")
            chunks.append(_Chunk("tag", inner.strip("-").strip(), left, right))
        else:
            chunks.append(_Chunk("text", piece))
    _apply_whitespace_control(chunks)
    return chunks


def _apply_whitespace_control(chunks: list[_Chunk]) -> None:
    """``{%-``/``-%}`` trim the adjacent text. Lint rejects them, the renderer honours them."""
    for index, chunk in enumerate(chunks):
        if chunk.kind == "text":
            continue
        if chunk.left_trim and index > 0 and chunks[index - 1].kind == "text":
            chunks[index - 1].body = chunks[index - 1].body.rstrip()
        if chunk.right_trim and index + 1 < len(chunks) and chunks[index + 1].kind == "text":
            chunks[index + 1].body = chunks[index + 1].body.lstrip()


def _tag_name(body: str) -> str:
    match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)", body.strip())
    return match.group(1) if match else body.strip()


# ---------------------------------------------------------------------------
# template


class Template:
    """A parsed template. Immutable once built, and safe to reuse across threads."""

    __slots__ = ("nodes", "source")

    def __init__(self, nodes: list[_Node], source: str) -> None:
        self.nodes = nodes
        self.source = source

    def render(self, variables: Mapping[str, Any] | None = None) -> str:
        """Render with ``variables``. Raises :class:`~prompton.errors.MissingVariableError`."""
        ctx = _Context(variables)
        out: list[str] = []
        with contextlib.suppress(_BreakSignal, _ContinueSignal):
            _render_all(self.nodes, ctx, out)
        return "".join(out)

    def variables(self) -> list[str]:
        """Top-level input variables this template reads, sorted and deduplicated."""
        referenced: list[str] = []
        bound: set[str] = set()
        _collect_variables(self.nodes, referenced, bound)
        return sorted(
            {name for name in referenced if name not in bound and name not in _BUILTIN_VARIABLES}
        )

    def filters(self) -> list[str]:
        """Every filter name used, in order of first appearance."""
        found: list[str] = []
        _collect_filters(self.nodes, found)
        ordered: list[str] = []
        for name in found:
            if name not in ordered:
                ordered.append(name)
        return ordered


class _Parser:
    def __init__(self, source: str) -> None:
        self.source = source
        self.chunks = _chunks(source)
        self.index = 0

    def parse(self) -> Template:
        return Template(self._parse_until(frozenset()), self.source)

    def _parse_until(self, stops: frozenset[str]) -> list[_Node]:
        nodes: list[_Node] = []
        while self.index < len(self.chunks):
            chunk = self.chunks[self.index]
            if chunk.kind == "text":
                self.index += 1
                nodes.append(_Text(chunk.body))
                continue
            if chunk.kind == "output":
                self.index += 1
                nodes.append(_Output(_parse_value(chunk.body)))
                continue
            name = _tag_name(chunk.body)
            if name in stops:
                return nodes
            if name not in _BLOCK_TAGS:
                raise TemplateSyntaxError(f"Unexpected tag '{name}'")
            self.index += 1
            nodes.append(self._parse_tag(name, chunk.body))
        return nodes

    def _parse_tag(self, name: str, body: str) -> _Node:
        rest = body[len(name) :].strip()
        if name == "if":
            return self._parse_if(rest)
        if name == "unless":
            return self._parse_unless(rest)
        if name == "for":
            return self._parse_for(rest)
        if name == "assign":
            return _parse_assign(rest)
        if name == "break":
            return _BreakNode()
        if name == "continue":
            return _ContinueNode()
        raise TemplateSyntaxError(f"Unexpected tag '{name}'")

    _IF_STOPS = frozenset({"elsif", "else", "endif"})

    def _parse_if(self, condition_text: str) -> _If:
        branches = [(_parse_condition(condition_text), self._parse_until(self._IF_STOPS))]
        otherwise: list[_Node] = []
        while True:
            chunk = self._expect(self._IF_STOPS, "endif")
            name = _tag_name(chunk.body)
            self.index += 1
            if name == "endif":
                break
            if name == "elsif":
                condition = _parse_condition(chunk.body[len("elsif") :].strip())
                branches.append((condition, self._parse_until(self._IF_STOPS)))
                continue
            otherwise = self._parse_until(frozenset({"endif"}))
        return _If(
            [(condition, _strip_blank_body(body)) for condition, body in branches],
            _strip_blank_body(otherwise),
            strict_condition=False,
        )

    def _parse_unless(self, condition_text: str) -> _If:
        condition = _parse_condition(condition_text)
        body = self._parse_until(frozenset({"else", "endunless"}))
        otherwise: list[_Node] = []
        chunk = self._expect(frozenset({"else", "endunless"}), "endunless")
        self.index += 1
        if _tag_name(chunk.body) == "else":
            otherwise = self._parse_until(frozenset({"endunless"}))
            self._expect(frozenset({"endunless"}), "endunless")
            self.index += 1
        # `unless cond` is `if not cond`. Unlike `if`, the reference reports a missing variable
        # read by the condition, so this branch evaluates strictly.
        negated = _Binary("==", _Truthiness(condition), _Literal(False))
        return _If(
            [(negated, _strip_blank_body(body))],
            _strip_blank_body(otherwise),
            strict_condition=True,
        )

    def _parse_for(self, rest: str) -> _For:
        match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s+in\s+(.+)$", rest, re.DOTALL)
        if not match:
            raise TemplateSyntaxError(f"malformed for tag: {rest!r}")
        variable, enumerable = match.group(1), match.group(2).strip()
        body = self._parse_until(frozenset({"else", "endfor"}))
        otherwise: list[_Node] = []
        chunk = self._expect(frozenset({"else", "endfor"}), "endfor")
        self.index += 1
        if _tag_name(chunk.body) == "else":
            otherwise = self._parse_until(frozenset({"endfor"}))
            self._expect(frozenset({"endfor"}), "endfor")
            self.index += 1
        return _For(
            variable,
            _parse_value(enumerable),
            _strip_blank_body(body),
            _strip_blank_body(otherwise),
        )

    def _expect(self, names: frozenset[str], expected: str) -> _Chunk:
        if self.index >= len(self.chunks):
            raise TemplateSyntaxError(f"Expected '{expected}'")
        chunk = self.chunks[self.index]
        if chunk.kind != "tag" or _tag_name(chunk.body) not in names:
            raise TemplateSyntaxError(f"Expected '{expected}'")
        return chunk


# ---------------------------------------------------------------------------
# expression parsing

_KEYWORD_LITERALS: dict[str, Any] = {
    "true": True,
    "false": False,
    "nil": None,
    "null": None,
    "empty": "",
    "blank": "",
}


def _parse_value(text: str) -> _Expr:
    tokens = _tokenize_expression(text)
    if not tokens:
        raise TemplateSyntaxError("empty expression")
    expr, index = _parse_filtered(tokens, 0)
    if index != len(tokens):
        raise TemplateSyntaxError(f"unexpected token in expression: {text!r}")
    return expr


def _parse_condition(text: str) -> _Expr:
    tokens = _tokenize_expression(text)
    if not tokens:
        raise TemplateSyntaxError("empty condition")
    expr, index = _parse_bool(tokens, 0)
    if index != len(tokens):
        raise TemplateSyntaxError(f"unexpected token in condition: {text!r}")
    return expr


def _parse_bool(tokens: list[_Tok], index: int) -> tuple[_Expr, int]:
    left, index = _parse_comparison(tokens, index)
    if (
        index < len(tokens)
        and tokens[index].kind == "IDENT"
        and tokens[index].value in {"and", "or"}
    ):
        operator = tokens[index].value
        # Liquid has no precedence between and/or: it evaluates right to left.
        right, index = _parse_bool(tokens, index + 1)
        return _Binary(operator, left, right), index
    return left, index


def _parse_comparison(tokens: list[_Tok], index: int) -> tuple[_Expr, int]:
    left, index = _parse_filtered(tokens, index)
    if index < len(tokens):
        token = tokens[index]
        if token.kind == "OP":
            right, index = _parse_filtered(tokens, index + 1)
            return _Binary(token.value, left, right), index
        if token.kind == "IDENT" and token.value == "contains":
            right, index = _parse_filtered(tokens, index + 1)
            return _Binary("contains", left, right), index
    return left, index


def _parse_filtered(tokens: list[_Tok], index: int) -> tuple[_Expr, int]:
    base, index = _parse_primary(tokens, index)
    filters: list[tuple[str, tuple[_Expr, ...]]] = []
    while index < len(tokens) and tokens[index].kind == "PIPE":
        index += 1
        if index >= len(tokens) or tokens[index].kind != "IDENT":
            raise TemplateSyntaxError("expected a filter name after '|'")
        name = tokens[index].value
        index += 1
        args: list[_Expr] = []
        if index < len(tokens) and tokens[index].kind == "COLON":
            index += 1
            while True:
                arg, index = _parse_primary(tokens, index)
                args.append(arg)
                if index < len(tokens) and tokens[index].kind == "COMMA":
                    index += 1
                    continue
                break
        filters.append((name, tuple(args)))
    if filters:
        return _Filtered(base, tuple(filters)), index
    return base, index


def _parse_primary(tokens: list[_Tok], index: int) -> tuple[_Expr, int]:
    if index >= len(tokens):
        raise TemplateSyntaxError("unexpected end of expression")
    token = tokens[index]
    if token.kind in ("STRING", "NUMBER"):
        return _Literal(token.value), index + 1
    if token.kind != "IDENT":
        raise TemplateSyntaxError(f"unexpected token {token.value!r} in expression")
    if token.value in _KEYWORD_LITERALS:
        return _Literal(_KEYWORD_LITERALS[token.value]), index + 1
    root = token.value
    index += 1
    path: list[tuple[str, Any]] = []
    while index < len(tokens):
        token = tokens[index]
        if token.kind == "DOT":
            if index + 1 >= len(tokens) or tokens[index + 1].kind != "IDENT":
                raise TemplateSyntaxError("expected a property name after '.'")
            path.append(("key", tokens[index + 1].value))
            index += 2
            continue
        if token.kind == "LBRACKET":
            if index + 1 >= len(tokens) or tokens[index + 1].kind not in ("NUMBER", "STRING"):
                raise TemplateSyntaxError("expected a number or string inside '[]'")
            key = tokens[index + 1].value
            if index + 2 >= len(tokens) or tokens[index + 2].kind != "RBRACKET":
                raise TemplateSyntaxError("expected ']'")
            path.append(("index" if isinstance(key, int) else "key", key))
            index += 3
            continue
        break
    return _Var(root, tuple(path)), index


def _parse_assign(rest: str) -> _Assign:
    match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+)$", rest, re.DOTALL)
    if not match:
        raise TemplateSyntaxError(f"malformed assign tag: {rest!r}")
    return _Assign(match.group(1), _parse_value(match.group(2).strip()))


# ---------------------------------------------------------------------------
# AST walking


def _collect_variables(nodes: list[_Node], referenced: list[str], bound: set[str]) -> None:
    for node in nodes:
        if isinstance(node, _Output):
            referenced.extend(node.expr.roots())
        elif isinstance(node, _If):
            for condition, body in node.branches:
                referenced.extend(condition.roots())
                _collect_variables(body, referenced, bound)
            _collect_variables(node.otherwise, referenced, bound)
        elif isinstance(node, _For):
            referenced.extend(node.enumerable.roots())
            bound.add(node.variable)
            _collect_variables(node.body, referenced, bound)
            _collect_variables(node.otherwise, referenced, bound)
        elif isinstance(node, _Assign):
            referenced.extend(node.expr.roots())
            bound.add(node.target)


def _collect_filters(nodes: list[_Node], found: list[str]) -> None:
    for node in nodes:
        if isinstance(node, _Output):
            _walk_expr_filters(node.expr, found)
        elif isinstance(node, _If):
            for condition, body in node.branches:
                _walk_expr_filters(condition, found)
                _collect_filters(body, found)
            _collect_filters(node.otherwise, found)
        elif isinstance(node, _For):
            _walk_expr_filters(node.enumerable, found)
            _collect_filters(node.body, found)
            _collect_filters(node.otherwise, found)
        elif isinstance(node, _Assign):
            _walk_expr_filters(node.expr, found)


def _walk_expr_filters(expr: _Expr, found: list[str]) -> None:
    if isinstance(expr, _Filtered):
        _walk_expr_filters(expr.base, found)
        for name, args in expr.filters:
            found.append(name)
            for arg in args:
                _walk_expr_filters(arg, found)
    elif isinstance(expr, _Binary):
        _walk_expr_filters(expr.left, found)
        _walk_expr_filters(expr.right, found)
    elif isinstance(expr, _Truthiness):
        _walk_expr_filters(expr.inner, found)


# ---------------------------------------------------------------------------
# public API


def parse(source: str) -> Template:
    """Parse a template. Raises :class:`~prompton.errors.TemplateSyntaxError`."""
    return _Parser(source).parse()


def render(
    source: str | Template,
    variables: Mapping[str, Any] | None = None,
    engine: Engine = "liquid",
) -> str:
    """Render ``source``. ``engine="raw"`` returns it verbatim without parsing."""
    if engine == "raw":
        return source.source if isinstance(source, Template) else source
    template = source if isinstance(source, Template) else parse(source)
    return template.render(variables)


def render_messages(
    messages: Sequence[Mapping[str, Any]],
    variables: Mapping[str, Any] | None = None,
    engine: Engine = "liquid",
) -> list[dict[str, Any]]:
    """Render the ``content`` of every message. Other keys (``role``, ``name``) pass through."""
    rendered: list[dict[str, Any]] = []
    for message in messages:
        item = dict(message)
        item["content"] = render(str(message.get("content") or ""), variables, engine)
        rendered.append(item)
    return rendered


def variables_of(source: str) -> list[str]:
    """The top-level input variables a template reads.

    Falls back to a conservative regex scrape when the template does not parse.
    """
    try:
        return parse(source).variables()
    except TemplateSyntaxError:
        return _scrape_variables(source)


def _scrape_variables(source: str) -> list[str]:
    names: set[str] = set()
    for match in re.finditer(r"\{\{-?\s*([A-Za-z_][A-Za-z0-9_]*)", source):
        names.add(match.group(1))
    for match in re.finditer(
        r"\{%-?\s*(?:if|unless|elsif)\s+([A-Za-z_][A-Za-z0-9_]*)"
        r"|\{%-?\s*for\s+\w+\s+in\s+([A-Za-z_][A-Za-z0-9_]*)",
        source,
    ):
        names.add(match.group(1) or match.group(2))
    names -= set(_BUILTIN_VARIABLES) | {"true", "false", "nil", "empty", "blank"}
    return sorted(names)


@dataclass(frozen=True)
class LintReason:
    """One reason a template fails the whitelist check."""

    kind: str  # whitespace_control | disallowed_tag | disallowed_filter | parse
    value: str

    def as_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "value": self.value}


def lint(source: str) -> list[LintReason]:
    """Static whitelist check. An empty list means the template is allowed.

    The server runs the same check when a prompt version is committed, so a template that fails
    lint can never reach a snapshot - which is why the renderer does not repeat the check.
    """
    reasons: list[LintReason] = list(_whitespace_reasons(source))

    disallowed_tags = _disallowed_tags(source)
    if disallowed_tags:
        reasons.extend(LintReason("disallowed_tag", name) for name in disallowed_tags)
        return _dedupe(reasons)

    try:
        template = parse(source)
    except TemplateSyntaxError as error:
        reasons.append(LintReason("parse", str(error)))
        return _dedupe(reasons)

    reasons.extend(
        LintReason("disallowed_filter", name)
        for name in template.filters()
        if name not in ALLOWED_FILTERS
    )
    return _dedupe(reasons)


def _dedupe(reasons: list[LintReason]) -> list[LintReason]:
    ordered: list[LintReason] = []
    for reason in reasons:
        if reason not in ordered:
            ordered.append(reason)
    return ordered


def _whitespace_reasons(source: str) -> list[LintReason]:
    markers: list[str] = []
    for match in _WS_CONTROL_RE.finditer(source):
        if match.group() not in markers:
            markers.append(match.group())
    return [
        LintReason("whitespace_control", marker)
        for marker in markers
        if marker in ("{{-", "{%-") or _inside_block(source, marker)
    ]


def _inside_block(source: str, marker: str) -> bool:
    pattern = re.compile(r"(\{\{|\{%)(?:(?!\}\}|%\}).)*?" + re.escape(marker), re.DOTALL)
    return bool(pattern.search(source))


def _disallowed_tags(source: str) -> list[str]:
    found: list[str] = []
    for match in _TAG_NAME_RE.finditer(source):
        name = match.group(1)
        if name not in _BLOCK_TAGS and name not in found:
            found.append(name)
    return found
