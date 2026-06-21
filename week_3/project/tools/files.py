"""
tools/files.py — OpenCode-style sandboxed file tools (Lesson 2)
====================================================================
Conventions, all deliberate (see week_3/2_agent_class.md for the why):
  - resolve_path()  sandboxes every path inside WORKSPACE_ROOT.
  - read_file()     returns numbered lines, supports start_line/read_lines
                     pagination, reports has_more and total_lines.
  - write_file()    creates or overwrites a whole file.
  - list_files()    glob-based directory listing.
  - edit_file()     line-based replace / delete / append, returns a diff
                     preview so mistakes are visible in the tool log.
  - Every function returns {"content": ...} or {"error": ...}. Never raises
    out to the caller — the model should see structured errors, not tracebacks.
"""

import os
from glob import glob as _glob
from pathlib import Path

WORKSPACE_ROOT = Path(os.environ.get("WORKSPACE_ROOT", ".")).resolve()

DEFAULT_READ_LINES = 200
MAX_FILE_CHARS = 12_000  # safety-net truncation even when read_lines is generous


# ---------------------------------------------------------------------------
# Sandbox
# ---------------------------------------------------------------------------
def resolve_path(path: str) -> Path:
    """
    Resolve `path` relative to WORKSPACE_ROOT and verify it doesn't escape
    the sandbox. Raises ValueError on escape attempts — callers catch this
    and turn it into a structured {"error": ...} response.
    """
    candidate = (WORKSPACE_ROOT / path).resolve()
    try:
        candidate.relative_to(WORKSPACE_ROOT)
    except ValueError:
        raise ValueError(f"Path '{path}' escapes workspace root '{WORKSPACE_ROOT}'")
    return candidate


# ---------------------------------------------------------------------------
# read_file — numbered lines, pagination, truncation
# ---------------------------------------------------------------------------
def read_file(path: str, start_line: int = 1, read_lines: int = DEFAULT_READ_LINES) -> dict:
    """
    Read a window of lines from a file, prefixed with line numbers so the
    model can pass them straight to edit_file.

    Returns:
      {"content": "  1\\t...\\n  2\\t...", "start_line": int, "end_line": int,
       "total_lines": int, "has_more": bool}
      or {"error": str}
    """
    try:
        full_path = resolve_path(path)
    except ValueError as exc:
        return {"error": str(exc)}

    if not full_path.is_file():
        return {"error": f"File not found: {path}"}
    if start_line < 1:
        return {"error": "start_line must be >= 1"}
    if read_lines < 1:
        return {"error": "read_lines must be >= 1"}

    try:
        all_lines = full_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as exc:
        return {"error": f"Could not read {path}: {exc}"}

    total_lines = len(all_lines)
    if total_lines == 0:
        return {
            "content": "",
            "start_line": 1,
            "end_line": 0,
            "total_lines": 0,
            "has_more": False,
            "note": "File is empty.",
        }
    if start_line > total_lines:
        return {"error": f"start_line {start_line} is beyond end of file ({total_lines} lines total)"}

    end_line = min(start_line + read_lines - 1, total_lines)
    window = all_lines[start_line - 1:end_line]

    width = len(str(end_line))
    numbered = "\n".join(f"{str(start_line + i).rjust(width)}\t{line}" for i, line in enumerate(window))

    truncated = False
    if len(numbered) > MAX_FILE_CHARS:
        numbered = numbered[:MAX_FILE_CHARS] + "\n[...truncated — narrow start_line/read_lines]"
        truncated = True

    return {
        "content": numbered,
        "start_line": start_line,
        "end_line": end_line,
        "total_lines": total_lines,
        "has_more": end_line < total_lines,
        "truncated": truncated,
    }


# ---------------------------------------------------------------------------
# write_file — create or overwrite
# ---------------------------------------------------------------------------
def write_file(path: str, content: str) -> dict:
    """
    Create or fully overwrite a file with `content`. Creates parent
    directories as needed. Returns {"content": "Wrote N lines to <path>"}.
    """
    try:
        full_path = resolve_path(path)
    except ValueError as exc:
        return {"error": str(exc)}

    try:
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding="utf-8")
    except Exception as exc:
        return {"error": f"Could not write {path}: {exc}"}

    n_lines = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
    return {"content": f"Wrote {n_lines} lines to {path}"}


# ---------------------------------------------------------------------------
# list_files — glob
# ---------------------------------------------------------------------------
def list_files(pattern: str = "**/*", path: str = ".") -> dict:
    """
    Glob for files under `path` (relative to WORKSPACE_ROOT) matching
    `pattern` (default: everything, recursive). Returns relative paths.
    """
    try:
        base = resolve_path(path)
    except ValueError as exc:
        return {"error": str(exc)}

    if not base.is_dir():
        return {"error": f"Not a directory: {path}"}

    try:
        matches = sorted(base.glob(pattern))
    except Exception as exc:
        return {"error": f"Bad glob pattern '{pattern}': {exc}"}

    files = [str(m.relative_to(WORKSPACE_ROOT)) for m in matches if m.is_file()]

    if not files:
        return {"content": [], "note": f"No files matched '{pattern}' under {path}"}
    return {"content": files}


# ---------------------------------------------------------------------------
# edit_file — line-based replace / delete / append, with diff preview
# ---------------------------------------------------------------------------
def edit_file(
    path: str,
    operation: str,
    start_line: int,
    content: str | None = None,
    end_line: int | None = None,
) -> dict:
    """
    Line-level edit. operation in {"replace", "delete", "append"}.

      replace: lines start_line..end_line (inclusive) -> content.split("\\n")
      delete:  remove lines start_line..end_line (inclusive)
      append:  insert content's lines after start_line (0 = before line 1)

    Returns {"content": "<diff preview>"} or {"error": str}.
    Always read_file() immediately before calling this — line numbers must
    be fresh, or the edit can land on the wrong lines.
    """
    try:
        full_path = resolve_path(path)
    except ValueError as exc:
        return {"error": str(exc)}

    if not full_path.is_file():
        return {"error": f"File not found: {path}"}
    if operation not in ("replace", "delete", "append"):
        return {"error": f"Unknown operation '{operation}' — must be replace, delete, or append"}

    try:
        lines = full_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as exc:
        return {"error": f"Could not read {path}: {exc}"}

    total = len(lines)
    diff_lines: list[str] = []

    if operation == "replace":
        if end_line is None:
            return {"error": "replace requires end_line"}
        if not (1 <= start_line <= end_line <= total):
            return {"error": f"Invalid range {start_line}-{end_line} for file with {total} lines"}
        if content is None:
            return {"error": "replace requires content"}

        old_block = lines[start_line - 1:end_line]
        new_block = content.split("\n")
        diff_lines += [f"-{ln}" for ln in old_block]
        diff_lines += [f"+{ln}" for ln in new_block]
        lines[start_line - 1:end_line] = new_block

    elif operation == "delete":
        if end_line is None:
            return {"error": "delete requires end_line"}
        if not (1 <= start_line <= end_line <= total):
            return {"error": f"Invalid range {start_line}-{end_line} for file with {total} lines"}

        old_block = lines[start_line - 1:end_line]
        diff_lines += [f"-{ln}" for ln in old_block]
        del lines[start_line - 1:end_line]

    elif operation == "append":
        if content is None:
            return {"error": "append requires content"}
        if not (0 <= start_line <= total):
            return {"error": f"Invalid start_line {start_line} for file with {total} lines (0 = before line 1)"}

        new_block = content.split("\n")
        diff_lines += [f"+{ln}" for ln in new_block]
        lines[start_line:start_line] = new_block

    try:
        full_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    except Exception as exc:
        return {"error": f"Could not write {path}: {exc}"}

    preview = "\n".join(diff_lines[:40])
    if len(diff_lines) > 40:
        preview += f"\n[...{len(diff_lines) - 40} more diff lines omitted]"

    return {
        "content": f"{operation} applied to {path} (lines {start_line}-{end_line or start_line}).\n\nDiff:\n{preview}",
        "new_total_lines": len(lines),
    }


# ---------------------------------------------------------------------------
# OpenAI tool schemas
# ---------------------------------------------------------------------------
FILE_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read a file from the sandboxed workspace, with line numbers. "
                "Use start_line/read_lines to page through large files — check "
                "has_more in the response and read the next window if needed. "
                "Always re-read immediately before edit_file to get fresh line numbers."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to the workspace root"},
                    "start_line": {"type": "integer", "description": "First line to read, 1-indexed (default 1)"},
                    "read_lines": {"type": "integer", "description": "How many lines to return (default 200)"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Create a new file or fully overwrite an existing one. "
                "Use for new notes; use edit_file to update existing notes instead "
                "of rewriting the whole file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to the workspace root, e.g. notes/topic.md"},
                    "content": {"type": "string", "description": "Full file content"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": (
                "List files under a directory using a glob pattern "
                "(e.g. '**/*.md' for all markdown files recursively, '*' for "
                "the top level). Use this to explore notes/ before reading."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory to search, relative to workspace root (default '.')"},
                    "pattern": {"type": "string", "description": "Glob pattern (default '**/*')"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": (
                "Make a surgical line-level edit to an existing file. "
                "operation='replace' needs start_line, end_line, content. "
                "operation='delete' needs start_line, end_line. "
                "operation='append' needs start_line (0 = before line 1) and content. "
                "Always call read_file right before this to get current line numbers — "
                "the response includes a diff preview so you can verify the change."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to the workspace root"},
                    "operation": {"type": "string", "enum": ["replace", "delete", "append"]},
                    "start_line": {"type": "integer", "description": "1-indexed start line (0 allowed for append)"},
                    "end_line": {"type": "integer", "description": "1-indexed end line, inclusive (required for replace/delete)"},
                    "content": {"type": "string", "description": "New content (required for replace/append), newline-separated"},
                },
                "required": ["path", "operation", "start_line"],
            },
        },
    },
]
