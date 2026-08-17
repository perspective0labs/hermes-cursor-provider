"""Bridge Cursor exec requests to Hermes tool execution."""

from __future__ import annotations

import json
import shlex
from pathlib import PurePosixPath
from typing import Any

from cursor_backend.proto import agent_pb2


def _parse_jsonish(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except Exception:
            return {"content": value}
        return parsed if isinstance(parsed, dict) else {"content": parsed}
    return {"content": value}


def _tool_text(result: dict[str, Any]) -> str:
    if isinstance(result.get("content"), str):
        return result["content"]
    if isinstance(result.get("output"), str):
        return result["output"]
    content = result.get("content")
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                chunks.append(str(item.get("text", "")))
        if chunks:
            return "".join(chunks)
    matches = result.get("matches")
    if isinstance(matches, list):
        lines: list[str] = []
        for match in matches:
            if not isinstance(match, dict):
                lines.append(str(match))
                continue
            path = match.get("path", "")
            line_number = match.get("line_number") or match.get("line")
            content = match.get("content", "")
            if path and line_number is not None:
                lines.append(f"{path}:{line_number}:{content}")
            else:
                lines.append(str(content))
        return "\n".join(lines)
    files = result.get("files")
    if isinstance(files, list):
        return "\n".join(str(item) for item in files)
    counts = result.get("counts")
    if isinstance(counts, dict):
        return "\n".join(f"{path}:{count}" for path, count in counts.items())
    error = result.get("error")
    if error is not None:
        return str(error)
    return ""


def _decode_mcp_arg_value(raw: bytes) -> Any:
    text = raw.decode("utf-8")
    normalized = (
        text.replace("None", "null")
        .replace("True", "true")
        .replace("False", "false")
    )
    try:
        return json.loads(normalized)
    except Exception:
        return text


def decode_mcp_args(mcp_args: agent_pb2.McpArgs) -> dict[str, Any]:
    return {
        key: _decode_mcp_arg_value(value)
        for key, value in dict(mcp_args.args).items()
    }


def build_read_result_from_tool_result(path: str, tool_result: Any) -> agent_pb2.ReadResult:
    payload = _parse_jsonish(tool_result)
    if payload.get("error"):
        return agent_pb2.ReadResult(
            error=agent_pb2.ReadError(path=path, error=str(payload["error"]))
        )

    content = _tool_text(payload)
    total_lines = payload.get("total_lines")
    if not isinstance(total_lines, int):
        total_lines = len(content.splitlines()) if content else 0
    return agent_pb2.ReadResult(
        success=agent_pb2.ReadSuccess(
            path=path,
            total_lines=total_lines,
            file_size=payload.get("file_size", len(content.encode("utf-8"))),
            truncated=bool(payload.get("truncated", False)),
            content=content,
        )
    )


def build_write_result_from_tool_result(
    args: agent_pb2.WriteArgs, tool_result: Any
) -> agent_pb2.WriteResult:
    payload = _parse_jsonish(tool_result)
    if payload.get("error"):
        return agent_pb2.WriteResult(
            error=agent_pb2.WriteError(path=args.path, error=str(payload["error"]))
        )

    content = args.file_text or payload.get("content", "")
    if not isinstance(content, str):
        content = ""
    return agent_pb2.WriteResult(
        success=agent_pb2.WriteSuccess(
            path=args.path,
            lines_created=payload.get("lines_created", len(content.splitlines()) if content else 0),
            file_size=payload.get("file_size", len(content.encode("utf-8"))),
            file_content_after_write=content if args.return_file_content_after_write else "",
        )
    )


def build_delete_result_from_tool_result(
    path: str, tool_result: Any
) -> agent_pb2.DeleteResult:
    payload = _parse_jsonish(tool_result)
    if payload.get("error"):
        return agent_pb2.DeleteResult(
            error=agent_pb2.DeleteError(path=path, error=str(payload["error"]))
        )

    deleted_file = payload.get("deleted_file") or path
    prev_content = payload.get("prev_content") or ""
    file_size = payload.get("file_size")
    if not isinstance(file_size, int):
        file_size = len(prev_content.encode("utf-8")) if prev_content else 0
    return agent_pb2.DeleteResult(
        success=agent_pb2.DeleteSuccess(
            path=path,
            deleted_file=str(deleted_file),
            file_size=file_size,
            prev_content=prev_content,
        )
    )


def build_shell_result_from_tool_result(
    args: agent_pb2.ShellArgs, tool_result: Any
) -> agent_pb2.ShellResult:
    payload = _parse_jsonish(tool_result)
    error = payload.get("error")
    exit_code = payload.get("exit_code", 0)
    output = payload.get("output")
    if output is None:
        output = _tool_text(payload)
    stdout = payload.get("stdout", output if not error else "")
    stderr = payload.get("stderr", str(error) if error else "")
    execution_time = payload.get("execution_time", 0)

    if error or (isinstance(exit_code, int) and exit_code != 0):
        return agent_pb2.ShellResult(
            failure=agent_pb2.ShellFailure(
                command=args.command,
                working_directory=args.working_directory,
                exit_code=int(exit_code) if isinstance(exit_code, int) else 1,
                signal="",
                stdout=str(stdout or ""),
                stderr=str(stderr or ""),
                execution_time=int(execution_time or 0),
                aborted=bool(payload.get("aborted", False)),
                abort_reason=payload.get("abort_reason"),
            )
        )

    return agent_pb2.ShellResult(
        success=agent_pb2.ShellSuccess(
            command=args.command,
            working_directory=args.working_directory,
            exit_code=0,
            signal="",
            stdout=str(stdout or ""),
            stderr=str(stderr or ""),
            execution_time=int(execution_time or 0),
            pid=payload.get("pid"),
        )
    )


def build_ls_result_from_tool_result(path: str, tool_result: Any) -> agent_pb2.LsResult:
    payload = _parse_jsonish(tool_result)
    if payload.get("error"):
        return agent_pb2.LsResult(
            error=agent_pb2.LsError(path=path, error=str(payload["error"]))
        )

    root_path = path or "."
    child_dirs: list[agent_pb2.LsDirectoryTreeNode] = []
    child_files: list[agent_pb2.LsDirectoryTreeNode_File] = []

    files = payload.get("files")
    if isinstance(files, list):
        for item in files:
            name = PurePosixPath(str(item)).name or str(item)
            if name.endswith("/"):
                child_dirs.append(
                    agent_pb2.LsDirectoryTreeNode(
                        abs_path=str(PurePosixPath(root_path) / name.rstrip("/")),
                        children_dirs=[],
                        children_files=[],
                        children_were_processed=False,
                        full_subtree_extension_counts={},
                        num_files=0,
                    )
                )
            else:
                child_files.append(agent_pb2.LsDirectoryTreeNode_File(name=name))
    else:
        for line in _tool_text(payload).splitlines():
            entry = line.strip()
            if not entry:
                continue
            if entry.endswith("/"):
                child_dirs.append(
                    agent_pb2.LsDirectoryTreeNode(
                        abs_path=str(PurePosixPath(root_path) / entry.rstrip("/")),
                        children_dirs=[],
                        children_files=[],
                        children_were_processed=False,
                        full_subtree_extension_counts={},
                        num_files=0,
                    )
                )
            else:
                child_files.append(agent_pb2.LsDirectoryTreeNode_File(name=entry))

    root = agent_pb2.LsDirectoryTreeNode(
        abs_path=root_path,
        children_dirs=child_dirs,
        children_files=child_files,
        children_were_processed=True,
        full_subtree_extension_counts={},
        num_files=len(child_files),
    )
    return agent_pb2.LsResult(success=agent_pb2.LsSuccess(directory_tree_root=root))


def build_grep_result_from_tool_result(
    args: agent_pb2.GrepArgs, tool_result: Any
) -> agent_pb2.GrepResult:
    payload = _parse_jsonish(tool_result)
    if payload.get("error"):
        return agent_pb2.GrepResult(
            error=agent_pb2.GrepError(error=str(payload["error"]))
        )

    output_mode = args.output_mode or "content"
    workspace_key = args.path or "."

    if output_mode == "files_with_matches":
        files = payload.get("files")
        if not isinstance(files, list):
            files = [line for line in _tool_text(payload).splitlines() if line.strip()]
        union = agent_pb2.GrepUnionResult(
            files=agent_pb2.GrepFilesResult(
                files=[str(item) for item in files],
                total_files=len(files),
                client_truncated=bool(payload.get("truncated", False)),
                ripgrep_truncated=False,
            )
        )
    elif output_mode == "count":
        counts_payload = payload.get("counts")
        counts: list[agent_pb2.GrepFileCount] = []
        if isinstance(counts_payload, dict):
            for file_path, count in counts_payload.items():
                counts.append(
                    agent_pb2.GrepFileCount(file=str(file_path), count=int(count))
                )
        else:
            for line in _tool_text(payload).splitlines():
                if ":" not in line:
                    continue
                file_path, count_text = line.rsplit(":", 1)
                try:
                    count = int(count_text)
                except ValueError:
                    continue
                counts.append(agent_pb2.GrepFileCount(file=file_path, count=count))
        union = agent_pb2.GrepUnionResult(
            count=agent_pb2.GrepCountResult(
                counts=counts,
                total_files=len(counts),
                total_matches=sum(item.count for item in counts),
                client_truncated=bool(payload.get("truncated", False)),
                ripgrep_truncated=False,
            )
        )
    else:
        raw_matches = payload.get("matches")
        grouped: dict[str, list[agent_pb2.GrepContentMatch]] = {}
        if isinstance(raw_matches, list) and raw_matches and isinstance(raw_matches[0], dict):
            for match in raw_matches:
                file_path = str(match.get("path", workspace_key))
                grouped.setdefault(file_path, []).append(
                    agent_pb2.GrepContentMatch(
                        line_number=int(match.get("line_number", 0) or 0),
                        content=str(match.get("content", "")),
                        content_truncated=False,
                        is_context_line=bool(match.get("is_context_line", False)),
                    )
                )
        else:
            for line in _tool_text(payload).splitlines():
                if not line.strip():
                    continue
                parts = line.split(":", 2)
                if len(parts) < 3:
                    continue
                file_path, line_number, content = parts
                try:
                    line_number_int = int(line_number)
                except ValueError:
                    continue
                grouped.setdefault(file_path, []).append(
                    agent_pb2.GrepContentMatch(
                        line_number=line_number_int,
                        content=content,
                        content_truncated=False,
                        is_context_line=False,
                    )
                )
        file_matches = [
            agent_pb2.GrepFileMatch(file=file_path, matches=matches)
            for file_path, matches in grouped.items()
        ]
        union = agent_pb2.GrepUnionResult(
            content=agent_pb2.GrepContentResult(
                matches=file_matches,
                total_lines=sum(len(entry.matches) for entry in file_matches),
                total_matched_lines=sum(
                    sum(0 if match.is_context_line else 1 for match in entry.matches)
                    for entry in file_matches
                ),
                client_truncated=bool(payload.get("truncated", False)),
                ripgrep_truncated=False,
            )
        )

    return agent_pb2.GrepResult(
        success=agent_pb2.GrepSuccess(
            pattern=args.pattern,
            path=args.path or "",
            output_mode=output_mode,
            workspace_results={workspace_key: union},
        )
    )


def build_diagnostics_result_from_tool_result(
    path: str, tool_result: Any
) -> agent_pb2.DiagnosticsResult:
    payload = _parse_jsonish(tool_result)
    if payload.get("error"):
        return agent_pb2.DiagnosticsResult(
            error=agent_pb2.DiagnosticsError(path=path, error=str(payload["error"]))
        )

    diagnostics_payload = payload.get("diagnostics") or []
    diagnostics: list[agent_pb2.Diagnostic] = []
    for item in diagnostics_payload:
        if not isinstance(item, dict):
            continue
        start = item.get("start", {})
        end = item.get("end", {})
        diagnostics.append(
            agent_pb2.Diagnostic(
                severity=int(item.get("severity", 0) or 0),
                range=agent_pb2.Range(
                    start=agent_pb2.Position(
                        line=int(start.get("line", 0) or 0),
                        column=int(start.get("column", 0) or 0),
                    ),
                    end=agent_pb2.Position(
                        line=int(end.get("line", start.get("line", 0)) or 0),
                        column=int(end.get("column", start.get("column", 0)) or 0),
                    ),
                ),
                message=str(item.get("message", "")),
                source=str(item.get("source", "")),
                code=str(item.get("code", "")),
                is_stale=bool(item.get("is_stale", False)),
            )
        )

    return agent_pb2.DiagnosticsResult(
        success=agent_pb2.DiagnosticsSuccess(
            path=path,
            diagnostics=diagnostics,
            total_diagnostics=len(diagnostics),
        )
    )


def build_mcp_result_from_tool_result(tool_result: Any) -> agent_pb2.McpResult:
    payload = _parse_jsonish(tool_result)
    if payload.get("error"):
        return agent_pb2.McpResult(
            error=agent_pb2.McpError(error=str(payload["error"]))
        )

    content_items = payload.get("content")
    if not isinstance(content_items, list):
        content_items = [{"type": "text", "text": _tool_text(payload)}]

    converted: list[agent_pb2.McpToolResultContentItem] = []
    for item in content_items:
        if not isinstance(item, dict):
            converted.append(
                agent_pb2.McpToolResultContentItem(
                    text=agent_pb2.McpTextContent(text=str(item))
                )
            )
            continue
        if item.get("type") == "image":
            converted.append(
                agent_pb2.McpToolResultContentItem(
                    image=agent_pb2.McpImageContent(
                        data=item.get("data", b""),
                        mime_type=str(item.get("mimeType", item.get("mime_type", ""))),
                    )
                )
            )
        else:
            converted.append(
                agent_pb2.McpToolResultContentItem(
                    text=agent_pb2.McpTextContent(text=str(item.get("text", "")))
                )
            )

    return agent_pb2.McpResult(
        success=agent_pb2.McpSuccess(
            content=converted,
            is_error=bool(payload.get("is_error", False)),
        )
    )


class CursorExecHandlers:
    """Adapt Cursor exec calls onto Hermes tool execution."""

    def __init__(
        self,
        agent: Any | None = None,
        cwd: str = ".",
        tools_registry: Any | None = None,
        execute_tool: Any | None = None,
        allowed_tools: set[str] | None = None,
    ) -> None:
        self.agent = agent
        self.cwd = cwd
        self.tools_registry = tools_registry
        self._execute_tool_callback = execute_tool
        self._allowed_tools = (
            set(allowed_tools) if allowed_tools is not None else self._discover_allowed_tools(agent)
        )

    @staticmethod
    def _discover_allowed_tools(agent: Any | None) -> set[str]:
        if agent is None:
            return set()
        allowed_tools: set[str] = set()
        for tool in getattr(agent, "tools", None) or []:
            if isinstance(tool, dict):
                name = tool.get("name")
                if not name and isinstance(tool.get("function"), dict):
                    name = tool["function"].get("name")
                if isinstance(name, str) and name:
                    allowed_tools.add(name)
        return allowed_tools

    def _execute_tool(
        self,
        tool_name: str,
        args: dict[str, Any],
        *,
        on_update: Any | None = None,
        tool_call_id: str | None = None,
    ) -> Any:
        if self._execute_tool_callback is not None:
            return self._execute_tool_callback(
                tool_name,
                args,
                on_update=on_update,
                tool_call_id=tool_call_id,
            )

        if self.agent is not None and hasattr(self.agent, "_invoke_tool"):
            task_id = getattr(self.agent, "_current_task_id", None) or "default"
            return self.agent._invoke_tool(
                tool_name,
                args,
                task_id,
                tool_call_id=tool_call_id,
            )

        registry = self.tools_registry
        if registry is None and self.agent is not None:
            registry = getattr(self.agent, "tools_registry", None)
            if registry is None:
                registry = getattr(self.agent, "tool_registry", None)
        if registry is None:
            raise RuntimeError("No Hermes tool executor available")
        return registry.dispatch(tool_name, args, task_id="default")

    def read(self, args: agent_pb2.ReadArgs) -> agent_pb2.ReadResult:
        result = self._execute_tool(
            "read_file",
            {"path": args.path},
            tool_call_id=args.tool_call_id,
        )
        return build_read_result_from_tool_result(args.path, result)

    def ls(self, args: agent_pb2.LsArgs) -> agent_pb2.LsResult:
        result = self._execute_tool(
            "search_files",
            {"pattern": "*", "target": "files", "path": args.path or "."},
            tool_call_id=args.tool_call_id,
        )
        return build_ls_result_from_tool_result(args.path, result)

    def grep(self, args: agent_pb2.GrepArgs) -> agent_pb2.GrepResult:
        result = self._execute_tool(
            "search_files",
            {
                "pattern": args.pattern,
                "target": "content",
                "path": args.path or ".",
                "file_glob": args.glob or None,
                "output_mode": args.output_mode or "content",
                "context": args.context or 0,
            },
            tool_call_id=args.tool_call_id,
        )
        return build_grep_result_from_tool_result(args, result)

    def write(self, args: agent_pb2.WriteArgs) -> agent_pb2.WriteResult:
        content = args.file_text
        if not content and args.file_bytes:
            content = bytes(args.file_bytes).decode("utf-8")
        result = self._execute_tool(
            "write_file",
            {"path": args.path, "content": content},
            tool_call_id=args.tool_call_id,
        )
        return build_write_result_from_tool_result(args, result)

    def delete(self, args: agent_pb2.DeleteArgs) -> agent_pb2.DeleteResult:
        result = self._execute_tool(
            "terminal",
            {
                "command": f"rm {shlex.quote(args.path)}",
                "workdir": self.cwd,
                "timeout": 30,
            },
            tool_call_id=args.tool_call_id,
        )
        return build_delete_result_from_tool_result(args.path, result)

    def shell(self, args: agent_pb2.ShellArgs) -> agent_pb2.ShellResult:
        result = self._execute_tool(
            "terminal",
            {
                "command": args.command,
                "workdir": args.working_directory or self.cwd,
                "timeout": args.timeout,
            },
            tool_call_id=args.tool_call_id,
        )
        return build_shell_result_from_tool_result(args, result)

    def shell_stream(
        self,
        args: agent_pb2.ShellArgs,
        callbacks: Any,
    ) -> agent_pb2.ShellResult:
        def on_update(partial: Any) -> None:
            payload = _parse_jsonish(partial)
            stdout = payload.get("stdout")
            stderr = payload.get("stderr")
            if stdout:
                callbacks.onStdout(str(stdout))
            if stderr and hasattr(callbacks, "onStderr"):
                callbacks.onStderr(str(stderr))

        result = self._execute_tool(
            "terminal",
            {
                "command": args.command,
                "workdir": args.working_directory or self.cwd,
                "timeout": args.timeout,
            },
            on_update=on_update,
            tool_call_id=args.tool_call_id,
        )
        return build_shell_result_from_tool_result(args, result)

    def diagnostics(self, args: agent_pb2.DiagnosticsArgs) -> agent_pb2.DiagnosticsResult:
        result = self._execute_tool(
            "lsp",
            {"action": "diagnostics", "file": args.path},
            tool_call_id=args.tool_call_id,
        )
        return build_diagnostics_result_from_tool_result(args.path, result)

    def mcp(self, args: agent_pb2.McpArgs) -> agent_pb2.McpResult:
        tool_name = args.tool_name or args.name
        if self._allowed_tools and tool_name not in self._allowed_tools:
            return agent_pb2.McpResult(
                error=agent_pb2.McpError(error=f"Unknown MCP tool: {tool_name}")
            )
        result = self._execute_tool(
            tool_name,
            decode_mcp_args(args),
            tool_call_id=args.tool_call_id,
        )
        return build_mcp_result_from_tool_result(result)