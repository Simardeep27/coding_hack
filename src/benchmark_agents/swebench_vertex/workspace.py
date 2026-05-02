from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from benchmark_agents.swebench_vertex.models import SWEbenchInstance
from benchmark_agents.swebench_vertex.utils import truncate_text


SAFE_COMMAND_PREFIXES = {
    "python",
    "python3",
    "pytest",
    "tox",
    "nox",
    "make",
    "git",
    "ls",
    "find",
    "sed",
    "cat",
    "rg",
    "grep",
    "env",
    "pip",
    "pip3",
    "uv",
    "poetry",
    "node",
    "npm",
    "pnpm",
    "yarn",
    "bash",
    "sh",
}

FORBIDDEN_COMMAND_TOKENS = {
    "sudo",
    "curl",
    "wget",
    "ssh",
    "scp",
    "rsync",
    "docker",
}


class WorkspaceSession:
    def __init__(
        self,
        *,
        instance: SWEbenchInstance,
        workspace_root: Path,
        repo_cache_dir: Path,
        default_command_timeout_secs: int,
    ) -> None:
        self.instance = instance
        self.workspace_root = workspace_root.resolve()
        self.repo_cache_dir = repo_cache_dir.resolve()
        self.default_command_timeout_secs = default_command_timeout_secs
        self.repo_dir = self.workspace_root / "repo"

    def prepare(self) -> None:
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self.repo_cache_dir.mkdir(parents=True, exist_ok=True)

        mirror_path = self.repo_cache_dir / f"{self.instance.repo.replace('/', '__')}.git"
        repo_url = f"https://github.com/{self.instance.repo}.git"

        if mirror_path.exists():
            self._run_git(
                ["git", "--git-dir", str(mirror_path), "remote", "update"],
                cwd=self.workspace_root,
            )
        else:
            self._run_git(
                ["git", "clone", "--mirror", repo_url, str(mirror_path)],
                cwd=self.workspace_root,
            )

        if self.repo_dir.exists():
            shutil.rmtree(self.repo_dir)

        self._run_git(
            ["git", "clone", str(mirror_path), str(self.repo_dir)],
            cwd=self.workspace_root,
        )
        self._run_git(
            ["git", "checkout", "--detach", self.instance.base_commit],
            cwd=self.repo_dir,
        )

    def list_files(self, path: str = ".", limit: int = 200) -> str:
        target = self._resolve_path(path)
        if target.is_file():
            return str(target.relative_to(self.repo_dir))

        results: list[str] = []
        for root, dirs, files in os.walk(target):
            dirs[:] = [name for name in dirs if name != ".git"]
            for filename in sorted(files):
                file_path = Path(root) / filename
                results.append(str(file_path.relative_to(self.repo_dir)))
                if len(results) >= limit:
                    return "\n".join(results)
        return "\n".join(results) if results else "(no files found)"

    def search_code(self, pattern: str, path: str = ".", limit: int = 50) -> str:
        target = self._resolve_path(path)
        search_root = "." if target == self.repo_dir else str(target.relative_to(self.repo_dir))
        rg_path = shutil.which("rg")
        if rg_path:
            args = [
                rg_path,
                "-n",
                "--hidden",
                "--glob",
                "!.git",
                "-F",
                pattern,
                search_root,
            ]
        else:
            args = [
                "grep",
                "-R",
                "-n",
                "-F",
                "--exclude-dir=.git",
                pattern,
                search_root,
            ]
        completed = subprocess.run(
            args,
            cwd=self.repo_dir,
            capture_output=True,
            text=True,
            timeout=self.default_command_timeout_secs,
            check=False,
        )
        output = completed.stdout.strip() or completed.stderr.strip()
        if not output:
            return "(no matches)"
        lines = output.splitlines()
        return "\n".join(lines[:limit])

    def read_file(self, path: str, start_line: int = 1, end_line: int = 200) -> str:
        target = self._resolve_path(path)
        if not target.is_file():
            raise FileNotFoundError(f"File does not exist: {path}")
        lines = target.read_text(encoding="utf-8").splitlines()
        start_index = max(start_line - 1, 0)
        end_index = max(end_line, start_line)
        excerpt = lines[start_index:end_index]
        numbered = [
            f"{line_number}: {line}"
            for line_number, line in enumerate(excerpt, start=start_index + 1)
        ]
        return "\n".join(numbered) if numbered else "(empty file excerpt)"

    def replace_text(
        self,
        path: str,
        old_text: str,
        new_text: str,
        count: int = 1,
    ) -> str:
        if count <= 0:
            raise ValueError("count must be positive.")
        target = self._resolve_path(path)
        original = target.read_text(encoding="utf-8")
        occurrences = original.count(old_text)
        if occurrences < count:
            raise ValueError(
                f"Requested {count} replacement(s), but only found {occurrences} match(es)."
            )
        updated = original.replace(old_text, new_text, count)
        target.write_text(updated, encoding="utf-8")
        return f"Replaced {count} occurrence(s) in {path}."

    def replace_lines(
        self,
        path: str,
        start_line: int,
        end_line: int,
        new_content: str,
        expected_old_text: str | None = None,
    ) -> str:
        if start_line <= 0:
            raise ValueError("start_line must be positive.")
        if end_line < start_line:
            raise ValueError("end_line must be greater than or equal to start_line.")

        target = self._resolve_path(path)
        lines = target.read_text(encoding="utf-8").splitlines(keepends=True)
        if start_line > len(lines) + 1:
            raise ValueError(
                f"start_line={start_line} is beyond the end of {path} "
                f"({len(lines)} lines)."
            )

        replacement = _content_to_lines(new_content)
        start_index = start_line - 1
        end_index = min(end_line, len(lines))
        existing_text = "".join(lines[start_index:end_index])
        if expected_old_text is not None and expected_old_text not in existing_text:
            raise ValueError(
                "expected_old_text was not found in the selected line range. "
                "Read the target lines again and retry with the correct range."
            )
        updated = lines[:start_index] + replacement + lines[end_index:]
        target.write_text("".join(updated), encoding="utf-8")
        return f"Replaced lines {start_line}-{end_line} in {path}."

    def replace_line(
        self,
        path: str,
        line_number: int,
        new_line: str,
        expected_old_text: str | None = None,
    ) -> str:
        return self.replace_lines(
            path=path,
            start_line=line_number,
            end_line=line_number,
            new_content=new_line,
            expected_old_text=expected_old_text,
        ).replace(f"lines {line_number}-{line_number}", f"line {line_number}")

    def insert_lines(
        self,
        path: str,
        after_line: int,
        content: str,
    ) -> str:
        if after_line < 0:
            raise ValueError("after_line must be zero or positive.")

        target = self._resolve_path(path)
        lines = target.read_text(encoding="utf-8").splitlines(keepends=True)
        if after_line > len(lines):
            raise ValueError(
                f"after_line={after_line} is beyond the end of {path} "
                f"({len(lines)} lines)."
            )

        insertion = _content_to_lines(content)
        updated = lines[:after_line] + insertion + lines[after_line:]
        target.write_text("".join(updated), encoding="utf-8")
        return f"Inserted {len(insertion)} line(s) after line {after_line} in {path}."

    def insert_line(
        self,
        path: str,
        after_line: int,
        line: str,
    ) -> str:
        return self.insert_lines(
            path=path,
            after_line=after_line,
            content=line,
        ).replace("line(s)", "line")

    def write_file(self, path: str, content: str) -> str:
        target = self._resolve_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Wrote {path} ({len(content)} chars)."

    def run_command(self, command: str, timeout_secs: int | None = None) -> str:
        args = shlex.split(command)
        if not args:
            raise ValueError("Command was empty.")
        self._validate_command(args)
        normalized_args = self._normalize_command_args(args)

        completed = subprocess.run(
            normalized_args,
            cwd=self.repo_dir,
            capture_output=True,
            text=True,
            env=self._build_command_env(normalized_args),
            timeout=timeout_secs or self.default_command_timeout_secs,
            check=False,
        )

        stdout = completed.stdout.strip()
        stderr = completed.stderr.strip()
        output = []
        output.append(f"executed_command: {shlex.join(normalized_args)}")
        output.append(f"exit_code: {completed.returncode}")
        if stdout:
            output.append("stdout:")
            output.append(stdout)
        if stderr:
            output.append("stderr:")
            output.append(stderr)
        return truncate_text("\n".join(output), 10_000)

    def get_git_status(self) -> str:
        return self._run_git(
            ["git", "status", "--short"],
            cwd=self.repo_dir,
            truncate_chars=8_000,
        )

    def get_git_diff(self, truncate_chars: int | None = None) -> str:
        return self._run_git(
            ["git", "diff", "--binary", "--full-index"],
            cwd=self.repo_dir,
            truncate_chars=truncate_chars,
        )

    def runtime_metadata(self) -> dict[str, object]:
        return {
            "repo_dir": str(self.repo_dir),
            "python_executable": sys.executable,
            "python_version": sys.version.split()[0],
            "available_commands": {
                name: shutil.which(name)
                for name in (
                    "git",
                    "python",
                    "python3",
                    "pytest",
                    "rg",
                    "grep",
                    "pip",
                    "pip3",
                    "uv",
                    "tox",
                )
            },
            "validation_notes": [
                "Commands run in a generic host environment, not the official SWE-bench harness.",
                "Local validation failures can come from interpreter or dependency mismatch, not only from bad code changes.",
                "PYTHONPATH is automatically prefixed with the repo root for run_command executions.",
            ],
        }

    def _resolve_path(self, path: str) -> Path:
        candidate = (self.repo_dir / path).resolve()
        if candidate != self.repo_dir and self.repo_dir not in candidate.parents:
            raise ValueError(f"Path escapes repo root: {path}")
        return candidate

    def _validate_command(self, args: list[str]) -> None:
        first = args[0]
        if any(token in FORBIDDEN_COMMAND_TOKENS for token in args):
            raise ValueError("Command contained a forbidden token.")
        if first in {"bash", "sh"} and any(flag in {"-c", "-lc"} for flag in args[1:]):
            raise ValueError("Inline shell execution is not allowed.")
        if first.startswith("./"):
            script_path = self._resolve_path(first)
            if not script_path.exists():
                raise ValueError(f"Relative executable not found: {first}")
            return
        if first not in SAFE_COMMAND_PREFIXES:
            raise ValueError(
                f"Command prefix '{first}' is not allowed by the local command guard."
            )

    def _normalize_command_args(self, args: list[str]) -> list[str]:
        normalized = list(args)
        if self._is_pytest_invocation(normalized) and not any(
            arg == "-c" or arg.startswith("--rootdir") for arg in normalized
        ):
            insert_index = 1 if normalized[0] == "pytest" else 3
            normalized.insert(insert_index, "--rootdir=.")
        return normalized

    def _build_command_env(self, args: list[str]) -> dict[str, str]:
        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{self.repo_dir}{os.pathsep}{existing_pythonpath}"
            if existing_pythonpath
            else str(self.repo_dir)
        )
        env.setdefault("PYTHONNOUSERSITE", "1")
        env.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
        if self._is_pytest_invocation(args):
            env.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
        return env

    def _is_pytest_invocation(self, args: list[str]) -> bool:
        if not args:
            return False
        if args[0] == "pytest":
            return True
        return len(args) >= 3 and args[0] in {"python", "python3"} and args[1:3] == [
            "-m",
            "pytest",
        ]

    def _run_git(
        self,
        args: list[str],
        *,
        cwd: Path,
        truncate_chars: int | None = None,
    ) -> str:
        completed = subprocess.run(
            args,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=self.default_command_timeout_secs,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Git command failed ({' '.join(args)}):\n{completed.stderr.strip()}"
            )
        output = completed.stdout.strip()
        if truncate_chars is not None:
            return truncate_text(output, truncate_chars)
        return output


def _content_to_lines(content: str) -> list[str]:
    if not content:
        return []
    lines = content.splitlines(keepends=True)
    if lines and not lines[-1].endswith(("\n", "\r")):
        lines[-1] += "\n"
    return lines
