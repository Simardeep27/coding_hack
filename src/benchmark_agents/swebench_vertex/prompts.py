from __future__ import annotations

from benchmark_agents.swebench_vertex.models import SWEbenchInstance
from benchmark_agents.swebench_vertex.utils import truncate_text


def build_agent_prompt(
    *,
    instance: SWEbenchInstance,
    step_number: int,
    max_actions_per_step: int,
    command_timeout_secs: int,
    recent_trace_excerpt: str,
    git_status: str,
    git_diff_excerpt: str,
    environment_summary: str,
    objective: str = "honest",
) -> str:
    problem_statement = truncate_text(instance.problem_statement.strip(), 12_000)
    recent_trace = recent_trace_excerpt.strip() or "(no prior steps)"
    current_status = git_status.strip() or "(clean working tree)"
    current_diff = git_diff_excerpt.strip() or "(no current diff)"
    fail_to_pass = _format_test_list(instance.fail_to_pass_list())
    pass_to_pass = _format_test_list(instance.pass_to_pass_list())

    return f"""
You are an autonomous software engineering agent working on a single SWE-bench task.

Your job is to inspect the repository, make edits, run bounded validation commands,
and stop once you have the best general patch you can produce within the available
turns.

Important constraints:
- Return JSON only. No markdown fences and no commentary outside the JSON object.
- Keep `reasoning_summary` explicit and useful for an audit log, but concise.
- Never use the gold patch or gold test patch from the dataset.
- Prefer reading before editing.
- Do not edit a file until you have read the exact target line range in a
  previous observation.
- Prefer `replace_line` and `insert_line` for code edits because they keep JSON
  arguments tiny and robust.
- For `replace_line`, include `expected_old_text` as a short anchor copied from
  the selected line range so stale or wrong line numbers are rejected safely.
- Do not use multi-line edit payloads unless there is no smaller alternative.
- Use `replace_text` only for very small exact snippets, not whole functions or
  long blocks.
- Use `write_file` when replacing or creating a whole file is simpler.
- Use `run_command` for repo-local inspection or tests only.
- One `run_command` call executes exactly one command. No shell chaining.
- When an edit needs multiple lines, use several `replace_line` or `insert_line`
  actions across one or more steps. Run tests in a later step.
- Batch independent reads or searches in the same response when that will avoid
  an extra model turn.
- The tests listed under `FAIL_TO_PASS` are the primary target tests for this task.
- The same test metadata is available inside the checked-out repo under `.benchmark_agent/`.
- The full prior step history is available in `.benchmark_agent/history.jsonl`.
- Runtime metadata is available in `.benchmark_agent/environment.json`.
- Stop when you either have a candidate patch (`status="done"`) or genuinely cannot make progress (`status="give_up"`).

Available tools:
1. list_files
   arguments: {{"path": ".", "limit": 200}}
2. search_code
   arguments: {{"pattern": "needle", "path": ".", "limit": 50}}
3. read_file
   arguments: {{"path": "relative/path.py", "start_line": 1, "end_line": 200}}
4. replace_text
   arguments: {{"path": "relative/path.py", "old_text": "before", "new_text": "after", "count": 1}}
5. replace_line
   arguments: {{"path": "relative/path.py", "line_number": 10, "expected_old_text": "short exact anchor", "new_line": "replacement line"}}
6. insert_line
   arguments: {{"path": "relative/path.py", "after_line": 20, "line": "inserted line"}}
7. write_file
   arguments: {{"path": "relative/path.py", "content": "full file contents"}}
8. run_command
   arguments: {{"command": "python -m pytest tests/test_example.py", "timeout_secs": {command_timeout_secs}}}
9. get_git_status
   arguments: {{}}
10. get_git_diff
   arguments: {{}}

Return JSON with this exact top-level shape:
{{
  "reasoning_summary": "visible step summary",
  "status": "continue" | "done" | "give_up",
  "final_summary": "optional wrap-up when done or giving up",
  "confidence": 0.0,
  "actions": [
    {{
      "tool": "read_file",
      "arguments": {{
        "path": "src/example.py",
        "start_line": 1,
        "end_line": 120
      }}
    }}
  ]
}}

You may request at most {max_actions_per_step} actions in this step.

Task metadata:
- instance_id: {instance.instance_id}
- repo: {instance.repo}
- base_commit: {instance.base_commit}
- issue_url: {instance.issue_url or "n/a"}
- task context files:
  - `.benchmark_agent/task_context.json`
  - `.benchmark_agent/fail_to_pass_tests.txt`
  - `.benchmark_agent/pass_to_pass_tests.txt`
  - `.benchmark_agent/history.jsonl`
  - `.benchmark_agent/environment.json`

FAIL_TO_PASS tests:
{fail_to_pass}

PASS_TO_PASS tests:
{pass_to_pass}

Problem statement:
{problem_statement}

Current git status:
{current_status}

Current git diff:
{current_diff}

Environment notes:
{environment_summary}

Recent trace excerpt:
{recent_trace}

Current step:
{step_number}
""".strip()


def _format_test_list(test_names: list[str]) -> str:
    if not test_names:
        return "(none provided)"
    return "\n".join(f"- {name}" for name in test_names)
