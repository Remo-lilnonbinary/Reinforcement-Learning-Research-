"""
Multi-model rollout harness for door2door.

Usage:
   Claude, GPT and Gemini

Runs each configured model on the test split (CO-03 2022 and PA-08 2024,
3 seeds each = 6 tasks per model). Writes per-rollout JSON to results/.

Then:
    python rollout_multi.py --summarise
Prints a leaderboard and writes results/summary.json.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from openreward import OpenReward

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


# ---- Model adapters ---------------------------------------------------------
# Each adapter implements one method: run_episode(env_session, prompt, tools)
# and returns dict with keys: cumulative_reward, final_share, turns, tool_calls,
# trajectory (list of {week, action, reward, poll}).
# ---------------------------------------------------------------------------

class ClaudeAgent:
    name = "claude-sonnet-4-5"
    model_id = "claude-sonnet-4-5"

    def __init__(self):
        import anthropic
        self.client = anthropic.Anthropic()

    def run_episode(self, s, prompt, tools):
        msgs = [{"role": "user", "content": prompt}]
        cum = 0.0
        done = False
        trajectory = []
        tool_calls = 0

        for turn in range(80):
            resp = self.client.messages.create(
                model=self.model_id, max_tokens=2048,
                tools=tools, messages=msgs,
            )
            msgs.append({"role": "assistant", "content": resp.content})

            if resp.stop_reason != "tool_use":
                if resp.stop_reason == "end_turn" and not done:
                    msgs.append({"role": "user", "content": "Continue. Do not stop until week 12. Take the next action now."})
                    continue
                break

            results = []
            for block in resp.content:
                if block.type != "tool_use":
                    continue
                tool_calls += 1
                r = s.call_tool(block.name, block.input)
                cum += r.reward
                trajectory.append({
                    "tool": block.name, "input": block.input,
                    "reward": r.reward, "result": r.blocks[0].text[:200],
                })
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": r.blocks[0].text})
                if r.finished:
                    done = True
            msgs.append({"role": "user", "content": results})
            if done:
                break

        return {"cumulative_reward": cum, "turns": turn + 1, "tool_calls": tool_calls,
                "trajectory": trajectory, "finished": done}


class OpenAIAgent:
    """GPT-5 / GPT-4o via Chat Completions tool calling."""
    name = "gpt-4o"  # change to gpt-5 if you have access
    model_id = "gpt-4o"

    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI()

    @staticmethod
    def _convert_tools(anthropic_tools):
        """Convert anthropic-format tool defs to openai-format."""
        out = []
        for t in anthropic_tools:
            out.append({
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {}),
                },
            })
        return out

    def run_episode(self, s, prompt, tools):
        oai_tools = self._convert_tools(tools)
        msgs = [{"role": "user", "content": prompt}]
        cum = 0.0
        done = False
        trajectory = []
        tool_calls = 0

        for turn in range(80):
            resp = self.client.chat.completions.create(
                model=self.model_id, messages=msgs, tools=oai_tools,
                max_completion_tokens=2048,
            )
            msg = resp.choices[0].message
            msgs.append({"role": "assistant", "content": msg.content or "", "tool_calls": msg.tool_calls})

            if not msg.tool_calls:
                if resp.choices[0].finish_reason in ("stop", "end_turn") and not done:
                    msgs.append({"role": "user", "content": "Continue. Do not stop until week 12. Take the next action now."})
                    continue
                break

            for tc in msg.tool_calls:
                tool_calls += 1
                args = json.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else tc.function.arguments
                r = s.call_tool(tc.function.name, args)
                cum += r.reward
                trajectory.append({
                    "tool": tc.function.name, "input": args,
                    "reward": r.reward, "result": r.blocks[0].text[:200],
                })
                msgs.append({"role": "tool", "tool_call_id": tc.id, "content": r.blocks[0].text})
                if r.finished:
                    done = True
            if done:
                break

        return {"cumulative_reward": cum, "turns": turn + 1, "tool_calls": tool_calls,
                "trajectory": trajectory, "finished": done}


class GeminiAgent:
    """Gemini 2.5 Pro via google-genai SDK."""
    name = "gemini-2.5-pro"
    model_id = "gemini-2.5-pro"

    def __init__(self):
        from google import genai
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    def run_episode(self, s, prompt, tools):
        # Simplified, see Gemini docs for full tool-loop. Treating as best-effort.
        # Many users prefer to skip Gemini if the tool-call format gets fiddly;
        # Claude and GPT are enough for a portfolio comparison.
        raise NotImplementedError("Implement Gemini tool loop if you want this model.")


# ---- Driver ----------------------------------------------------------------

AGENTS = [
    ClaudeAgent,
    OpenAIAgent,
    # GeminiAgent,
]


def run_all():
    c = OpenReward()
    env = c.environments.get(name="Remo/door2door")
    test_tasks = env.list_tasks(split="test")
    tools = env.list_tools(format="anthropic")

    print(f"Found {len(test_tasks)} test tasks. Running {len(AGENTS)} agents.")

    for AgentCls in AGENTS:
        agent = AgentCls()
        print(f"\n=== {agent.name} ===")
        for task in test_tasks:
            tag = f"{task.task_spec['district_id']}_{task.task_spec['year']}_s{task.task_spec['seed']}"
            out_path = RESULTS_DIR / f"{agent.name}_{tag}.json"
            if out_path.exists():
                print(f"  skip {tag} (exists)")
                continue

            print(f"  running {tag}...", end=" ", flush=True)
            t0 = time.time()
            with env.session(task=task) as s:
                prompt = s.get_prompt()[0].text
                try:
                    result = agent.run_episode(s, prompt, tools)
                except Exception as e:
                    print(f"FAILED: {e}")
                    result = {"error": str(e), "cumulative_reward": 0.0, "finished": False}

            result["task"] = task.task_spec
            result["model"] = agent.name
            result["wall_seconds"] = round(time.time() - t0, 1)
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2, default=str)
            print(f"reward={result.get('cumulative_reward', 0):+.4f} ({result['wall_seconds']}s)")


def summarise():
    rows = []
    for path in RESULTS_DIR.glob("*.json"):
        if path.name == "summary.json":
            continue
        with open(path) as f:
            d = json.load(f)
        rows.append({
            "model": d.get("model", "?"),
            "task": f"{d['task']['district_id']}_{d['task']['year']}_s{d['task']['seed']}",
            "cumulative_reward": d.get("cumulative_reward", 0.0),
            "tool_calls": d.get("tool_calls", 0),
            "finished": d.get("finished", False),
            "wall_seconds": d.get("wall_seconds", 0),
        })

    if not rows:
        print("No results yet. Run without --summarise first.")
        return

    # Aggregate per model
    by_model = {}
    for r in rows:
        by_model.setdefault(r["model"], []).append(r)

    summary = {}
    print(f"\n{'Model':<25} {'Mean reward':>15} {'Min':>10} {'Max':>10} {'N':>5} {'Finished':>10}")
    print("-" * 80)
    for model, rs in sorted(by_model.items(), key=lambda x: -sum(r["cumulative_reward"] for r in x[1]) / len(x[1])):
        rewards = [r["cumulative_reward"] for r in rs]
        mean = sum(rewards) / len(rewards)
        finished_pct = 100 * sum(1 for r in rs if r["finished"]) / len(rs)
        summary[model] = {
            "mean_reward": mean,
            "min_reward": min(rewards),
            "max_reward": max(rewards),
            "n_tasks": len(rs),
            "finished_pct": finished_pct,
            "per_task": rs,
        }
        print(f"{model:<25} {mean:>+15.4f} {min(rewards):>+10.4f} {max(rewards):>+10.4f} {len(rs):>5} {finished_pct:>9.0f}%")

    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {RESULTS_DIR / 'summary.json'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--summarise", action="store_true", help="Aggregate existing results")
    args = p.parse_args()
    if args.summarise:
        summarise()
    else:
        run_all()
        summarise()
