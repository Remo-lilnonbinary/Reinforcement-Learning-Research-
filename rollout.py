from dotenv import load_dotenv
load_dotenv()

import anthropic
from openreward import OpenReward

c = OpenReward()
a = anthropic.Anthropic()

env = c.environments.get(name="Remo/door2door")
tasks = env.list_tasks(split="train")
tools = env.list_tools(format="anthropic")

task = next(t for t in tasks if t.task_spec["district_id"] == "IA-03" and t.task_spec["year"] == 2018 and t.task_spec["seed"] == 0)
print("running on", task.task_spec)

with env.session(task=task) as s:
    prompt_text = s.get_prompt()[0].text
    print("PROMPT FIRST 200 CHARS:", prompt_text[:200])
    print("PROMPT LENGTH:", len(prompt_text))
    print()

    msgs = [{"role": "user", "content": prompt_text}]
    cum = 0.0
    done = False

    for turn in range(80):
        resp = a.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=2048,
            tools=tools,
            messages=msgs,
        )
        print(f"--- turn {turn}, stop_reason={resp.stop_reason} ---")

        for block in resp.content:
            if block.type == "text":
                print(f"  [text] {block.text[:200]}")
            elif block.type == "tool_use":
                print(f"  [tool_use] {block.name}({block.input})")

        msgs.append({"role": "assistant", "content": resp.content})

        if resp.stop_reason != "tool_use":
            if resp.stop_reason == "end_turn" and not done:
                print("  >> nudging continue")
                msgs.append({"role": "user", "content": "Continue. Do not stop until week 12. Take the next action now."})
                continue
            print(f"  >> stopping: {resp.stop_reason}")
            break

        results = []
        for block in resp.content:
            if block.type != "tool_use":
                continue
            r = s.call_tool(block.name, block.input)
            cum += r.reward
            print(f"    -> {block.name}: reward={r.reward:+.4f} finished={r.finished}")
            print(f"       result: {r.blocks[0].text[:150]}")
            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": r.blocks[0].text,
            })
            if r.finished:
                done = True

        msgs.append({"role": "user", "content": results})

        if done:
            print(f"\nDONE. cumulative reward: {cum:+.4f}")
            print(f"that's {cum*100:+.2f}pp vs baseline")
            break

    print(f"\nloop ended. turns={turn}, cum={cum:+.4f}, done={done}")