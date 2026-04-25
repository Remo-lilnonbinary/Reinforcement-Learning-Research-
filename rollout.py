from dotenv import load_dotenv
load_dotenv()

import anthropic
from openreward import OpenReward

c = OpenReward()
a = anthropic.Anthropic()

env = c.environments.get(name="Remo/door2door")
tasks = env.list_tasks(split="train")
tools = env.list_tools(format="anthropic")

task = next(t for t in tasks if t.task_spec["race_id"] == "IA_3_2018" and t.task_spec["seed"] == 0)
print("running on", task.task_spec)

with env.session(task=task) as s:
    msgs = [{"role": "user", "content": s.get_prompt()[0].text}]
    cum = 0.0
    done = False

    for turn in range(80):
        resp = a.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=2048,
            tools=tools,
            messages=msgs,
        )
        msgs.append({"role": "assistant", "content": resp.content})

        if resp.stop_reason != "tool_use":
            if resp.stop_reason == "end_turn" and not done:
                msgs.append({"role": "user", "content": "Continue. Do not stop until week 12. Take the next action now."})
                continue
            print(f"stop: {resp.stop_reason}")
            break

        results = []
        for block in resp.content:
            if block.type != "tool_use":
                continue
            r = s.call_tool(block.name, block.input)
            cum += r.reward
            if block.name == "advance_week":
                print(f"  wk: r={r.reward:+.4f} cum={cum:+.4f}")
            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": r.blocks[0].text,
            })
            if r.finished:
                done = True

        msgs.append({"role": "user", "content": results})

        if done:
            print(f"done. cumulative reward: {cum:+.4f}")
            print(f"that's {cum*100:+.2f}pp vs baseline")
            break
