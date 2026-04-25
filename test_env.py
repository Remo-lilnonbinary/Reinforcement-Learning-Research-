from dotenv import load_dotenv
load_dotenv()

from openreward import OpenReward

c = OpenReward()
env = c.environments.get(name="Remo/door2door")
print(env.list_tasks(split="train"))
print(env.list_tools(format="anthropic"))
