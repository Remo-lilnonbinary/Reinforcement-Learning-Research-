from server import Door2Door, HireParams, AssignParams, _Empty

env = Door2Door(task_spec={"race_id": "IA_3_2018", "candidate": "Axne", "weeks": 12, "baseline_share": 0.487, "seed": 0})

print("baseline:", round(env._expected_share(), 4))

env.hire_canvassers(HireParams(count=20))
env.advance_week(_Empty())
print("after wk1, hours:", env._hours_available())

# simulate a sane strategy: persuasion in mid weeks, GOTV at the end
pid = list(env.precincts.keys())[0]

for _ in range(8):
    env.assign_canvassers(AssignParams(precinct_id=pid, hours=20, mode="persuasion", target_segment="swing"))
    env.advance_week(_Empty())

env.assign_canvassers(AssignParams(precinct_id=pid, hours=50, mode="gotv", target_segment="hard_sup_low"))
env.advance_week(_Empty())
env.assign_canvassers(AssignParams(precinct_id=pid, hours=50, mode="gotv", target_segment="hard_sup_low"))
env.advance_week(_Empty())

print("final:", round(env._expected_share(), 4))
print("delta from baseline:", round(env._expected_share() - env.baseline_share, 4))