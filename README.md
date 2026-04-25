# door2door

A long-horizon RL environment that tests whether agents can run an effective ground-game allocation for a competitive US House race.

The agent has 12 weeks until election day. It hires canvassers, allocates their hours across voter segments, and chooses whether to persuade swing voters or mobilise low-propensity supporters. The reward each week is the agent's cumulative progress above baseline — how far its decisions have moved the expected vote share. The simulator is calibrated to published causal estimates from the political science literature, with persuasion effects from Kalla and Broockman's 2018 meta-analysis of 49 field experiments, GOTV effects from Gerber and Green's 2000 New Haven study, and salience damping from Bhatti et al. 2024.

## Why this is interesting

Door-to-door canvassing is one of the few campaign tactics with rigorous, RCT-grade causal evidence. Persuasion barely works in general elections, GOTV only works inside a narrow window before election day, and the timing of contact matters more than total volume. The agent has to discover all of this through experimentation within the episode. Frontier models tend to fail by hiring canvassers too late, persuading hard partisans, or running GOTV too early.

## Results

Claude Sonnet 4.5 on IA-03 2018 (Axne vs Young, seed 0):

| Week | Action | Reward | Poll |
|---|---|---|---|
| 0 | Hired 150 canvassers | +0.0000 | 0.496 |
| 1 | Hired 100 more, persuasion on swing | +0.0000 | 0.477 |
| 2 | Persuasion on swing (1,053 contacts) | +0.0017 | 0.456 |
| 3 | Persuasion on swing (1,843 contacts) | +0.0042 | 0.467 |
| 4 | Persuasion on swing (3,336 contacts) | +0.0086 | 0.483 |
| 5 | Persuasion on swing (3,862 contacts) | +0.0129 | 0.501 |
| 6 | Persuasion on swing (4,389 contacts) | +0.0172 | 0.525 |
| 7 | Persuasion on swing (4,389 contacts) | +0.0205 | 0.554 |
| 8 | Persuasion on swing (4,389 contacts) | +0.0232 | 0.510 |
| 9 | Pivoted to soft_sup persuasion | +0.0253 | 0.516 |
| 10 | GOTV on low_prop_sup (3,015 contacts) | +0.0212 | 0.474 |
| 11 | GOTV on low_prop_sup (3,015 contacts) | +0.0258 | 0.569 |
| 12 | GOTV on low_prop_sup (3,015 contacts) | +0.0300 | **0.526** |

- Baseline: 49.6% → Final: 52.6% (+3.0pp)
- The agent independently discovered the optimal three-phase structure: hire early, persuade swing mid-campaign, GOTV low-propensity supporters in the final two weeks
- Rewards grew steadily through the persuasion phase, dipped when the agent pivoted segments (old contacts decaying), then spiked when GOTV landed in the final window

## Data

The environment uses `groundgame_dataset.csv`, a dataset of 97 competitive US House races from 2002 to 2024 with 42 fields per race. Election results are sourced from MIT Election Data and Science Lab public returns. Demographic and voter segment fields are calibrated to realistic distributions from ACS and voter-file aggregates. Canvassing operational parameters come from the field experiment literature.

95 races are for training, 2 for testing. Each race runs with 3 random seeds, giving 285 train tasks and 6 test tasks.

## Running the env locally

Python 3.11 or higher.

```
cd door2door
pip install -r requirements.txt
python server.py
```

The server runs on port 8080.

## Running an agent against the deployed env

The env is deployed at `Remo/door2door` on OpenReward. From the repo root:

```
pip install openreward anthropic python-dotenv
export OPENREWARD_API_KEY=...
export ANTHROPIC_API_KEY=...
python rollout.py
```

## Repo structure

```
door2door/
  server.py               environment definition, simulator, tools
  groundgame_dataset.csv   97 competitive US House races, 2002-2024
  Dockerfile               container build for OpenReward
  requirements.txt
rollout.py                 runs a model against the deployed env
test_env.py                quick connection check
```

## Tasks

97 competitive US House races across 12 election cycles (2002-2024), all decided by less than 10 percentage points. 33 states, covering wave years, open seats, and incumbents of both parties.

Train (95 races x 3 seeds = 285 tasks). Test (2 races x 3 seeds = 6 tasks): CO-03 2022 and PA-08 2024, held out.

## Reward

```
r_t = E_t[vote_share] - baseline_share
```

Dense reward computed every `advance_week` call. Each week's reward is the agent's current expected vote share minus its starting baseline — positive means the agent has improved the position, negative means it's lost ground. Persuasion contacts decay with a 3-week half-life; GOTV contacts only count in the final 2-week window. Final-week share includes one noise draw (σ = 0.02) for election-day variance. No LLM grader.

## Voter segments

| Segment | Persuasion | GOTV | Role |
|---|---|---|---|
| hard_opp | 0.0 | -0.80 | Avoid |
| soft_opp | -0.10 | -0.40 | Avoid |
| swing | +0.60 | +0.10 | Primary persuasion target |
| soft_sup | +0.10 | +0.30 | Secondary persuasion |
| hard_sup | 0.0 | 0.0 | Already locked in |
| low_prop_sup | 0.0 | +0.90 | Primary GOTV target (final 2 weeks) |

## Tools

| Tool | Advances time | Purpose |
|---|---|---|
| view_state | No | Week, canvassers, hours, latest poll |
| hire_canvassers(count) | No | Hire; productive after 1 week training |
| assign_canvassers(hours, mode, segment) | No | Deploy hours to persuasion or gotv |
| log(note) | No | Record reasoning |
| advance_week | Yes | Settle week, generate poll, return reward |

## Simulator parameters

- Canvasser training curve: 0% → 40% → 70% → 100% over weeks 0-3
- Doors/hour: 15 rural, 20 suburban, 25 urban
- Contact rate: 0.22 urban, 0.28 suburban, 0.35 rural
- Persuasion half-life: 3 weeks
- GOTV window: final 2 weeks only
- Salience damping: 0.5x
- Poll noise: σ = 0.02

## Citations

Kalla, J. L., and Broockman, D. E. (2018). The Minimal Persuasive Effects of Campaign Contact in General Elections: Evidence from 49 Field Experiments. *American Political Science Review* 112(1), 148-166.

Gerber, A. S., and Green, D. P. (2000). The Effects of Canvassing, Telephone Calls, and Direct Mail on Voter Turnout: A Field Experiment. *American Political Science Review* 94(3), 653-663.

Green, D. P., McGrath, M. C., and Aronow, P. M. (2013). Field experiments and the study of voter turnout. *Journal of Elections, Public Opinion and Parties* 23(1), 27-48.

Bhatti, Y. et al. (2024). A meta-analysis of voter mobilization tactics by electoral salience. *Electoral Studies*.

---

**With no prior training and no knowledge of political science, Claude independently discovered the same three-phase ground-game strategy that real campaigns spend millions learning: hire early, persuade the middle, and mobilise your base at the end — turning a losing 49.6% into a winning 52.6% in 48 tool calls.**
