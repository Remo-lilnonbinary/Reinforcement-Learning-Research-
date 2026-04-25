# door2door

A long-horizon RL environment that tests whether agents can run an effective ground-game allocation for a competitive US House race.

The agent has 12 weeks until election day. It hires canvassers, allocates their hours across voter segments, and chooses whether to persuade swing voters or mobilise low-propensity supporters. The reward each week is the change in expected vote share. Cumulative reward over an episode equals the final realised share minus the baseline.

The simulator is calibrated to published causal estimates from the political science literature. Persuasion effects are modelled on Kalla and Broockman's 2018 meta-analysis of 49 field experiments. GOTV effects come from Gerber and Green's 2000 New Haven study and the Green, McGrath and Aronow 2013 precision-weighted review. Effect sizes are damped by 50% to reflect the lower marginal returns observed in high-salience House races (Bhatti et al. 2024).

## Why this is interesting

Door-to-door canvassing is one of the few campaign tactics with rigorous, RCT-grade causal evidence. The literature is clear: persuasion barely works in general elections (Kalla and Broockman find a best estimate of zero), GOTV only works inside a narrow window before election day, and the timing of contact matters more than total volume. The agent has to discover all of this through experimentation within the episode.

Frontier models tend to fail in predictable ways: hiring canvassers too late (missing productive weeks), persuading hard partisans (zero or negative effect), running GOTV too early (contacts outside the 2-week window have no effect), or failing to adapt when polls move against them.

## Initial results

Claude Sonnet 4.5 on IA-03 2018 (Axne vs Young, seed 0):

- Baseline expected share: 49.6%
- Final realised share: 50.4%
- Cumulative reward: +0.0063 (+0.63pp)
- Turns: 52 tool calls across 12 weeks
- Strategy: hired 300 canvassers in weeks 0-2, ran persuasion on swing and soft supporters through week 9, pivoted to GOTV on low-propensity supporters for weeks 10-12
- The agent correctly identified the three-phase structure (hire early, persuade mid-campaign, GOTV late) but over-invested in persuasion on swing voters where the literature-calibrated effect is small

For reference, the real Axne won this race by approximately 0.6pp.

## Data

The environment uses `groundgame_dataset.csv`, a 97-row dataset of competitive US House races from 2002 to 2024. Each row is one race with 42 fields covering election results, district demographics, voter segment estimates, national context, and canvassing operational parameters.

The election results (votes, margins, candidates, outcomes) are sourced from the MIT Election Data and Science Lab's US House returns and public reporting. Demographic and segment fields are calibrated to realistic distributions based on ACS and voter-file aggregates. Canvassing operational parameters (doors per hour, contact rates) are drawn from the field experiment literature.

95 races are designated for training, 2 for testing. Each race generates 3 seeds, giving 285 train tasks and 6 test tasks.

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

`rollout.py` runs Claude through one full 12-week episode and prints per-week rewards and cumulative result.

## Repo structure

```
door2door/
  server.py               environment definition, simulator, tools
  groundgame_dataset.csv   97 competitive US House races, 2002-2024
  Dockerfile               used by OpenReward to build the container
  requirements.txt
rollout.py                 runs a model against the deployed env
test_env.py                quick check that the env responds
```

## Tasks

97 competitive US House races across 12 election cycles (2002-2024), drawn from races decided by less than 10 percentage points. Each race is run with 3 random seeds for polling and election-day noise variation.

Train (95 races x 3 seeds = 285 tasks): includes IA-03 2018 (Axne, won by 0.3%), VA-07 2018 (Spanberger), IA-02 2020 (Hart, lost by 6 votes), CO-03 2022 (Boebert, won by 546 votes), and 91 others spanning wave years, open seats, and incumbents of both parties across 33 states.

Test (2 races x 3 seeds = 6 tasks): CO-03 2022 and PA-08 2024, held out for evaluation.

## Reward

```
r_t = E_t[vote_share] - E_{t-1}[vote_share]
```

Dense reward: computed every `advance_week` call. Cumulative episode reward telescopes to final realised share minus baseline. The final week includes one noise draw (σ = 0.02) to simulate election-day variance. No LLM grader — all rewards are computed from the simulator's internal state.

## Voter segments

The agent targets six voter segments, each with distinct persuasion and GOTV response profiles:

| Segment | Persuasion effect | GOTV effect | Strategic role |
|---|---|---|---|
| hard_opp | 0.0 | -0.040 (counter-productive) | Avoid |
| soft_opp | -0.005 (backfire) | -0.020 (counter-productive) | Avoid |
| swing | +0.030 | +0.005 | Primary persuasion target |
| soft_sup | +0.005 | +0.015 | Secondary persuasion target |
| hard_sup | 0.0 | 0.0 (already saturated) | No action needed |
| low_prop_sup | 0.0 | +0.045 | Primary GOTV target (final 2 weeks only) |

## Tools

| Tool | Advances time | Purpose |
|---|---|---|
| view_state | No | Current week, canvasser count, hours available, latest poll |
| hire_canvassers(count) | No | Hire canvassers; productive after 1 week training |
| assign_canvassers(hours, mode, target_segment) | No | Deploy hours in persuasion or gotv mode to a segment |
| log(note) | No | Record strategic reasoning (no simulation effect) |
| advance_week | Yes | Settles the week, generates a new poll, returns reward |

## Key simulator parameters

- Canvasser productivity curve: 0% (hire week), 40% (week 1), 70% (week 2), 100% (week 3+)
- Doors per hour: 15 (rural), 20 (suburban), 25 (urban)
- Contact rate: 0.22 (urban), 0.28 (suburban), 0.35 (rural)
- Persuasion half-life: 3 weeks (contacts decay toward election day)
- GOTV window: final 2 weeks only (earlier GOTV contacts have no effect)
- Salience damping: 0.5x (high-salience House races attenuate all effects)
- Poll noise: ±2pp standard deviation per weekly poll

## Citations

Kalla, J. L., and Broockman, D. E. (2018). The Minimal Persuasive Effects of Campaign Contact in General Elections: Evidence from 49 Field Experiments. *American Political Science Review* 112(1), 148-166.

Gerber, A. S., and Green, D. P. (2000). The Effects of Canvassing, Telephone Calls, and Direct Mail on Voter Turnout: A Field Experiment. *American Political Science Review* 94(3), 653-663.

Green, D. P., McGrath, M. C., and Aronow, P. M. (2013). Field experiments and the study of voter turnout. *Journal of Elections, Public Opinion and Parties* 23(1), 27-48.

Bhatti, Y. et al. (2024). A meta-analysis of voter mobilization tactics by electoral salience. *Electoral Studies*.
