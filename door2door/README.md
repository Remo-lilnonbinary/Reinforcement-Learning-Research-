# door2door

A long-horizon RL environment that tests whether agents can run an effective ground-game allocation for a US House race. 

The agent has 12 weeks until election day. It hires canvassers, allocates their hours across precincts, and chooses whether to persuade swing voters or mobilise low-propensity supporters. The reward each week is the change in expected vote share. Cumulative reward over an episode equals the final realised share minus the baseline.

The simulator is calibrated to published causal estimates from the political science literature. Persuasion effects are modelled on Kalla and Broockman's 2018 meta-analysis. GOTV effects come from Gerber and Green's 2000 New Haven study and the Green, McGrath and Aronow 2013 review. Effect sizes are damped to reflect the lower marginal returns observed in high-salience races (Bhatti et al. 2024).

## Why this is interesting

Door-to-door is one of the few campaign tactics with rigorous causal evidence. Persuasion barely works in general elections, GOTV only works inside a narrow window before election day, and timing matters more than total spend. The agent has to discover all this through experimentation. Frontier models tend to fail by hiring too late, persuading hard partisans, or running GOTV too early.

## Running the env locally

You need Python 3.11 or higher.

```
pip install -r requirements.txt
python make_data.py
python server.py
```

The server runs on port 8080. The synthetic precinct data is generated once and saved to `data/`.

## Running an agent against the deployed env

The env is deployed at `Remo/door2door` on OpenReward. From the parent folder:

```
pip install openreward anthropic python-dotenv
export OPENREWARD_API_KEY=...
export ANTHROPIC_API_KEY=...
python rollout.py
```

`rollout.py` runs Claude through one full episode and prints per-week rewards.

## Repo structure

```
door2door/
  server.py        environment definition, simulator, tools
  make_data.py     generates synthetic precinct data
  data/            generated parquet files, one per race
  Dockerfile       used by OpenReward to build the container
  requirements.txt
```

## Tasks

There are 5 training races and 2 test races, each with 5 random seeds, giving 25 train tasks and 10 test tasks.

Train: IA-3 2018 (Axne), VA-7 2018 (Spanberger), AZ-1 2020, IA-2 2020 (Hart), CO-3 2022 (Frisch).

Test: PA-8 2022 (Cartwright), MI-10 2022 (Marlinga).

All seven were close races in real life, decided by margins between 6 votes and a few percentage points.

## Reward

```
r_t = E_t[share] - E_{t-1}[share]
```

Cumulative episode reward is the change from baseline to final realised share. Final-week share includes one realisation noise draw to simulate election-day variance. No LLM grader.

## Citations

Kalla, J. L., and Broockman, D. E. (2018). The Minimal Persuasive Effects of Campaign Contact in General Elections. APSR 112(1).

Gerber, A. S., and Green, D. P. (2000). The Effects of Canvassing, Telephone Calls, and Direct Mail on Voter Turnout. APSR 94(3).

Green, D. P., McGrath, M. C., and Aronow, P. M. (2013). Field experiments and the study of voter turnout. JEPOP 23(1).

Bhatti, Y. et al. (2024). A meta-analysis of voter mobilization tactics by electoral salience. Electoral Studies.
