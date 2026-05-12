import math
import random
from pathlib import Path
from typing import List

import pandas as pd
from pydantic import BaseModel, Field
from openreward.environments import (
    Environment, JSONObject, Split, TextBlock, ToolOutput, tool
)


SEGMENTS = ["hard_opp", "soft_opp", "swing", "soft_sup", "hard_sup", "low_prop_sup"]

# The CSV uses dem-coded column names. Map to neutral candidate-side names
# depending on whether the agent is running the dem or rep campaign.
CSV_DEM_COLS = {
    "hard_opp": "pct_hard_rep",
    "soft_opp": "pct_soft_rep",
    "swing": "pct_swing",
    "soft_sup": "pct_soft_dem",
    "hard_sup": "pct_hard_dem",
    "low_prop_sup": "pct_low_prop_dem",
}
CSV_REP_COLS = {
    "hard_opp": "pct_hard_dem",
    "soft_opp": "pct_soft_dem",
    "swing": "pct_swing",
    "soft_sup": "pct_soft_rep",
    "hard_sup": "pct_hard_rep",
    "low_prop_sup": "pct_low_prop_dem",  # csv only has dem low-prop; use as proxy
}

# Vote share shift per contact. Kalla & Broockman 2018: persuasion is near-zero
# for general elections, with measurable effect concentrated in swing voters.
PERSUASION = {
    "hard_opp": 0.0,
    "soft_opp": -0.10,
    "swing": 0.60,
    "soft_sup": 0.10,
    "hard_sup": 0.0,
    "low_prop_sup": 0.0,
}

# Turnout-derived vote share lift per contact, for contacts in the GOTV window.
# Gerber & Green 2000: ~6pp turnout lift on receptive segments.
GOTV = {
    "hard_opp": -0.80,
    "soft_opp": -0.40,
    "swing": 0.10,
    "soft_sup": 0.30,
    "hard_sup": 0.0,
    "low_prop_sup": 0.90,
}

SALIENCE = 0.5
HALF_LIFE = 3.0
GOTV_WINDOW = 2
HOURS_PER_CANVASSER = 30
PRODUCTIVITY = [0.0, 0.4, 0.7, 1.0]
POLL_NOISE = 0.02

CSV_PATH = Path(__file__).parent / "groundgame_dataset.csv"


def _load_races(split: str) -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    return df[df["split"] == split].reset_index(drop=True)


class _Empty(BaseModel):
    class Config:
        extra = "forbid"


class HireParams(BaseModel):
    count: int = Field(..., ge=1, le=500)
    class Config:
        extra = "forbid"


class AssignParams(BaseModel):
    hours: float = Field(..., gt=0, le=50000)
    mode: str
    target_segment: str
    class Config:
        extra = "forbid"


class LogParams(BaseModel):
    note: str = Field(..., max_length=2000)
    class Config:
        extra = "forbid"


class Door2Door(Environment):
    def __init__(self, task_spec: JSONObject, secrets: dict = None):
        super().__init__(task_spec=task_spec, secrets=secrets or {})

        self.district_id = task_spec["district_id"]
        self.year = int(task_spec["year"])
        seed = int(task_spec.get("seed", 0))
        self.weeks_total = int(task_spec.get("weeks", 12))
        self.rng = random.Random(hash((self.district_id, self.year, seed)) & 0xFFFFFFFF)

        df = pd.read_csv(CSV_PATH)
        row = df[(df["district_id"] == self.district_id) & (df["year"] == self.year)]
        if row.empty:
            raise ValueError(f"no race found: {self.district_id} {self.year}")
        r = row.iloc[0]

        self.dem_candidate = r["dem_candidate"]
        self.rep_candidate = r["rep_candidate"]
        self.party = task_spec.get("party", "dem")
        self.candidate = self.dem_candidate if self.party == "dem" else self.rep_candidate
        self.opponent = self.rep_candidate if self.party == "dem" else self.dem_candidate
        self.state = r["state"]
        self.district = int(r["district"])
        self.midterm = bool(r["midterm"])
        self.is_open = bool(r["is_open_seat"])
        self.incumbent_party = r["incumbent_party"]
        self.cook_pvi = float(r["cook_pvi_approx"])
        self.competitiveness = float(r["competitiveness"])
        self.median_income = int(r["median_hh_income"])
        self.pct_bachelors = float(r["pct_bachelors_plus"])
        self.pct_white = float(r["pct_white"])
        self.pct_urban = float(r["pct_urban"])
        self.pop_density = float(r["pop_density_sqmi"])
        self.doors_per_hour = float(r["est_doors_per_hour"])
        self.contact_rate = float(r["est_contact_rate"])
        self.total_voters = int(r["est_vap"])
        self.actual_dem_share = float(r["dem_share"])

        # baseline = dem share before any campaigning, taken from generic ballot
        # at midpoint of the historical campaign period
        if self.party == "dem":
            self.baseline_share = self.actual_dem_share
        else:
            self.baseline_share = 1.0 - self.actual_dem_share

        col_map = CSV_DEM_COLS if self.party == "dem" else CSV_REP_COLS
        self.segments = {seg: float(r[col]) for seg, col in col_map.items()}

        self.week = 0
        self.canvassers = []
        self.contacts = []
        self.polls = []

    @classmethod
    def list_splits(cls):
        return [Split(name="train", type="train"), Split(name="test", type="test")]

    @classmethod
    def list_tasks(cls, split: str):
        df = _load_races(split)
        tasks = []
        for _, r in df.iterrows():
            for seed in range(3):
                tasks.append({
                    "district_id": r["district_id"],
                    "year": int(r["year"]),
                    "weeks": 12,
                    "party": "dem",
                    "seed": seed,
                })
        return tasks

    def _hours_available(self):
        total = 0.0
        for c in self.canvassers:
            xp = self.week - c["hire_week"]
            if xp < 0:
                continue
            p = PRODUCTIVITY[min(xp, len(PRODUCTIVITY) - 1)]
            total += HOURS_PER_CANVASSER * p
        return total

    def _hours_used(self):
        return sum(c["hours"] for c in self.contacts if c["week"] == self.week)

    def _expected_share(self):
        turnout_estimate = 0.55
        effective_voters = self.total_voters * turnout_estimate

        share = self.baseline_share
        for c in self.contacts:
            wks_to_election = self.weeks_total - c["week"]
            weeks_since = self.week - c["week"]
            if c["mode"] == "persuasion":
                decay = 0.5 ** (weeks_since / HALF_LIFE)
                effect = PERSUASION.get(c["segment"], 0.0)
                share += (c["contacts"] * effect * decay) / effective_voters
            elif c["mode"] == "gotv" and wks_to_election <= GOTV_WINDOW:
                effect = GOTV.get(c["segment"], 0.0)
                share += (c["contacts"] * effect) / effective_voters
        return max(0.0, min(1.0, share))

    def get_prompt(self) -> List[TextBlock]:
        seg_lines = "\n".join(f"  {k}: {v:.3f}" for k, v in self.segments.items())
        return [TextBlock(text=(
            f"You are running the ground game for {self.candidate} ({self.party.upper()}) "
            f"in {self.district_id}, {self.year}. The opponent is {self.opponent}. "
            f"You have {self.weeks_total} weeks until election day.\n\n"
            f"District profile:\n"
            f"  state: {self.state}, district: {self.district}\n"
            f"  midterm: {self.midterm}, open seat: {self.is_open}, incumbent: {self.incumbent_party}\n"
            f"  Cook PVI: {self.cook_pvi:+.1f} (positive = favours dems)\n"
            f"  competitiveness score: {self.competitiveness:.2f}\n"
            f"  voting-age population: {self.total_voters:,}\n"
            f"  density: {self.pop_density:.0f}/sqmi, % urban: {self.pct_urban:.2f}\n"
            f"  median income: ${self.median_income:,}, % bachelors: {self.pct_bachelors:.2f}\n"
            f"  doors/hr per canvasser: {self.doors_per_hour:.0f}\n"
            f"  contact rate: {self.contact_rate:.2f}\n\n"
            f"Voter segments (your candidate's perspective):\n{seg_lines}\n\n"
            f"Starting expected share: {self.baseline_share:.3f}\n\n"
            f"You must complete all {self.weeks_total} weeks. Take the next action immediately after each result. "
            f"Never wait for confirmation. Your job is to maximise final vote share.\n\n"
            f"Tools:\n"
            f"  view_state: current week, hires, hours, latest poll\n"
            f"  hire_canvassers(count): productive after 1 week of training\n"
            f"  assign_canvassers(hours, mode, target_segment): deploy hours\n"
            f"      mode = 'persuasion' or 'gotv'\n"
            f"      target_segment one of: {', '.join(SEGMENTS)}\n"
            f"  log(note): record reasoning\n"
            f"  advance_week: settle the week, drop a poll, return reward\n\n"
            f"Reward = change in expected vote share each week. Begin now."
        ))]

    @tool
    def view_state(self, params: _Empty) -> ToolOutput:
        latest = self.polls[-1] if self.polls else (0, self.baseline_share)
        text = (
            f"week {self.week}/{self.weeks_total}\n"
            f"canvassers hired: {len(self.canvassers)}\n"
            f"productive hours this week: {self._hours_available():.0f}\n"
            f"hours used this week: {self._hours_used():.0f}\n"
            f"latest poll (wk {latest[0]}): {latest[1]:.3f} (~3pp moe)\n"
            f"contacts to date: {sum(c['contacts'] for c in self.contacts):.0f}"
        )
        return ToolOutput(blocks=[TextBlock(text=text)], reward=0.0, finished=False)

    @tool
    def hire_canvassers(self, params: HireParams) -> ToolOutput:
        for _ in range(params.count):
            self.canvassers.append({"hire_week": self.week})
        text = f"hired {params.count}. total: {len(self.canvassers)}. productive next week."
        return ToolOutput(blocks=[TextBlock(text=text)], reward=0.0, finished=False)

    @tool
    def assign_canvassers(self, params: AssignParams) -> ToolOutput:
        if params.mode not in ("persuasion", "gotv"):
            return ToolOutput(blocks=[TextBlock(text=f"mode must be persuasion or gotv, got {params.mode}")],
                              reward=0.0, finished=False)
        if params.target_segment not in SEGMENTS:
            return ToolOutput(blocks=[TextBlock(text=f"segment must be one of {SEGMENTS}")],
                              reward=0.0, finished=False)

        avail = self._hours_available() - self._hours_used()
        if params.hours > avail + 0.01:
            return ToolOutput(
                blocks=[TextBlock(text=f"only {avail:.0f} hours available, requested {params.hours:.0f}")],
                reward=0.0, finished=False
            )

        seg_share = self.segments[params.target_segment]
        contacts = params.hours * self.doors_per_hour * self.contact_rate * seg_share
        self.contacts.append({
            "week": self.week,
            "mode": params.mode,
            "segment": params.target_segment,
            "hours": params.hours,
            "contacts": contacts,
        })
        return ToolOutput(
            blocks=[TextBlock(text=f"assigned {params.hours:.0f}h [{params.mode}/{params.target_segment}]: ~{contacts:.0f} contacts")],
            reward=0.0, finished=False,
        )

    @tool
    def log(self, params: LogParams) -> ToolOutput:
        return ToolOutput(blocks=[TextBlock(text="noted")], reward=0.0, finished=False)

    @tool
    def advance_week(self, params: _Empty) -> ToolOutput:
        self.week += 1
        finished = self.week >= self.weeks_total

        if finished:
            new = max(0.0, min(1.0, self.rng.gauss(self._expected_share(), POLL_NOISE)))
        else:
            new = self._expected_share()

        reward = new - self.baseline_share

        if not finished:
            poll = max(0.0, min(1.0, self.rng.gauss(new, POLL_NOISE)))
            self.polls.append((self.week, poll))
            txt = f"week {self.week}/{self.weeks_total}. reward {reward:+.4f}. poll: {poll:.3f}"
        else:
            txt = (f"election day. final share: {new:.3f}. baseline was {self.baseline_share:.3f}. "
                   f"actual historical dem share: {self.actual_dem_share:.3f}. reward {reward:+.4f}")

        return ToolOutput(blocks=[TextBlock(text=txt)], reward=reward, finished=finished)


if __name__ == "__main__":
    from openreward.environments import Server
    Server([Door2Door]).run(port=8080)
