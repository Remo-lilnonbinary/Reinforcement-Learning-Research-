import math
import random
from pathlib import Path
from typing import List, Optional

import pandas as pd
from pydantic import BaseModel, Field
from openreward.environments import (
    Environment, JSONObject, Split, TextBlock, ToolOutput, tool
)


SEGMENTS = ["hard_opp", "soft_opp", "swing", "soft_sup", "hard_sup_high", "hard_sup_low"]

# Vote share shift per contact. Kalla & Broockman 2018: persuasion is near-zero
# for general elections, with measurable effect concentrated in swing voters.
PERSUASION = {
    "hard_opp": 0.0,
    "soft_opp": -0.005,
    "swing": 0.030,
    "soft_sup": 0.005,
    "hard_sup_high": 0.0,
    "hard_sup_low": 0.0,
}

# Turnout-derived vote share lift per contact, for contacts inside the GOTV window.
# Gerber & Green 2000: ~6pp turnout lift on receptive segments. Hard partisans
# negative because mobilising them is wasted *or* turns out their opponents.
GOTV = {
    "hard_opp": -0.040,
    "soft_opp": -0.020,
    "swing": 0.005,
    "soft_sup": 0.015,
    "hard_sup_high": 0.0,
    "hard_sup_low": 0.045,
}

# Bhatti et al 2024 meta-analysis: high-salience races attenuate effects 33-76%.
SALIENCE = 0.5

# Persuasion effects fade. 3-week half-life is the central estimate from
# replication studies summarised in Kalla & Broockman.
HALF_LIFE = 3.0

# GOTV only counts if contact is within this many weeks of election day.
GOTV_WINDOW = 2

DOORS_PER_HOUR = 30
CONTACT_RATE = 0.35
HOURS_PER_CANVASSER = 30

# Productivity by weeks of experience. Brand new = useless that first week.
PRODUCTIVITY = [0.0, 0.4, 0.7, 1.0]

POLL_NOISE = 0.02


class _Empty(BaseModel):
    class Config:
        extra = "forbid"


class HireParams(BaseModel):
    count: int = Field(..., ge=1, le=200)
    class Config:
        extra = "forbid"


class AssignParams(BaseModel):
    precinct_id: str
    hours: float = Field(..., gt=0, le=500)
    mode: str
    target_segment: str
    class Config:
        extra = "forbid"


class ViewPrecinctParams(BaseModel):
    precinct_id: str
    class Config:
        extra = "forbid"


class LogParams(BaseModel):
    note: str = Field(..., max_length=2000)
    class Config:
        extra = "forbid"


class Door2Door(Environment):
    def __init__(self, task_spec: JSONObject, secrets: dict = None):
        super().__init__(task_spec=task_spec, secrets=secrets or {})
        self.race_id = task_spec.get("race_id", "IA_3_2018")
        self.candidate = task_spec.get("candidate", "Axne")
        self.weeks_total = task_spec.get("weeks", 12)
        self.baseline_share = task_spec.get("baseline_share", 0.487)
        seed = task_spec.get("seed", 0)
        self.rng = random.Random(hash((self.race_id, seed)) & 0xFFFFFFFF)

        path = Path("data") / f"{self.race_id}.parquet"
        if not path.exists():
            path = Path("/orwd_data") / f"{self.race_id}.parquet"
        self.precincts = pd.read_parquet(path).set_index("precinct_id").to_dict("index")
        self.total_voters = sum(p["n_voters"] for p in self.precincts.values())

        self.week = 0
        self.canvassers = []  # list of {hire_week}
        self.contacts = []    # list of {week, precinct_id, mode, segment, contacts}
        self.polls = []       # list of (week, observed_share)

    @classmethod
    def list_splits(cls):
        return [Split(name="train", type="train"), Split(name="test", type="test")]

    @classmethod
    def list_tasks(cls, split: str):
        train = [
            ("IA_3_2018", "Axne", 0.487),
            ("VA_7_2018", "Spanberger", 0.491),
            ("AZ_1_2020", "OHalleran", 0.490),
            ("IA_2_2020", "Hart", 0.499),
            ("CO_3_2022", "Frisch", 0.485),
        ]
        test = [
            ("PA_8_2022", "Cartwright", 0.490),
            ("MI_10_2022", "Marlinga", 0.488),
        ]
        races = train if split == "train" else test if split == "test" else []
        return [
            {"race_id": r, "candidate": c, "weeks": 12, "baseline_share": s, "seed": k}
            for r, c, s in races for k in range(5)
        ]

    # ---- simulator ----

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
        baseline = sum(p["baseline_lean"] * p["n_voters"] for p in self.precincts.values()) / self.total_voters
        share = baseline

        for c in self.contacts:
            voters = self.precincts[c["precinct_id"]]["n_voters"]
            weight = voters / self.total_voters
            wks_to_election = self.weeks_total - c["week"]

            if c["mode"] == "persuasion":
                decay = 0.5 ** (max(0, wks_to_election) / HALF_LIFE)
                effect = PERSUASION.get(c["segment"], 0.0) * c["contacts"] * decay * SALIENCE
                share += effect * weight / max(voters, 1)
            elif c["mode"] == "gotv" and wks_to_election <= GOTV_WINDOW:
                effect = GOTV.get(c["segment"], 0.0) * c["contacts"] * SALIENCE
                share += effect * weight / max(voters, 1)

        return max(0.0, min(1.0, share))

    # ---- prompt ----

    def get_prompt(self) -> List[TextBlock]:
        return [TextBlock(text=(
            f"You're running the ground game for {self.candidate} in {self.race_id}. "
            f"You have {self.weeks_total} weeks. Starting share: {self.baseline_share:.3f}. "
            f"There are {len(self.precincts)} precincts. "
            f"Each canvasser becomes productive after a week of training and contributes ~{HOURS_PER_CANVASSER} hours/wk at full productivity.\n\n"
            f"Tools:\n"
            f"  view_state: current week, hires, hours, latest poll\n"
            f"  view_precinct(id): demographics + segment mix for one precinct\n"
            f"  hire_canvassers(count): hires; productive next week\n"
            f"  assign_canvassers(precinct_id, hours, mode, target_segment): deploy hours\n"
            f"      mode='persuasion' shifts swing voters; effect decays with 3-week half-life\n"
            f"      mode='gotv' boosts turnout but only counts in the final 2 weeks\n"
            f"      target_segment is one of: {', '.join(SEGMENTS)}\n"
            f"  log(note): record reasoning, no effect\n"
            f"  advance_week: settles the week, drops a poll, returns reward = change in expected share.\n\n"
            f"Reward = change in expected vote share each week. Cumulative reward = final share - baseline."
        ))]

    # ---- tools ----

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
    def view_precinct(self, params: ViewPrecinctParams) -> ToolOutput:
        if params.precinct_id not in self.precincts:
            return ToolOutput(blocks=[TextBlock(text=f"unknown precinct: {params.precinct_id}")],
                              reward=0.0, finished=False)
        p = self.precincts[params.precinct_id]
        contacts_here = sum(c["contacts"] for c in self.contacts if c["precinct_id"] == params.precinct_id)
        text = (
            f"{params.precinct_id} ({p['density']}, {p['n_voters']} voters)\n"
            f"  baseline lean: {p['baseline_lean']:.3f}\n"
            f"  segments:\n"
            f"    hard_opp:      {p['hard_opp']:.3f}\n"
            f"    soft_opp:      {p['soft_opp']:.3f}\n"
            f"    swing:         {p['swing']:.3f}\n"
            f"    soft_sup:      {p['soft_sup']:.3f}\n"
            f"    hard_sup_high: {p['hard_sup_high']:.3f}\n"
            f"    hard_sup_low:  {p['hard_sup_low']:.3f}\n"
            f"  contacts here so far: {contacts_here:.0f}"
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
        if params.precinct_id not in self.precincts:
            return ToolOutput(blocks=[TextBlock(text=f"unknown precinct: {params.precinct_id}")],
                              reward=0.0, finished=False)

        avail = self._hours_available() - self._hours_used()
        if params.hours > avail + 0.01:
            return ToolOutput(
                blocks=[TextBlock(text=f"only {avail:.0f} hours available, requested {params.hours:.0f}")],
                reward=0.0, finished=False
            )

        p = self.precincts[params.precinct_id]
        seg_share = p[params.target_segment]
        contacts = params.hours * DOORS_PER_HOUR * CONTACT_RATE * seg_share
        self.contacts.append({
            "week": self.week,
            "precinct_id": params.precinct_id,
            "mode": params.mode,
            "segment": params.target_segment,
            "hours": params.hours,
            "contacts": contacts,
        })
        return ToolOutput(
            blocks=[TextBlock(text=f"assigned {params.hours:.0f}h to {params.precinct_id} [{params.mode}/{params.target_segment}]: ~{contacts:.0f} contacts")],
            reward=0.0, finished=False,
        )

    @tool
    def log(self, params: LogParams) -> ToolOutput:
        return ToolOutput(blocks=[TextBlock(text="noted")], reward=0.0, finished=False)

    @tool
    def advance_week(self, params: _Empty) -> ToolOutput:
        prev = self._expected_share()
        self.week += 1
        finished = self.week >= self.weeks_total

        if finished:
            # one realisation draw at election day
            new = max(0.0, min(1.0, self.rng.gauss(self._expected_share(), POLL_NOISE)))
        else:
            new = self._expected_share()

        reward = new - prev

        if not finished:
            poll = max(0.0, min(1.0, self.rng.gauss(new, POLL_NOISE)))
            self.polls.append((self.week, poll))
            txt = f"week {self.week}/{self.weeks_total}. reward {reward:+.4f}. poll: {poll:.3f}"
        else:
            txt = f"election day. final share: {new:.3f}. baseline was {self.baseline_share:.3f}. reward {reward:+.4f}"

        return ToolOutput(blocks=[TextBlock(text=txt)], reward=reward, finished=finished)


if __name__ == "__main__":
    from openreward.environments import Server
    Server([Door2Door]).run(port=8080)