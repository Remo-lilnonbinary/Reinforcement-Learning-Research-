from typing import List
from pydantic import BaseModel
from openreward.environments import (
    Environment, JSONObject, Split, TextBlock, ToolOutput, tool
)


class AdvanceWeekParams(BaseModel):
    class Config:
        extra = "forbid"


class Door2Door(Environment):
    def __init__(self, task_spec: JSONObject, secrets: dict = None):
        super().__init__(task_spec=task_spec, secrets=secrets or {})
        self.race_id = task_spec.get("race_id", "IA_3_2018")
        self.candidate = task_spec.get("candidate", "Axne")
        self.weeks_total = task_spec.get("weeks", 12)
        self.baseline_share = task_spec.get("baseline_share", 0.487)
        self.current_week = 0
        self.share = self.baseline_share

    @classmethod
    def list_splits(cls):
        return [
            Split(name="train", type="train"),
            Split(name="test", type="test"),
        ]

    @classmethod
    def list_tasks(cls, split: str):
        if split == "train":
            return [
                {"race_id": "IA_3_2018", "candidate": "Axne", "weeks": 12, "baseline_share": 0.487},
            ]
        if split == "test":
            return [
                {"race_id": "PA_8_2022", "candidate": "Cartwright", "weeks": 12, "baseline_share": 0.490},
            ]
        return []

    def get_prompt(self) -> List[TextBlock]:
        text = (
            f"You're running the ground game for {self.candidate} in the {self.race_id} US House race. "
            f"You have {self.weeks_total} weeks until election day. Starting share: {self.baseline_share:.3f}. "
            f"Call advance_week to move forward. Reward each week is the change in expected vote share."
        )
        return [TextBlock(text=text)]

    @tool
    def advance_week(self, params: AdvanceWeekParams) -> ToolOutput:
        prev = self.share
        self.current_week += 1
        reward = self.share - prev
        finished = self.current_week >= self.weeks_total
        msg = f"Week {self.current_week}/{self.weeks_total}. Share: {self.share:.3f}."
        return ToolOutput(
            blocks=[TextBlock(text=msg)],
            reward=reward,
            finished=finished,
        )


if __name__ == "__main__":
    from openreward.environments import Server
    Server([Door2Door]).run(port=8080)