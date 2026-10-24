"""
Staged (Two-Phase) Execution
============================

In the base router the output check G4 runs after an action has executed, so
it can only detect a bad output. For actions with external effects ORASR
supports two remedies:

* **Irreversible actions** are wrapped in :class:`StagedAction`. The router
  calls ``prepare`` (which must have no external effect, e.g. build the order
  message), runs G4 on the *staged* output, and calls ``commit`` only if G4
  passes; otherwise ``abort`` is called and nothing is committed.
* **Reversible actions** may supply ``compensate``; if G4 fails after
  execution the router invokes it (e.g. discontinue the order just placed) and
  records the compensation in the result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass
class StagedAction:
    prepare: Callable[[Any], Any]
    commit: Callable[[Any], Any]
    abort: Optional[Callable[[Any], None]] = None
    reversible: bool = False

    def __call__(self, input_data: Any) -> Any:  # legacy (unstaged) behaviour
        return self.commit(self.prepare(input_data))


@dataclass
class CompensableAction:
    run: Callable[[Any], Any]
    compensate: Callable[[Any], None]
    reversible: bool = True

    def __call__(self, input_data: Any) -> Any:
        return self.run(input_data)
