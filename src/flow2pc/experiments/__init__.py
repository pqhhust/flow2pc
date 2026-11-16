"""Evaluation-only research experiments; imports never start model work."""

from importlib import import_module

EXPERIMENTS = {"e3_1": "flow2pc.experiments.e3_1"}


def run(config):
    module = import_module(EXPERIMENTS[config["experiment"]] + ".runner")
    return module.run(config)
