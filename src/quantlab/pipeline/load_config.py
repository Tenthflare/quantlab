from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class RunConfig:
    run_name: str
    start_date: str
    universe: str
    storage: str
    lookback: int
    lookback_end: int
    k: int
    cost_bps: float
    n_permutation: int
    seed: int
    output_dir: str


def load_config(path) -> RunConfig:
    config = yaml.safe_load(Path(path).read_text())
    feature, portfolio, permutation = (
        config.get("feature", {}),
        config.get("portfolio", {}),
        config.get("permutation", {}),
    )
    return RunConfig(
        run_name=config["run_name"],
        start_date=config["start_date"],
        universe=config["universe"],
        storage=config["storage"],
        lookback=feature["lookback"],
        lookback_end=feature["lookback_end"],
        k=portfolio["k"],
        cost_bps=config["cost_bps"],
        n_permutation=permutation["n"],
        seed=permutation["seed"],
        output_dir=config["output_dir"],
    )
