import argparse
from quantlab.pipeline.load_config import load_config
from quantlab.pipeline.runner import run_from_config


def main() -> None:
    arg_parser = argparse.ArgumentParser(description="Run a QuantLab backtest from a YAML config.")
    arg_parser.add_argument("config", help="e.g. configs/momentum_dow30.yaml")
    out = run_from_config(load_config(arg_parser.parse_args().config))
    print(f"Sharpe {out['stats']['sharpe_ratio']:.3f} | "
          f"perm p-value {out['permutation']['p_value']:.3f}")


if __name__ == "__main__":
    main()