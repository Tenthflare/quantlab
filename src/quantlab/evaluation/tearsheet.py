from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from quantlab.evaluation.metrics import Metrics

font = {'size' : 8}
plt.style.use("default")
plt.rc('text', usetex=True)
plt.rc('font', **font, family='serif')

def tearsheet(
    records: pd.DataFrame,
    title: str,
    permutation: dict | None = None,
    outpath: str | Path | None = None,
) -> "plt.Figure":
    fwd_returns, turnover = records["pnl"], records["turnover"]
    periods_per_year = 12
    metrics = Metrics(fwd_returns, turnover, periods_per_year)
    stats = metrics.summary()

    fig = plt.figure(figsize=(12, 8), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[2.2, 1.0], height_ratios=[2.0, 1.0])

    ax = fig.add_subplot(gs[0, 0])
    cumulative_return = stats["cumulative_return"]
    ax.plot(cumulative_return.index, cumulative_return.values, lw=1.6, color="green")
    ax.set_ylabel("Cumulative return (net of costs)")
    ax.grid(alpha=0.3)

    ax_dd = fig.add_subplot(gs[1, 0], sharex=ax)  # share x so time axes align
    dd = cumulative_return / cumulative_return.cummax() - 1.0
    ax_dd.fill_between(dd.index, dd.values, 0.0, color="red", alpha=0.4)
    ax_dd.set_title("Drawdown")
    ax_dd.set_ylabel("Peak-to-trough")
    ax_dd.grid(alpha=0.3)

    ax_tbl = fig.add_subplot(gs[:, 1])
    ax_tbl.axis("off")
    rows = [
        ("Annualised return", f"{stats['annualised_return']:.2%}"),
        ("Annualised volatility", f"{stats['annualised_volatility']:.2%}"),
        ("Sharpe ratio", f"{stats['sharpe_ratio']:.2f}"),
        ("Sortino ratio", f"{stats['sortino_ratio']:.2f}"),
        ("Calmar ratio", f"{stats['calmar_ratio']:.2f}"),
        ("Max drawdown", f"{stats['max_drawdown']:.2%}"),
        ("Hit rate", f"{stats['hit_rate']:.1%}"),
        ("Avg. Turnover", f"{stats['avg_turnover']:.2f}"),
        ("Num. Rebalance", f"{stats['n_periods']}"),
    ]
    if permutation is not None:
        rows += [
            ("—", "—"),
            ("Perm. p-value", f"{permutation['p_value']:.3f}"),
            ("Null mean PnL", f"{permutation['null_mean']:.2e}"),
        ]
    tbl = ax_tbl.table(
        cellText=rows, colLabels=["Metric", "Value"], loc="center", cellLoc="left", colLoc="left"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 1.5)
    ax_tbl.set_title("Performance summary", pad=12)

    fig.suptitle(title, fontsize=15, fontweight="bold")

    if outpath is not None:
        outpath = Path(outpath)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, dpi=150, bbox_inches="tight")
    return fig
