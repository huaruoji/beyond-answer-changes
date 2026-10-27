"""Descriptive figures from the shared confirm predictions; no mock data."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/evidence/evidence_table.json"
OUT = ROOT / "paper/figures"

# The manuscript requires Times New Roman. It is not installed everywhere (a
# stock Linux box usually has only metric-compatible substitutes), so resolve it
# at run time instead of hard-failing: the substitution is reported loudly
# because figures rendered in a substitute must not be shipped as final.
SERIF_PREFERENCE = (
    "Times New Roman",
    "Times",
    "Liberation Serif",   # metric-compatible with Times New Roman
    "Nimbus Roman",       # metric-compatible with Times New Roman
    "DejaVu Serif",
)


def resolve_font(requested: str | None) -> str:
    candidates = (requested,) if requested else SERIF_PREFERENCE
    for name in candidates:
        try:
            font_manager.findfont(name, fallback_to_default=False)
        except ValueError:
            continue
        if name != "Times New Roman":
            print(
                f"NOTE: rendering with {name!r}, not 'Times New Roman'.\n"
                "      Metrics differ slightly from the shipped figures, so this\n"
                "      is fine for layout checks but not for submission."
            )
        return name
    raise SystemExit(
        "no usable serif font found; tried "
        + ", ".join(repr(c) for c in candidates)
        + ". Install Times New Roman (ttf-mscorefonts-installer) or pass --font."
    )


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--font", help="font family to render with (default: Times New Roman)")
parser.add_argument(
    "--out",
    type=Path,
    default=OUT,
    help="output directory (default: paper/figures)",
)
args = parser.parse_args()

FONT = resolve_font(args.font)
OUT = args.out
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({
    "font.family": FONT, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#888888", "axes.linewidth": .6,
    "xtick.color": "#444444", "ytick.color": "#444444",
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "legend.frameon": False,
})
table = json.loads(SOURCE.read_text())
models = [table["models"][k] for k in ("r2_qwen7b", "r2_qwen3b", "fun_audio_chat_8b")]
colors = ["#3B6EA5", "#86ABC5", "#B76B63"]
markers = ["o", "s", "^"]


def save(fig, stem):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # Check all visible text against the exported canvas before saving.
    for text in fig.findobj(matplotlib.text.Text):
        if text.get_visible() and text.get_text():
            box = text.get_window_extent(renderer)
            assert box.x0 >= -1 and box.y0 >= -1, (stem, text.get_text())
            assert box.x1 <= fig.bbox.width+1 and box.y1 <= fig.bbox.height+1, (stem, text.get_text())
    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=450)
    svg = OUT / f"{stem}.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines())+"\n")
    plt.close(fig)


# Grouped bars with flat fills; model colors match the other figures. The last
# group is the different-class minus same-class answer-change gap, with its
# interval from Table 1.
fields = ["A_diff", "A_same", "A_nuisance_removed", "A_background_removed", "A_silence", "A_sham"]
labels = ["Different\nclass", "Same\nclass", "Nuisance\nremoval", "Background\nwindow", "Silence", "Sham",
          "$\\Delta$\n(95% CI)"]
fig, ax = plt.subplots(figsize=(7, 2.1))
for j, model in enumerate(models):
    values = [100*model["attribution"][key] for key in fields]
    ax.bar(np.arange(6)+(j-1)*.24, values, width=.23,
           color=colors[j], label=model["display"], zorder=3)
    attribution = model["attribution"]
    gap = 100*attribution["selectivity"]
    low, high = (100*np.asarray(attribution["selectivity_ci95"]))
    assert low < gap < high, (model["display"], low, gap, high)
    ax.bar([6 + (j-1)*.24], [gap], width=.23, color=colors[j], zorder=3)
    ax.errorbar([6 + (j-1)*.24], [gap], yerr=[[gap-low], [high-gap]],
                fmt="none", ecolor="#333333", elinewidth=.9, capsize=2.4, zorder=4)
ax.set_xticks(np.arange(7), labels)
ax.set_ylim(0, 80)
ax.set_yticks([0, 20, 40, 60, 80])
ax.set_ylabel("Answer changes (%)")
ax.grid(axis="y", color="#E8E8E8", linewidth=.6)
ax.tick_params(axis="x", length=0)
fig.legend(loc="upper center", ncol=3, handletextpad=.4, columnspacing=1.8)
fig.subplots_adjust(left=.08, right=.99, top=.77, bottom=.24)
save(fig, "reference_identity")

# Recompute each class directly from the exact predictions used in the main table.
profiles = []
for model in models:
    path = ROOT / model["follow"]["predictions"]
    views = defaultdict(dict)
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row["split"] == "confirm":
            views[row["pair_id"]][row["condition"]] = row
    groups = defaultdict(list)
    for view in views.values():
        o, d, s = [view[c] for c in ("original", "target_replaced_diff", "target_replaced_same")]
        groups[o["query_class"]].append((int(o["correct"]),
            int(d["parsed_choice"] != o["parsed_choice"])-int(s["parsed_choice"] != o["parsed_choice"])))
    profile = [(name, len(rows), *np.mean(rows, axis=0)) for name, rows in sorted(groups.items())]
    assert len(profile) == 33 and sum(r[1] for r in profile) == 1200
    assert np.isclose(sum(r[1]*r[3] for r in profile)/1200,
                      model["attribution"]["A_diff"]-model["attribution"]["A_same"])
    profiles.append(profile)
    print(model["display"], "33 classes; gap range", min(r[3] for r in profile), max(r[3] for r in profile))

fig, ax = plt.subplots(figsize=(3.39, 2.35))
for model, profile, color, marker in zip(models, profiles, colors, markers):
    ax.scatter([100*r[2] for r in profile], [100*r[3] for r in profile],
               s=23, color=color, marker=marker, alpha=.78, edgecolors="white", linewidths=.35,
               label=model["display"])
ax.set(xlim=(45, 102), ylim=(0, 102), xlabel="Class-level original accuracy (%)",
       ylabel="Answer-change gap (pp)")
ax.set_xticks([50, 75, 100])
ax.set_yticks([0, 25, 50, 75, 100])
ax.grid(color="#EEEEEE", linewidth=.5)
ax.set_axisbelow(True)
fig.legend(loc="upper center", ncol=1, fontsize=9, labelspacing=.15, handletextpad=.3)
fig.subplots_adjust(left=.18, right=.96, top=.71, bottom=.22)
save(fig, "reference_class")

# Equal-class empirical distributions complement the accuracy-versus-gap scatter.
fig, ax = plt.subplots(figsize=(3.39, 2.35))
for model, profile, color, style in zip(models, profiles, colors, ("-", "--", "-.")):
    gaps = np.sort([100*r[3] for r in profile])
    ax.step(np.r_[0, gaps, 100], np.r_[0, np.arange(1, 34)/33, 1],
            where="post", color=color, linewidth=1.7, linestyle=style, label=model["display"])
ax.set(xlim=(0, 100), ylim=(0, 1.03), xlabel="Class-level answer-change gap (pp)",
       ylabel="Cumulative fraction of classes")
ax.set_xticks([0, 25, 50, 75, 100])
ax.set_yticks([0, .5, 1])
ax.grid(color="#EEEEEE", linewidth=.5)
fig.legend(loc="upper center", ncol=1, fontsize=9, labelspacing=.15)
fig.subplots_adjust(left=.18, right=.96, top=.71, bottom=.22)
save(fig, "reference_distribution")

for model, profile in zip(models, profiles):
    gaps = sorted(100*r[3] for r in profile)
    print(f"{model['display']}: {len(gaps)} classes, min gap {gaps[0]:.1f} pp, "
          f"median {gaps[len(gaps)//2]:.1f} pp, max {gaps[-1]:.1f} pp, "
          f"negative {sum(1 for g in gaps if g < 0)}")
print("Source SHA256:", hashlib.sha256(SOURCE.read_bytes()).hexdigest())
print("Rendered with:", FONT, "->", OUT)
