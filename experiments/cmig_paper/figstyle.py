"""Figure style shared by the plotting scripts (Okabe-Ito palette, colour-blind safe)."""

# Okabe-Ito base palette (CVD-safe)
OKABE = {
    "blue":      "#0072B2",
    "orange":    "#E69F00",
    "green":     "#009E73",
    "vermillion":"#D55E00",
    "purple":    "#CC79A7",
    "sky":       "#56B4E9",
    "yellow":    "#F0E442",
    "black":     "#000000",
    "gray":      "#999999",
}

# per-patient colours, kept the same across all multi-patient figures
PATIENT = {
    "2-V2":  OKABE["blue"],
    "20-V1": OKABE["orange"],
    "30-V2": OKABE["green"],
    "9-V2":  OKABE["purple"],
    "50-V2": OKABE["sky"],
}

ROLE = {"ct": "#333333"}   # CT reference curves

# text and axis colours
INK        = "#262626"
INK_MUTED  = "#595959"
SPINE      = "#666666"

FONT_STACK = ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"]


def apply_style():
    """Set global matplotlib rcParams. Call once at the top of each figure script."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": FONT_STACK,
        "font.size": 11,
        "axes.titlesize": 12.5,
        "axes.labelsize": 11,
        "axes.labelcolor": INK,
        "axes.edgecolor": SPINE,
        "axes.linewidth": 0.9,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "text.color": INK,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 190,
    })
