"""
Corner Kick Analysis — StatsBomb Open Data
===========================================
Identifies optimal corner kick delivery zones using StatsBomb event data.

Methodology:
    - Parses corner kick events and classifies deliveries by zone, swing type,
      and recipient body part.
    - Uses chi-squared tests, z-tests, logistic regression, and polynomial
      fitting to determine statistically significant success factors.
    - Zone classification follows the heuristic from:
      https://www.tandfonline.com/doi/full/10.1080/24748668.2019.1677329

Usage:
    python corner_kicks.py

Data:
    Expects StatsBomb-format JSON event files in an `events/` directory.
    Free open-data available at: https://github.com/statsbomb/open-data
"""

import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Rectangle, Arc
from matplotlib.colors import Normalize
from scipy.stats import chi2_contingency
from statsmodels.stats.proportion import proportions_ztest
import statsmodels.formula.api as smf

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EVENTS_DIR = Path("events")
CORNERS_OUTPUT = "corner_deliveries.csv"
GOALS_OUTPUT = "goals.csv"
PLOTS_DIR = Path("plots")

# Success window: goal must occur within this many seconds of the corner
SUCCESS_WINDOW_SEC = 10
# Body-part scan window: look up to this far ahead in the same possession
BP_SCAN_WINDOW_SEC = 8
BP_SCAN_MAX_EVENTS = 40
# Fallback shot scan window
BP_FALLBACK_WINDOW_SEC = 15

# StatsBomb pitch midpoint (used to determine left/right side)
PITCH_MIDPOINT_X = 60

# Zone boundaries (xmin, xmax, ymin, ymax) — based on Casal et al. (2019)
ZONE_COORDS = {
    "Edge": (100, 108, 30, 50),
    "FZ":   (102, 120, 50, 62),
    "BZ":   (102, 120, 18, 30),
    "GA1":  (114, 120, 44, 50),
    "GA2":  (114, 120, 36, 44),
    "GA3":  (114, 120, 30, 36),
    "CA1":  (108, 114, 44, 50),
    "CA2":  (108, 114, 36, 44),
    "CA3":  (108, 114, 30, 36),
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def check_event_for_bp(ev: dict) -> tuple[bool, str | None, str | None]:
    """
    Extract the first available body part and player name from an event.

    Checks in order: shot body part, pass body part, carry body part,
    then falls back to scanning the event type name for 'head'/'header'.

    Returns:
        (found, player_name, body_part_name)
    """
    for field in ("shot", "pass", "carry"):
        bp = ev.get(field, {}).get("body_part", {}).get("name")
        if bp:
            return True, ev.get("player", {}).get("name"), bp

    tname = ev.get("type", {}).get("name", "")
    if isinstance(tname, str) and ("head" in tname.lower() or "header" in tname.lower()):
        return True, ev.get("player", {}).get("name"), "Head"

    return False, None, None


def get_recipient_body_part(
    event: dict,
    data: list[dict],
    event_index: int,
    id_lookup: dict,
) -> tuple[str | None, str]:
    """
    Determine the body part used by the first recipient of a corner kick.

    Three-stage lookup:
        1. Check related_events (most reliable).
        2. Scan next events in the same possession within BP_SCAN_WINDOW_SEC.
        3. Fallback: first Shot in possession within BP_FALLBACK_WINDOW_SEC.

    Returns:
        (recipient_player_name, body_part_simplified)
        where body_part_simplified is "Head", "Other", or "Unknown".
    """
    time_sec = event.get("minute", 0) * 60 + event.get("second", 0)
    pos_id = event.get("possession")
    recipient = None
    recipient_bp_full = None

    # Stage 1 — related events
    for rid in event.get("related_events", []) or []:
        rel = id_lookup.get(rid)
        if not rel:
            continue
        found, pname, bp = check_event_for_bp(rel)
        if found:
            recipient = pname or recipient
            recipient_bp_full = bp
            break

    # Stage 2 — scan forward in same possession
    if recipient_bp_full is None:
        for nxt in data[event_index + 1: event_index + 1 + BP_SCAN_MAX_EVENTS]:
            if nxt.get("possession") != pos_id:
                break
            nm, ns = nxt.get("minute"), nxt.get("second")
            if nm is None or ns is None:
                continue
            if (nm * 60 + ns) - time_sec > BP_SCAN_WINDOW_SEC:
                break
            found, pname, bp = check_event_for_bp(nxt)
            if found:
                recipient = pname or recipient
                recipient_bp_full = bp
                break

    # Stage 3 — fallback: first shot in possession within window
    if recipient_bp_full is None:
        for nxt in data[event_index + 1:]:
            if nxt.get("possession") != pos_id:
                break
            nm, ns = nxt.get("minute"), nxt.get("second")
            if nm is None or ns is None:
                continue
            if (nm * 60 + ns) - time_sec > BP_FALLBACK_WINDOW_SEC:
                break
            if nxt.get("type", {}).get("name") == "Shot":
                bp = nxt.get("shot", {}).get("body_part", {}).get("name")
                if bp:
                    recipient = nxt.get("player", {}).get("name") or recipient
                    recipient_bp_full = bp
                break

    if recipient_bp_full is None:
        bp_simple = "Unknown"
    elif isinstance(recipient_bp_full, str) and recipient_bp_full.lower() == "head":
        bp_simple = "Head"
    else:
        bp_simple = "Other"

    return recipient, bp_simple


def assign_corner_zone(x: float | None, y: float | None) -> str:
    """
    Assign a named zone to a corner kick delivery coordinate.

    Zones follow the Casal et al. (2019) heuristic for the Irish League study.
    Returns 'Other' if the coordinate falls outside all defined zones.
    """
    if x is None or y is None or (isinstance(x, float) and math.isnan(x)):
        return "Other"
    for zone, (xmin, xmax, ymin, ymax) in ZONE_COORDS.items():
        if xmin < x < xmax and ymin < y < ymax:
            return zone
    return "Other"


def zone_post_type(end_x: float | None, end_y: float | None) -> str:
    """Classify delivery as Near Post, Far Post, or Other."""
    if end_x is None or end_y is None:
        return "Other"
    if end_x > 102 and end_y < 40:
        return "Near Post"
    if end_x > 102 and end_y >= 40:
        return "Far Post"
    return "Other"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_events(events_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse all StatsBomb JSON event files and return corner and goal DataFrames.

    Args:
        events_dir: Path to directory containing per-match JSON event files.

    Returns:
        (df_corners, df_goals)
    """
    corner_rows = []
    goal_rows = []

    for file in events_dir.glob("*.json"):
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)

        match_id = file.stem.split("_")[0]
        id_lookup = {ev.get("id"): ev for ev in data if ev.get("id")}

        # Collect goal timestamps for this match
        goals_in_match = []
        for ev in data:
            if ev.get("type", {}).get("name") == "Shot":
                shot = ev.get("shot", {})
                if shot.get("outcome", {}).get("name") == "Goal":
                    m, s = ev.get("minute"), ev.get("second")
                    if m is not None and s is not None:
                        goals_in_match.append(m * 60 + s)
                        goal_rows.append({
                            "match_id": match_id,
                            "player": ev.get("player", {}).get("name"),
                            "team": ev.get("team", {}).get("name"),
                            "minute": m,
                            "second": s,
                            "time_sec": m * 60 + s,
                        })

        # Process corner kicks
        for i, event in enumerate(data):
            if event.get("pass", {}).get("type", {}).get("name") != "Corner":
                continue

            m, s = event.get("minute"), event.get("second")
            if m is None or s is None:
                continue
            time_sec = m * 60 + s

            player = event.get("player", {}).get("name")
            team = event.get("team", {}).get("name")
            x, y = event.get("location", [None, None])
            end_x, end_y = event.get("pass", {}).get("end_location", [None, None])
            taker_body_part = event.get("pass", {}).get("body_part", {}).get("name")
            angle = event.get("pass", {}).get("angle")

            # Swing direction
            if x is not None and taker_body_part in ("Right Foot", "Left Foot"):
                left_side = x < PITCH_MIDPOINT_X
                swing_type = "inswinging" if (
                    (taker_body_part == "Right Foot" and left_side) or
                    (taker_body_part == "Left Foot" and not left_side)
                ) else "outswinging"
            else:
                swing_type = None

            success = any(0 <= gt - time_sec <= SUCCESS_WINDOW_SEC for gt in goals_in_match)

            _, recipient_bp_simple = get_recipient_body_part(event, data, i, id_lookup)

            corner_rows.append({
                "match_id": match_id,
                "player": player,
                "team": team,
                "minute": m,
                "second": s,
                "time_sec": time_sec,
                "x": x,
                "y": y,
                "end_x": end_x,
                "end_y": end_y,
                "angle": angle,
                "swing_type": swing_type,
                "recipient_body_part": recipient_bp_simple,
                "success": success,
            })

    df_corners = pd.DataFrame(corner_rows).drop_duplicates(subset=["match_id", "time_sec"])
    df_goals = pd.DataFrame(goal_rows).drop_duplicates(subset=["match_id", "time_sec"])

    return df_corners, df_goals


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add zone, zone_post, and success_int columns to the corners DataFrame."""
    df = df.copy()

    # Flip Y for corners taken from the bottom side so all corners face the same goal
    bottom_mask = df["y"] < 40
    df.loc[bottom_mask, "end_y"] = 80 - df.loc[bottom_mask, "end_y"]

    df["zone"] = df.apply(lambda r: assign_corner_zone(r["end_x"], r["end_y"]), axis=1)
    df["zone_post"] = df.apply(lambda r: zone_post_type(r["end_x"], r["end_y"]), axis=1)
    df["recipient_body_part"] = df["recipient_body_part"].apply(
        lambda x: "Head" if x == "Head" else "Other"
    )
    df["success_int"] = df["success"].astype(int)
    return df


def run_statistical_tests(df: pd.DataFrame) -> None:
    """Print chi-squared, z-test, and logistic regression results."""
    print("\n--- Baseline ---")
    print(f"Total corners: {len(df)}")
    print(f"Overall success rate: {df['success'].mean():.4f}")
    print("\nRecipient body part counts:")
    print(df["recipient_body_part"].value_counts(dropna=False))

    print("\n--- Chi-squared: swing type vs success ---")
    table = pd.crosstab(df["swing_type"], df["success"])
    _, p, _, _ = chi2_contingency(table)
    print(f"p-value: {p:.4f}  ({'not significant' if p > 0.05 else 'significant'})")

    print("\n--- Chi-squared: in-box vs out-of-box ---")
    df["in_box"] = (
        (df["end_x"].between(102, 120)) & (df["end_y"].between(18, 62))
    )
    table = pd.crosstab(df["in_box"], df["success"])
    _, p, _, _ = chi2_contingency(table)
    print(f"p-value: {p:.4f}  ({'not significant' if p > 0.05 else 'significant'})")

    print("\n--- Zone success rates ---")
    zone_stats = df.groupby("zone")["success"].agg(["sum", "count"])
    zone_stats["success_rate"] = zone_stats["sum"] / zone_stats["count"]
    print(zone_stats.sort_values("success_rate", ascending=False))

    print("\n--- Z-test: GA3 inswinging vs all others ---")
    ga3_inswing = df[(df["zone"] == "GA3") & (df["swing_type"] == "inswinging")]
    other = df[~((df["zone"] == "GA3") & (df["swing_type"] == "inswinging"))]
    counts = [ga3_inswing["success"].sum(), other["success"].sum()]
    nobs = [len(ga3_inswing), len(other)]
    stat, pval = proportions_ztest(counts, nobs)
    print(f"z={stat:.3f}, p={pval:.4f}")

    print("\n--- Logistic regression: zone + swing_type ---")
    df["zone"] = df["zone"].astype("category")
    df["swing_type"] = df["swing_type"].astype("category")
    model = smf.logit("success_int ~ C(zone) + C(swing_type)", data=df).fit(disp=False)
    print(model.summary())


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------

def _draw_pitch_end(ax, xlim=(100, 120), ylim=(0, 80)):
    """Draw penalty box, six-yard box, and penalty arc on ax."""
    ax.add_patch(Rectangle((0, 0), 120, 80, lw=2, edgecolor="black", facecolor="none"))
    ax.add_patch(Rectangle((102, 18), 18, 44, lw=2, edgecolor="black", facecolor="none"))
    ax.add_patch(Rectangle((114, 30), 6, 20, lw=2, edgecolor="black", facecolor="none"))
    ax.add_patch(Arc((108, 40), 20, 20, theta1=127, theta2=233, lw=2, edgecolor="black"))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("X (length)")
    ax.set_ylabel("Y (width)")


def plot_delivery_scatter(df: pd.DataFrame) -> None:
    """Scatter plot of all corner delivery end-locations, coloured by success."""
    fig, ax = plt.subplots(figsize=(6, 8))
    _draw_pitch_end(ax, xlim=(60, 120))
    colors = df["success"].map({True: "green", False: "red"})
    alphas = df["success"].map({True: 0.6, False: 0.1})
    ax.scatter(df["end_x"], df["end_y"], c=colors, alpha=alphas, s=8)
    ax.set_title("Corner Delivery Locations\n(Green = Goal within 10 s, Red = No Goal)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "delivery_scatter.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.show()
    print("Saved: plots/delivery_scatter.png")


def plot_zone_heatmap(df: pd.DataFrame) -> None:
    """Heat-map of success rate per named zone."""
    zone_stats = df.groupby("zone")["success"].agg(["sum", "count"]).reset_index()
    zone_stats["success_rate"] = zone_stats["sum"] / zone_stats["count"]
    plot_zones = zone_stats[zone_stats["zone"] != "Other"]

    norm = Normalize(vmin=0, vmax=plot_zones["success_rate"].max())
    fig, ax = plt.subplots(figsize=(4, 8))
    _draw_pitch_end(ax)

    for _, row in plot_zones.iterrows():
        zone = row["zone"]
        if zone not in ZONE_COORDS:
            continue
        xmin, xmax, ymin, ymax = ZONE_COORDS[zone]
        color = plt.cm.RdYlGn(norm(row["success_rate"]))
        ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color=color, alpha=0.6))
        ax.text((xmin + xmax) / 2, (ymin + ymax) / 2,
                f"{zone}\n{row['success_rate']:.3f}",
                ha="center", va="center", fontsize=8)

    other = zone_stats[zone_stats["zone"] == "Other"]
    if not other.empty:
        r = other.iloc[0]
        ax.text(0.5, -0.08, f"Other: {int(r['count'])} attempts, {r['success_rate']:.3f} rate",
                ha="center", va="top", transform=ax.transAxes, fontsize=7)

    ax.set_title("Corner Kick Success Rate by Zone")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "zone_heatmap.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.show()
    print("Saved: plots/zone_heatmap.png")


def plot_swing_zone_heatmap(df: pd.DataFrame) -> None:
    """Side-by-side heat-maps of zone success rate split by swing type."""
    zone_swing_stats = (
        df.groupby(["zone", "swing_type"])["success"]
        .agg(["sum", "count"])
        .reset_index()
    )
    zone_swing_stats["success_rate"] = zone_swing_stats["sum"] / zone_swing_stats["count"]

    fig, axes = plt.subplots(1, 2, figsize=(8, 8), sharey=True)
    for ax, swing in zip(axes, ["inswinging", "outswinging"]):
        stats = zone_swing_stats[zone_swing_stats["swing_type"] == swing]
        norm = Normalize(vmin=0, vmax=stats["success_rate"].max())
        _draw_pitch_end(ax, xlim=(95, 120))
        ax.set_title(f"{swing.capitalize()} corners")

        for _, row in stats.iterrows():
            zone = row["zone"]
            if zone == "Other" or zone not in ZONE_COORDS:
                continue
            xmin, xmax, ymin, ymax = ZONE_COORDS[zone]
            color = plt.cm.RdYlGn(norm(row["success_rate"]))
            ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   color=color, alpha=0.6))
            ax.text((xmin + xmax) / 2, (ymin + ymax) / 2,
                    f"{zone}\n{row['success_rate']:.3f}",
                    ha="center", va="center", fontsize=8)

    plt.suptitle("Corner Kick Success Rate by Zone and Swing Type")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "swing_zone_heatmap.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.show()
    print("Saved: plots/swing_zone_heatmap.png")


def plot_polynomial_peak(df: pd.DataFrame) -> None:
    """Polynomial fit to find the peak success coordinate for end_x and end_y."""
    df_clean = df.dropna(subset=["end_x", "end_y"]).copy()

    polys = {}
    for coord, bins, fname in [
        ("end_y", np.arange(0, 81, 0.5),   "poly_end_y.png"),
        ("end_x", np.arange(90, 121, 0.5), "poly_end_x.png"),
    ]:
        df_clean["bin"] = pd.cut(df_clean[coord], bins=bins)
        success_by_bin = df_clean.groupby("bin")["success_int"].mean()
        centers = [iv.left + (iv.right - iv.left) / 2 for iv in success_by_bin.index]
        poly = np.poly1d(np.polyfit(centers, success_by_bin, deg=9))
        polys[coord] = (poly, centers)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(centers, success_by_bin, "o", alpha=0.6, label="Binned success rate")
        ax.plot(centers, poly(centers), "-", color="red", label="Polynomial fit (deg 9)")
        ax.set_xlabel(coord)
        ax.set_ylabel("Success rate")
        ax.set_title(f"Success rate vs {coord}")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / fname, dpi=150, bbox_inches="tight", facecolor="white")
        plt.show()
        print(f"Saved: plots/{fname}")

    # Peak coordinate
    y_poly, y_centers = polys["end_y"]
    x_poly, x_centers = polys["end_x"]
    y_fine = np.linspace(min(y_centers), max(y_centers), 500)
    x_fine = np.linspace(min(x_centers), max(x_centers), 500)
    y_peak = y_fine[np.argmax(y_poly(y_fine))]
    x_peak = x_fine[np.argmax(x_poly(x_fine))]
    print(f"\nEstimated peak delivery coordinate: x={x_peak:.2f}, y={y_peak:.2f}")


def plot_short_vs_zones(df: pd.DataFrame) -> None:
    """Bar chart comparing short-pass success rate against other delivery zones."""
    zone_stats = df.groupby("zone")["success"].agg(["sum", "count"]).reset_index()
    zone_stats["success_rate"] = zone_stats["sum"] / zone_stats["count"]

    short = df[df["end_x"].between(105, 120) & df["end_y"].between(65, 80)]
    short_rate = short["success"].mean() if len(short) > 0 else 0
    short_row = pd.DataFrame([{
        "zone": "Short Delivery",
        "sum": short["success"].sum(),
        "count": len(short),
        "success_rate": short_rate,
    }])
    combined = pd.concat([zone_stats, short_row], ignore_index=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(combined["zone"], combined["success_rate"])
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.001,
                f"{h:.2%}", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("Delivery Zone")
    ax.set_ylabel("Success Rate")
    ax.set_title("Success Rate: Short Passes vs Other Delivery Zones")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "short_vs_zones.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.show()
    print("Saved: plots/short_vs_zones.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    # Create plots directory if it doesn't exist
    PLOTS_DIR.mkdir(exist_ok=True)

    print("Loading events...")
    df_corners, df_goals = load_events(EVENTS_DIR)

    df_corners.to_csv(CORNERS_OUTPUT, index=False)
    df_goals.to_csv(GOALS_OUTPUT, index=False)
    print(f"Saved {len(df_corners)} corners → {CORNERS_OUTPUT}")
    print(f"Saved {len(df_goals)} goals → {GOALS_OUTPUT}")

    df_corners = add_derived_columns(df_corners)

    run_statistical_tests(df_corners)

    print("\nGenerating plots...")
    plot_delivery_scatter(df_corners)
    plot_zone_heatmap(df_corners)
    plot_swing_zone_heatmap(df_corners)
    plot_polynomial_peak(df_corners)
    plot_short_vs_zones(df_corners)

    print(f"\nAll plots saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()