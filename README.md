# Corner Kick Analysis — StatsBomb Open Data

A data analysis project identifying the **optimal delivery zones** for corner kicks in football, using StatsBomb open event data.

## Key Findings

- Corners delivered to **zone GA3** (back post, inside the six-yard box) yield the highest success rate (~5.7–7%), statistically significant at p < 0.05.
- **Swing type alone** (inswinging vs outswinging) has no significant effect on success rate (chi-squared p >> 0.05).
- **In-box deliveries** significantly outperform out-of-box deliveries (chi-squared p < 0.05).
- **Short deliveries** are consistently subpar compared to crosses into the box.
- Polynomial fitting estimates the single optimal delivery point at approximately **(x=116, y=33)** on a StatsBomb 120×80 pitch.

Zone classification follows the heuristic from:
> Casal et al. (2019) — *Performance indicators of corner kicks in elite football*
> https://www.tandfonline.com/doi/full/10.1080/24748668.2019.1677329

---

## Zone Map

Zones are defined relative to the attacking goal end of a 120×80 StatsBomb pitch:

| Zone | Description              | X range   | Y range |
|------|--------------------------|-----------|---------|
| GA1  | Six-yard box near post   | 114 – 120 | 44 – 50 |
| GA2  | Six-yard box centre      | 114 – 120 | 36 – 44 |
| GA3  | Six-yard box far post    | 114 – 120 | 30 – 36 |
| CA1  | Central area near post   | 108 – 114 | 44 – 50 |
| CA2  | Central area centre      | 108 – 114 | 36 – 44 |
| CA3  | Central area far post    | 108 – 114 | 30 – 36 |
| FZ   | Front zone               | 102 – 120 | 50 – 62 |
| BZ   | Back zone                | 102 – 120 | 18 – 30 |
| Edge | Edge of area             | 100 – 108 | 30 – 50 |

---

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/your-username/corner-kicks-analysis.git
cd corner-kicks-analysis
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the data

Get StatsBomb open event data from their official repository:

```bash
git clone https://github.com/statsbomb/open-data.git
cp -r open-data/data/events ./events
```

Or download just the competitions you want and place the JSON files in an `events/` directory.

### 4. Run the analysis

```bash
python corner_kicks.py
```

This will:
- Parse all event files in `events/`
- Output `corner_deliveries.csv` and `goals.csv`
- Output statistical test results
- Display a series of visualization plots

---

## Project Structure

```
corner-kicks-analysis/
├── corner_kicks.py      # Main analysis script
├── requirements.txt     # Python dependencies
└── README.md
```

---

## Methods

| Method | Purpose |
|--------|---------|
| Chi-squared test | Test independence of swing type / delivery zone vs success |
| Two-sample z-test | Compare GA3 inswinging success rate vs all others |
| Logistic regression | Model success probability across zone + swing type combinations |
| Polynomial fitting | Estimate the peak success delivery coordinate (end_x, end_y) |

---

## Data

Uses **StatsBomb open data** (free, publicly available):
https://github.com/statsbomb/open-data

StatsBomb pitch coordinates: 120 (length) × 80 (width).
