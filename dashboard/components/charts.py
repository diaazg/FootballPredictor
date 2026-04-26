"""
Reusable chart components used across dashboard pages.
All return Plotly figures — displayed with st.plotly_chart().
"""

import math
import plotly.graph_objects as go
import plotly.express as px


# ── Colour palette ────────────────────────────────────────────────────────────
BLUE   = "#1454a0"
GREEN  = "#2ecc71"
AMBER  = "#f39c12"
RED    = "#e74c3c"
GREY   = "#95a5a6"
WHITE  = "#ffffff"
BG     = "#0e1117"
CARD   = "#1a1f2e"


# ── Match outcome probability bar ─────────────────────────────────────────────

def match_probability_chart(prob_home: float, prob_draw: float, prob_away: float,
                             home_label: str = "Home", away_label: str = "Away") -> go.Figure:
    labels = [f"Home Win\n({home_label})", "Draw", f"Away Win\n({away_label})"]
    values = [prob_home, prob_draw, prob_away]
    colours = [BLUE, GREY, RED]

    fig = go.Figure(go.Bar(
        x=labels,
        y=[v * 100 for v in values],
        marker_color=colours,
        text=[f"{v*100:.1f}%" for v in values],
        textposition="outside",
        textfont=dict(size=16, color=WHITE),
    ))
    fig.update_layout(
        title="Match Outcome Probabilities",
        yaxis_title="Probability (%)",
        yaxis_range=[0, 100],
        plot_bgcolor=CARD,
        paper_bgcolor=BG,
        font=dict(color=WHITE),
        showlegend=False,
        height=380,
    )
    return fig


# ── xG pitch visualisation ────────────────────────────────────────────────────

def xg_pitch(shot_x: float, shot_y: float, xg: float) -> go.Figure:
    """
    Draw a top-down half-pitch with the shot location.
    shot_x, shot_y are on the 0-100 percentage scale.
    """
    fig = go.Figure()

    # Pitch background
    fig.add_shape(type="rect", x0=50, y0=0, x1=100, y1=100,
                  fillcolor="#2d7a2d", line=dict(color=WHITE, width=2))
    # Centre line
    fig.add_shape(type="line", x0=50, y0=0, x1=50, y1=100,
                  line=dict(color=WHITE, width=2))
    # Penalty box (big box)
    fig.add_shape(type="rect", x0=83, y0=21.1, x1=100, y1=78.9,
                  fillcolor="rgba(0,0,0,0)", line=dict(color=WHITE, width=1.5))
    # Six-yard box
    fig.add_shape(type="rect", x0=94.2, y0=36.8, x1=100, y1=63.2,
                  fillcolor="rgba(0,0,0,0)", line=dict(color=WHITE, width=1.5))
    # Goal
    fig.add_shape(type="rect", x0=99.5, y0=44.8, x1=100.5, y1=55.2,
                  fillcolor=GREY, line=dict(color=WHITE, width=2))
    # Penalty spot
    fig.add_shape(type="circle", x0=87.7, y0=49.2, x1=88.3, y1=50.8,
                  fillcolor=WHITE, line=dict(color=WHITE))
    # Centre circle (half)
    theta = [t * math.pi / 180 for t in range(270, 451)]
    cx = [50 + 9.15 * math.cos(t) for t in theta]
    cy = [50 + 9.15 * math.sin(t) for t in theta]
    fig.add_trace(go.Scatter(x=cx, y=cy, mode="lines",
                             line=dict(color=WHITE, width=1.5), showlegend=False))

    # Shot location
    colour = GREEN if xg >= 0.5 else (AMBER if xg >= 0.2 else RED)
    size   = 16 + int(xg * 20)
    fig.add_trace(go.Scatter(
        x=[shot_x], y=[shot_y],
        mode="markers+text",
        marker=dict(size=size, color=colour, symbol="circle",
                    line=dict(color=WHITE, width=2)),
        text=[f"xG={xg:.3f}"],
        textposition="top center",
        textfont=dict(size=13, color=WHITE),
        showlegend=False,
    ))

    fig.update_layout(
        title="Shot Location",
        xaxis=dict(range=[45, 105], showgrid=False, zeroline=False,
                   showticklabels=False, scaleanchor="y"),
        yaxis=dict(range=[-5, 105], showgrid=False, zeroline=False,
                   showticklabels=False),
        plot_bgcolor="#2d7a2d",
        paper_bgcolor=BG,
        font=dict(color=WHITE),
        height=420,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


# ── Injury risk gauge ─────────────────────────────────────────────────────────

def injury_gauge(prob_high: float) -> go.Figure:
    colour = RED if prob_high >= 0.6 else (AMBER if prob_high >= 0.4 else GREEN)
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(prob_high * 100, 1),
        number=dict(suffix="%", font=dict(size=36, color=WHITE)),
        title=dict(text="High Injury Risk Probability", font=dict(size=15, color=WHITE)),
        delta=dict(reference=50, increasing=dict(color=RED), decreasing=dict(color=GREEN)),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor=WHITE, tickfont=dict(color=WHITE)),
            bar=dict(color=colour),
            bgcolor=CARD,
            steps=[
                dict(range=[0,  40], color="#1a3a1a"),
                dict(range=[40, 60], color="#3a3a1a"),
                dict(range=[60, 100], color="#3a1a1a"),
            ],
            threshold=dict(
                line=dict(color=WHITE, width=3),
                thickness=0.8,
                value=50,
            ),
        ),
    ))
    fig.update_layout(
        paper_bgcolor=BG,
        font=dict(color=WHITE),
        height=300,
        margin=dict(l=30, r=30, t=60, b=20),
    )
    return fig


# ── SHAP horizontal bar chart ─────────────────────────────────────────────────

def shap_bar_chart(top_features: list, title: str = "Feature Impact (SHAP)") -> go.Figure:
    features = [f["feature"] for f in reversed(top_features)]
    values   = [f["shap_value"] for f in reversed(top_features)]
    colours  = [GREEN if v > 0 else RED for v in values]
    hover    = [
        f"Feature: {f['feature']}<br>Value: {f['value']}<br>SHAP: {f['shap_value']:+.4f}"
        for f in reversed(top_features)
    ]

    fig = go.Figure(go.Bar(
        x=values,
        y=features,
        orientation="h",
        marker_color=colours,
        hovertext=hover,
        hoverinfo="text",
    ))
    fig.add_vline(x=0, line=dict(color=WHITE, width=1.5))
    fig.update_layout(
        title=title,
        xaxis_title="SHAP value (+ increases prediction, - decreases)",
        plot_bgcolor=CARD,
        paper_bgcolor=BG,
        font=dict(color=WHITE),
        height=max(300, len(top_features) * 38),
        margin=dict(l=10, r=10, t=50, b=40),
    )
    return fig


# ── Model comparison table ────────────────────────────────────────────────────

def model_compare_table(variants: list, metric_col: str = "auc") -> go.Figure:
    names     = [v["name"] for v in variants]
    metric    = [v.get(metric_col) for v in variants]
    accuracy  = [v.get("accuracy") for v in variants]
    notes     = [v["note"] for v in variants]

    def fmt(val):
        return f"{val:.4f}" if val is not None else "—"

    fig = go.Figure(go.Table(
        header=dict(
            values=["Model", "AUC-ROC", "Accuracy", "Note"],
            fill_color=BLUE,
            font=dict(color=WHITE, size=12),
            align="left",
            height=30,
        ),
        cells=dict(
            values=[
                names,
                [fmt(m) for m in metric],
                [fmt(a) for a in accuracy],
                notes,
            ],
            fill_color=[[CARD if i % 2 == 0 else "#232840" for i in range(len(names))]],
            font=dict(color=WHITE, size=11),
            align="left",
            height=26,
        ),
    ))
    fig.update_layout(
        paper_bgcolor=BG,
        font=dict(color=WHITE),
        height=80 + len(variants) * 32,
        margin=dict(l=0, r=0, t=0, b=0),
    )
    return fig
