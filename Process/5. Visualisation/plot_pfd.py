"""
Process Flow Diagram (PFD) for Glacial Acetic Acid Separation
=============================================================
Liquid-liquid extraction with ethyl acetate solvent,
followed by distillation train for acid purification and solvent recovery.

Generates a publication-quality PFD using matplotlib.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Arc, Polygon
import numpy as np

# ── Style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 8,
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.0,
})

BLUE = '#2B6CB0'
ORANGE = '#DD6B20'
GREEN = '#2F855A'
GRAY = '#4A5568'
LIGHT_BLUE = '#BEE3F8'
LIGHT_ORANGE = '#FEEBC8'
LIGHT_GREEN = '#C6F6D5'
LIGHT_GRAY = '#E2E8F0'
LIGHT_PURPLE = '#E9D8FD'
STREAM_COLOR = '#2D3748'
RECYCLE_COLOR = '#9B2C2C'


# ── Helper drawing functions ───────────────────────────────────────────

def draw_column(ax, cx, cy, w=0.6, h=1.8, label='', tag='', color=LIGHT_BLUE, edge=BLUE):
    """Draw a distillation column (tall rounded rectangle with internals)."""
    rect = FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle=f"round,pad=0.08",
        facecolor=color, edgecolor=edge, linewidth=1.5, zorder=3
    )
    ax.add_patch(rect)
    # Draw tray lines inside
    n_trays = 5
    for i in range(1, n_trays + 1):
        y_tray = cy - h/2 + i * h / (n_trays + 1)
        ax.plot([cx - w/2 + 0.06, cx + w/2 - 0.06], [y_tray, y_tray],
                color=edge, linewidth=0.5, alpha=0.5, zorder=4)
    # Labels
    ax.text(cx, cy, label, ha='center', va='center', fontsize=7,
            fontweight='bold', color=edge, zorder=5)
    ax.text(cx, cy - h/2 - 0.15, tag, ha='center', va='top', fontsize=6.5,
            color=GRAY, zorder=5)
    return cx, cy, w, h


def draw_hx(ax, cx, cy, r=0.3, label='', tag='', color=LIGHT_ORANGE, edge=ORANGE):
    """Draw a heat exchanger (circle with internal sine wave)."""
    circle = plt.Circle((cx, cy), r, facecolor=color, edgecolor=edge,
                         linewidth=1.5, zorder=3)
    ax.add_patch(circle)
    # Sine wave inside
    t = np.linspace(-0.7, 0.7, 40)
    xs = cx + t * r * 0.8
    ys = cy + np.sin(t * 6) * r * 0.25
    ax.plot(xs, ys, color=edge, linewidth=1.0, zorder=4)
    ax.text(cx, cy - r - 0.13, tag, ha='center', va='top', fontsize=6.5,
            color=GRAY, zorder=5)
    if label:
        ax.text(cx, cy + r + 0.1, label, ha='center', va='bottom', fontsize=6.5,
                color=edge, fontweight='bold', zorder=5)
    return cx, cy, r


def draw_extractor(ax, cx, cy, w=1.0, h=0.8, label='', tag='', color=LIGHT_GREEN, edge=GREEN):
    """Draw an extractor (wide rounded rectangle with wave pattern)."""
    rect = FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0.08",
        facecolor=color, edgecolor=edge, linewidth=1.5, zorder=3
    )
    ax.add_patch(rect)
    # Wavy interface line (liquid-liquid)
    xs = np.linspace(cx - w/2 + 0.1, cx + w/2 - 0.1, 30)
    ys = cy + 0.05 * np.sin(xs * 15)
    ax.plot(xs, ys, color=edge, linewidth=1.0, alpha=0.6, zorder=4)
    ax.text(cx, cy, label, ha='center', va='center', fontsize=7,
            fontweight='bold', color=edge, zorder=5)
    ax.text(cx, cy - h/2 - 0.15, tag, ha='center', va='top', fontsize=6.5,
            color=GRAY, zorder=5)
    return cx, cy, w, h


def draw_mixer(ax, cx, cy, r=0.22, label='', tag='', color=LIGHT_GRAY, edge=GRAY):
    """Draw a mixer (circle with M)."""
    circle = plt.Circle((cx, cy), r, facecolor=color, edgecolor=edge,
                         linewidth=1.5, zorder=3)
    ax.add_patch(circle)
    ax.text(cx, cy, 'M', ha='center', va='center', fontsize=8,
            fontweight='bold', color=edge, zorder=5)
    if tag:
        ax.text(cx, cy - r - 0.13, tag, ha='center', va='top', fontsize=6.5,
                color=GRAY, zorder=5)
    return cx, cy, r


def draw_settler(ax, cx, cy, w=0.8, h=0.6, label='', tag='', color='#FEFCBF', edge='#B7791F'):
    """Draw a settler/decanter (horizontal vessel with phase interface)."""
    rect = FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0.06",
        facecolor=color, edgecolor=edge, linewidth=1.5, zorder=3
    )
    ax.add_patch(rect)
    # Phase interface (dashed)
    ax.plot([cx - w/2 + 0.08, cx + w/2 - 0.08], [cy, cy],
            color=edge, linewidth=1.0, linestyle='--', alpha=0.7, zorder=4)
    ax.text(cx, cy + 0.10, 'org.', ha='center', va='center', fontsize=5.5,
            color=edge, style='italic', zorder=5)
    ax.text(cx, cy - 0.10, 'aq.', ha='center', va='center', fontsize=5.5,
            color=edge, style='italic', zorder=5)
    ax.text(cx, cy + h/2 + 0.10, label, ha='center', va='bottom', fontsize=7,
            fontweight='bold', color=edge, zorder=5)
    ax.text(cx, cy - h/2 - 0.13, tag, ha='center', va='top', fontsize=6.5,
            color=GRAY, zorder=5)
    return cx, cy, w, h


def draw_splitter(ax, cx, cy, r=0.18, tag=''):
    """Draw a splitter (small triangle)."""
    tri = Polygon(
        [(cx, cy + r), (cx - r, cy - r), (cx + r, cy - r)],
        closed=True, facecolor=LIGHT_PURPLE, edgecolor='#6B46C1',
        linewidth=1.2, zorder=3
    )
    ax.add_patch(tri)
    ax.text(cx, cy - 0.02, 'S', ha='center', va='center', fontsize=6.5,
            fontweight='bold', color='#6B46C1', zorder=5)
    if tag:
        ax.text(cx, cy - r - 0.13, tag, ha='center', va='top', fontsize=6.5,
                color=GRAY, zorder=5)
    return cx, cy, r


def arrow(ax, x1, y1, x2, y2, color=STREAM_COLOR, style='->', lw=1.2,
          shrinkA=0, shrinkB=0, connectionstyle='arc3,rad=0'):
    """Draw a stream arrow."""
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style,
        connectionstyle=connectionstyle,
        color=color, linewidth=lw,
        mutation_scale=10,
        shrinkA=shrinkA, shrinkB=shrinkB,
        zorder=2
    )
    ax.add_patch(arr)
    return arr


def stream_label(ax, x, y, text, color=STREAM_COLOR, fontsize=6, ha='center',
                 va='bottom', offset=(0, 0.08), rotation=0, bbox=True):
    """Add a stream label."""
    kwargs = dict(ha=ha, va=va, fontsize=fontsize, color=color, zorder=6, rotation=rotation)
    if bbox:
        kwargs['bbox'] = dict(boxstyle='round,pad=0.15', facecolor='white',
                              edgecolor='none', alpha=0.85)
    ax.text(x + offset[0], y + offset[1], text, **kwargs)


# ── Main PFD ───────────────────────────────────────────────────────────

def create_pfd(save_path='pfd_acetic_acid.pdf'):
    fig, ax = plt.subplots(1, 1, figsize=(16, 9.5))
    ax.set_xlim(-1.5, 16.5)
    ax.set_ylim(-3.0, 8.0)
    ax.set_aspect('equal')
    ax.axis('off')

    # ── Equipment positions ────────────────────────────────────────
    # Row 1 (top): main separation train
    ext_x, ext_y = 1.5, 4.0     # Extractor
    hx1_x, hx1_y = 4.0, 4.0     # Extract heater
    ed_x, ed_y   = 6.5, 4.0      # Extract distiller
    ed2_x, ed2_y = 9.5, 4.0      # Acid purification

    # Row 2 (middle): recycle loop
    m1_x, m1_y     = 8.0, 1.5    # Mixer 1 (overheads)
    hx2_x, hx2_y   = 6.0, 1.5    # Cooling HX
    set_x, set_y   = 4.0, 1.5    # Settler
    spl_x, spl_y   = 2.5, 1.5    # Splitter

    # Row 3 (bottom-right): raffinate path
    m2_x, m2_y     = 3.0, -0.8   # Mixer 2 (raffinate + water-rich)
    rd_x, rd_y     = 5.5, -0.8   # Raffinate distiller

    # ── Draw equipment ─────────────────────────────────────────────
    draw_extractor(ax, ext_x, ext_y, w=1.2, h=0.9,
                   label='Extractor', tag='E-101')

    draw_hx(ax, hx1_x, hx1_y, r=0.3,
            label='Heater', tag='H-101')

    draw_column(ax, ed_x, ed_y, w=0.65, h=2.0,
                label='ED', tag='T-101\n(Extract Distiller)')

    draw_column(ax, ed2_x, ed2_y, w=0.65, h=2.0,
                label='ED2', tag='T-102\n(Acid Purification)')

    draw_mixer(ax, m1_x, m1_y, r=0.22, tag='M-101')
    draw_hx(ax, hx2_x, hx2_y, r=0.3, label='Cooler', tag='H-102')
    draw_settler(ax, set_x, set_y, w=0.9, h=0.6,
                 label='Settler', tag='V-101')
    draw_splitter(ax, spl_x, spl_y, tag='S-101')

    draw_mixer(ax, m2_x, m2_y, r=0.22, tag='M-102')
    draw_column(ax, rd_x, rd_y, w=0.65, h=1.6,
                label='RD', tag='T-103\n(Raffinate Distiller)')

    # ── Feed streams ───────────────────────────────────────────────
    # Acetic acid broth feed
    arrow(ax, -0.8, 4.3, ext_x - 0.6, 4.3)
    stream_label(ax, -0.3, 4.3, 'Acetic Acid Broth\n(1,000 kg/h AA\n+ 9,000 kg/h H₂O)',
                 fontsize=5.5, ha='center', va='bottom', offset=(0, 0.1))

    # Fresh ethyl acetate feed
    arrow(ax, -0.8, 3.7, ext_x - 0.6, 3.7)
    stream_label(ax, -0.3, 3.7, 'Fresh Ethyl Acetate',
                 fontsize=5.5, ha='center', va='top', offset=(0, -0.12))

    # ── Extractor → HX1 (extract) ─────────────────────────────────
    arrow(ax, ext_x + 0.6, 4.0, hx1_x - 0.3, 4.0)
    stream_label(ax, 2.95, 4.0, 'Extract', fontsize=5.5)

    # ── HX1 → ED ──────────────────────────────────────────────────
    arrow(ax, hx1_x + 0.3, 4.0, ed_x - 0.325, 4.0)
    stream_label(ax, 5.25, 4.0, 'Hot Extract', fontsize=5.5)

    # ── ED bottoms → ED2 ──────────────────────────────────────────
    arrow(ax, ed_x + 0.325, 3.5, ed2_x - 0.325, 3.5)
    stream_label(ax, 8.0, 3.5, 'AA + EtAc', fontsize=5.5,
                 va='top', offset=(0, -0.15))

    # ── ED2 bottoms → Product ─────────────────────────────────────
    # Route down from ED2 bottom, then right to product box
    ed2_bot_x = ed2_x + 0.325
    ed2_bot_y = 3.0
    ax.plot([ed2_bot_x, ed2_bot_x], [ed2_bot_y, 2.2], color=STREAM_COLOR, linewidth=1.2, zorder=2)
    arrow(ax, ed2_bot_x, 2.2, 11.3, 2.2)
    # Product box
    product_box = FancyBboxPatch(
        (11.3, 1.8), 3.0, 0.8,
        boxstyle="round,pad=0.1",
        facecolor='#FED7D7', edgecolor='#C53030', linewidth=1.5, zorder=3
    )
    ax.add_patch(product_box)
    ax.text(12.8, 2.2, 'Glacial Acetic Acid\n(Product, ≥98 wt%)', ha='center',
            va='center', fontsize=7.5, fontweight='bold', color='#C53030', zorder=5)

    # ── ED overhead → Mixer 1 ─────────────────────────────────────
    # ED top goes down-right to M1
    ed_top_x = ed_x + 0.325
    ed_top_y = 4.7
    arrow(ax, ed_top_x, ed_top_y, ed_top_x + 0.3, ed_top_y)
    ax.plot([ed_top_x + 0.3, ed_top_x + 0.3], [ed_top_y, ed_top_y + 0.4],
            color=STREAM_COLOR, linewidth=1.2, zorder=2)
    ax.plot([ed_top_x + 0.3, m1_x + 0.6], [ed_top_y + 0.4, ed_top_y + 0.4],
            color=STREAM_COLOR, linewidth=1.2, zorder=2)
    ax.plot([m1_x + 0.6, m1_x + 0.6], [ed_top_y + 0.4, m1_y + 0.5],
            color=STREAM_COLOR, linewidth=1.2, zorder=2)
    arrow(ax, m1_x + 0.6, m1_y + 0.5, m1_x + 0.22, m1_y + 0.1, shrinkB=2)
    stream_label(ax, ed_top_x + 0.9, ed_top_y + 0.4, 'ED Overhead\n(H₂O + EtAc)',
                 fontsize=5.5, va='bottom', offset=(0, 0.05))

    # ── ED2 overhead → Mixer 1 ────────────────────────────────────
    ed2_top_x = ed2_x
    ed2_top_y = ed2_y + 1.0
    ax.plot([ed2_top_x, ed2_top_x], [ed2_top_y, ed2_top_y + 0.5],
            color=STREAM_COLOR, linewidth=1.2, zorder=2)
    ax.plot([ed2_top_x, m1_x + 0.4], [ed2_top_y + 0.5, ed2_top_y + 0.5],
            color=STREAM_COLOR, linewidth=1.2, zorder=2)
    ax.plot([m1_x + 0.4, m1_x + 0.4], [ed2_top_y + 0.5, m1_y + 0.6],
            color=STREAM_COLOR, linewidth=1.2, zorder=2)
    arrow(ax, m1_x + 0.4, m1_y + 0.6, m1_x + 0.12, m1_y + 0.18, shrinkB=2)
    stream_label(ax, ed2_top_x - 0.3, ed2_top_y + 0.55, 'ED2 Overhead\n(EtAc)',
                 fontsize=5.5, va='bottom', offset=(0, 0.05))

    # ── Mixer 1 → HX2 ────────────────────────────────────────────
    arrow(ax, m1_x - 0.22, m1_y, hx2_x + 0.3, hx2_y)

    # ── HX2 → Settler ────────────────────────────────────────────
    arrow(ax, hx2_x - 0.3, hx2_y, set_x + 0.45, set_y)
    stream_label(ax, 5.0, 1.5, 'Cooled', fontsize=5.5, va='top', offset=(0, -0.35))

    # ── Settler → Splitter (top product: organic/solvent-rich) ────
    arrow(ax, set_x - 0.45, set_y + 0.15, spl_x + 0.18, spl_y + 0.12)
    stream_label(ax, 3.15, 1.75, 'Organic\n(EtAc-rich)',
                 fontsize=5.5, va='bottom', offset=(0, 0.05))

    # ── Splitter → Solvent recycle (back to Extractor) ────────────
    spl_out_x = spl_x
    spl_out_y = spl_y + 0.18
    ax.plot([spl_out_x, spl_out_x], [spl_out_y, ext_y - 0.45 - 0.5],
            color=RECYCLE_COLOR, linewidth=1.5, linestyle='--', zorder=2)
    ax.plot([spl_out_x, ext_x + 0.3], [ext_y - 0.45 - 0.5, ext_y - 0.45 - 0.5],
            color=RECYCLE_COLOR, linewidth=1.5, linestyle='--', zorder=2)
    arrow(ax, ext_x + 0.3, ext_y - 0.45 - 0.5, ext_x + 0.3, ext_y - 0.45,
          color=RECYCLE_COLOR, lw=1.5)
    stream_label(ax, spl_out_x + 0.1, 2.6, 'Solvent\nRecycle',
                 fontsize=5.5, color=RECYCLE_COLOR, va='center', ha='left')

    # ── Splitter → Reflux to ED (dashed, only in rigorous model) ──
    spl_ref_x = spl_x - 0.18
    spl_ref_y = spl_y - 0.18 + 0.05
    ax.plot([spl_ref_x, spl_ref_x - 0.3], [spl_ref_y, spl_ref_y],
            color='#6B46C1', linewidth=1.0, linestyle=':', zorder=2)
    ax.plot([spl_ref_x - 0.3, spl_ref_x - 0.3], [spl_ref_y, 5.5],
            color='#6B46C1', linewidth=1.0, linestyle=':', zorder=2)
    ax.plot([spl_ref_x - 0.3, ed_x - 0.5], [5.5, 5.5],
            color='#6B46C1', linewidth=1.0, linestyle=':', zorder=2)
    arrow(ax, ed_x - 0.5, 5.5, ed_x - 0.5, ed_y + 1.0,
          color='#6B46C1', lw=1.0, shrinkB=2)
    stream_label(ax, 3.5, 5.55, 'Reflux*', fontsize=5, color='#6B46C1',
                 va='bottom', offset=(0, 0.02))

    # ── Settler bottom → Mixer 2 (aqueous/water-rich) ─────────────
    set_bot_x = set_x
    set_bot_y = set_y - 0.3
    ax.plot([set_bot_x, set_bot_x], [set_bot_y, m2_y + 0.5],
            color=STREAM_COLOR, linewidth=1.2, zorder=2)
    arrow(ax, set_bot_x, m2_y + 0.5, m2_x + 0.15, m2_y + 0.22, shrinkB=2)
    stream_label(ax, set_bot_x + 0.1, 0.3, 'Aqueous\n(H₂O-rich)',
                 fontsize=5.5, ha='left', va='center')

    # ── Extractor raffinate → Mixer 2 ─────────────────────────────
    raf_x = ext_x
    raf_y = ext_y - 0.45
    ax.plot([raf_x, raf_x], [raf_y, m2_y],
            color=STREAM_COLOR, linewidth=1.2, zorder=2)
    arrow(ax, raf_x, m2_y, m2_x - 0.22, m2_y)
    stream_label(ax, raf_x + 0.1, 1.5, 'Raffinate',
                 fontsize=5.5, ha='left', va='center')

    # ── Mixer 2 → RD ─────────────────────────────────────────────
    arrow(ax, m2_x + 0.22, m2_y, rd_x - 0.325, m2_y)

    # ── RD overhead → Mixer 1 (distillate recycle) ────────────────
    rd_top_x = rd_x
    rd_top_y = rd_y + 0.8
    ax.plot([rd_top_x, rd_top_x], [rd_top_y, rd_top_y + 0.6],
            color=RECYCLE_COLOR, linewidth=1.3, linestyle='--', zorder=2)
    ax.plot([rd_top_x, m1_x + 0.8], [rd_top_y + 0.6, rd_top_y + 0.6],
            color=RECYCLE_COLOR, linewidth=1.3, linestyle='--', zorder=2)
    ax.plot([m1_x + 0.8, m1_x + 0.8], [rd_top_y + 0.6, m1_y],
            color=RECYCLE_COLOR, linewidth=1.3, linestyle='--', zorder=2)
    arrow(ax, m1_x + 0.8, m1_y, m1_x + 0.22, m1_y,
          color=RECYCLE_COLOR, lw=1.3)
    stream_label(ax, 7.2, rd_top_y + 0.65, 'RD Distillate\n(EtAc recycle)',
                 fontsize=5.5, color=RECYCLE_COLOR, va='bottom', offset=(0, 0.05))

    # ── RD bottoms → Wastewater ───────────────────────────────────
    rd_bot_y = rd_y - 0.5
    arrow(ax, rd_x + 0.325, rd_bot_y, 8.0, rd_bot_y)
    waste_box = FancyBboxPatch(
        (8.0, rd_bot_y - 0.35), 2.2, 0.7,
        boxstyle="round,pad=0.1",
        facecolor=LIGHT_GRAY, edgecolor=GRAY, linewidth=1.2, zorder=3
    )
    ax.add_patch(waste_box)
    ax.text(9.1, rd_bot_y, 'Wastewater', ha='center', va='center',
            fontsize=7, fontweight='bold', color=GRAY, zorder=5)

    # ── Legend ─────────────────────────────────────────────────────
    legend_x, legend_y = 12.0, 7.0
    ax.text(legend_x, legend_y, 'Legend', fontsize=8, fontweight='bold', color=GRAY)

    legend_items = [
        ('─── ', STREAM_COLOR, 'Process stream'),
        ('- - - ', RECYCLE_COLOR, 'Recycle stream'),
        ('····· ', '#6B46C1', 'Reflux (rigorous only)'),
    ]
    for i, (sym, col, desc) in enumerate(legend_items):
        y = legend_y - 0.35 * (i + 1)
        ax.text(legend_x, y, sym, fontsize=7, color=col, fontfamily='monospace')
        ax.text(legend_x + 0.65, y, desc, fontsize=6.5, color=GRAY)

    # Equipment legend
    eq_y = legend_y - 0.35 * (len(legend_items) + 1.5)
    ax.text(legend_x, eq_y, 'Equipment', fontsize=8, fontweight='bold', color=GRAY)
    equip_items = [
        (LIGHT_GREEN, GREEN, 'E-101: Liquid-Liquid Extractor'),
        (LIGHT_ORANGE, ORANGE, 'H-101/102: Heat Exchangers'),
        (LIGHT_BLUE, BLUE, 'T-101/102/103: Distillation Columns'),
        ('#FEFCBF', '#B7791F', 'V-101: Settler (L-L Separator)'),
        (LIGHT_PURPLE, '#6B46C1', 'S-101: Stream Splitter'),
    ]
    for i, (fc, ec, desc) in enumerate(equip_items):
        y = eq_y - 0.35 * (i + 1)
        rect = FancyBboxPatch(
            (legend_x, y - 0.1), 0.3, 0.2,
            boxstyle="round,pad=0.02",
            facecolor=fc, edgecolor=ec, linewidth=1.0, zorder=3
        )
        ax.add_patch(rect)
        ax.text(legend_x + 0.45, y, desc, fontsize=6, color=GRAY, va='center')

    # ── Title ──────────────────────────────────────────────────────
    ax.text(5.5, 7.6, 'Process Flow Diagram: Glacial Acetic Acid Separation',
            ha='center', va='center', fontsize=13, fontweight='bold', color='#1A202C')
    ax.text(5.5, 7.2,
            'Liquid-liquid extraction with ethyl acetate, followed by distillation for purification and solvent recovery',
            ha='center', va='center', fontsize=8, color=GRAY, style='italic')

    # ── Operating conditions annotation ────────────────────────────
    ann_x, ann_y = -1.0, -2.2
    conditions = (
        'Operating Conditions:\n'
        '• Feed: 10,000 kg/h (10 wt% acetic acid)\n'
        '• Solvent: ethyl acetate (1.5:1 S/F ratio)\n'
        '• Product purity target: ≥98 wt% AA\n'
        '• Annual operation: 330 days/yr (7,920 h/yr)'
    )
    ax.text(ann_x, ann_y, conditions, fontsize=6, color=GRAY, va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#F7FAFC',
                      edgecolor='#CBD5E0', linewidth=0.8))

    plt.tight_layout()

    # Save multiple formats
    for ext in ['pdf', 'png', 'svg']:
        out = save_path.rsplit('.', 1)[0] + '.' + ext
        dpi = 300 if ext == 'png' else None
        fig.savefig(out, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f'Saved: {out}')

    plt.close(fig)


if __name__ == '__main__':
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, 'pfd_acetic_acid.pdf')
    create_pfd(save_path)
