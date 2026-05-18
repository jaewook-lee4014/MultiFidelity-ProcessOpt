"""
Generate Process Flow Diagram using BioSTEAM's built-in diagram().

BioSTEAM uses Graphviz under the hood to render the flowsheet
directly from the simulation objects.
"""
import os
import sys

# Ensure the graphviz dot binary is found
env_bin = os.path.dirname(sys.executable)
os.environ['PATH'] = env_bin + ':' + os.environ.get('PATH', '')

import biosteam as bst

bst.nbtutorial()

# ── Define chemicals ───────────────────────────────────────────────
bst.settings.set_thermo(['Water', 'AceticAcid', 'EthylAcetate'])

# ── Streams ────────────────────────────────────────────────────────
acetic_acid_broth = bst.Stream(
    'acetic_acid_broth',
    AceticAcid=1000, Water=9000, units='kg/hr'
)
ethyl_acetate = bst.Stream('fresh_EtAc', EthylAcetate=1)
glacial_acetic_acid = bst.Stream('glacial_acetic_acid')
wastewater = bst.Stream('wastewater')

# Recycle streams
solvent_recycle = bst.Stream('solvent_recycle')
water_rich = bst.Stream('water_rich')
distillate = bst.Stream('raffinate_distillate')

solvent_feed_ratio = 1.5

# ── System ─────────────────────────────────────────────────────────
with bst.System('AcAc_separation') as sys:

    # 1) Liquid-liquid extractor
    extractor = bst.MixerSettler(
        'E101_Extractor',
        ins=(acetic_acid_broth, ethyl_acetate, solvent_recycle),
        outs=('extract', 'raffinate'),
        top_chemical='EthylAcetate',
    )

    @extractor.add_specification(run=True)
    def adjust_fresh_solvent():
        broth = acetic_acid_broth.F_mass
        EtAc_recycle = solvent_recycle.imass['EthylAcetate']
        EtAc_required = broth * solvent_feed_ratio
        if EtAc_required < EtAc_recycle:
            solvent_recycle.F_mass *= EtAc_required / EtAc_recycle
            EtAc_recycle = solvent_recycle.imass['EthylAcetate']
        ethyl_acetate.imass['EthylAcetate'] = max(0, EtAc_required - EtAc_recycle)

    # 2) Extract heater
    H101 = bst.HXutility(
        'H101_ExtractHeater',
        ins=extractor.extract,
        outs='hot_extract',
        rigorous=True,
        V=0,
    )

    # 3) Extract distiller (ED) — separates Water (LK) from AceticAcid (HK)
    ED = bst.ShortcutColumn(
        'T101_ExtractDistiller',
        ins=H101-0,
        outs=['ED_overhead', 'ED_bottoms'],
        LHK=('Water', 'AceticAcid'),
        Lr=0.95, Hr=0.95, k=1.4,
        partial_condenser=False,
    )

    # 4) Acid purification (ED2) — separates EtAc (LK) from AceticAcid (HK)
    ED2 = bst.ShortcutColumn(
        'T102_AcidPurification',
        ins=ED-1,
        outs=('ED2_overhead', glacial_acetic_acid),
        LHK=('EthylAcetate', 'AceticAcid'),
        Lr=0.999, Hr=0.999, k=1.4,
        partial_condenser=False,
    )
    ED.check_LHK = ED2.check_LHK = False

    # 5) Mixer — combine overheads + raffinate distillate for recycle
    M101 = bst.Mixer('M101_OverheadMixer', ins=(ED-0, ED2-0, distillate))

    # 6) Cooling HX — cool for liquid-liquid phase separation
    H102 = bst.HXutility('H102_Cooler', ins=M101-0, T=310)

    # 7) Settler — phase separation
    settler = bst.MixerSettler(
        'V101_Settler',
        ins=H102-0,
        outs=(solvent_recycle, water_rich),
        top_chemical='EthylAcetate',
    )

    # 8) Mixer — combine raffinate + water-rich for solvent recovery
    M102 = bst.Mixer('M102_RaffinateMixer', ins=[extractor.raffinate, water_rich])

    # 9) Raffinate distiller (RD) — recover EtAc from aqueous phase
    RD = bst.ShortcutColumn(
        'T103_RaffinateDistiller',
        LHK=('EthylAcetate', 'Water'),
        ins=M102-0,
        outs=[distillate, wastewater],
        partial_condenser=False,
        Lr=0.99, Hr=0.99, k=1.5,
    )

sys.operating_hours = 330 * 24

# ── Simulate ───────────────────────────────────────────────────────
print("Simulating system...")
sys.simulate()
print("Simulation complete.")

# ── Generate diagrams ──────────────────────────────────────────────
out_dir = os.path.dirname(os.path.abspath(__file__))

for kind in ['thorough', 'surface', 'cluster', 'minimal']:
    # Get graphviz Digraph object (display=False)
    g = sys.diagram(kind, display=False, label=True, title=f'AcAc Separation ({kind})')

    for fmt in ['svg', 'png']:
        fname = os.path.join(out_dir, f'pfd_biosteam_{kind}')
        g.format = fmt
        if fmt == 'png':
            g.graph_attr['dpi'] = '300'
        rendered = g.render(filename=fname, cleanup=True)
        print(f"Saved: {rendered}")

print("\nAll BioSTEAM PFDs generated successfully.")
