import biosteam as bst

# Define function to calculate CAPEX, OPEX, and Acetic Acid purity
def OneColumn(Lr, Hr):
    # try:
        bst.settings.set_thermo(['Water', 'AceticAcid'])
        
        # Fermentation broth with dilute acetic acid
        acetic_acid_broth = bst.Stream(ID='acetic_acid_broth', AceticAcid=1000, Water=9000, units='kg/hr')
        
        # Products
        glacial_acetic_acid = bst.Stream(ID='glacial_acetic_acid')
        wastewater = bst.Stream(ID='wastewater')
        
        # System and unit operations
        with bst.System('AAsep') as sys:
            ED = bst.ShortcutColumn(
                'extract_distiller',
                ins=acetic_acid_broth,  # ED에 직접 연결
                outs=[wastewater, glacial_acetic_acid],  # 상부: Wastewater, 하부: Glacial Acetic Acid
                LHK=('Water', 'AceticAcid'),  # Water 제거, AceticAcid 농축
                Lr=Lr,  # Light key recovery
                Hr=Hr,  # Heavy key recovery
                k=1.4,
                partial_condenser=False,
            )
        
        sys.simulate()
        sys.operating_hours = 330 * 24  # 연간 운영 시간
        
        CAPEX = round(sys.installed_equipment_cost / 1e6, 3)  # MMUSD
        OPEX = round(sys.material_cost + sys.utility_cost / 1e6, 4)  # MMUSD/yr
        
        # Calculate Acetic Acid purity
        stream = [stream for stream in sys.streams if stream.ID == 'glacial_acetic_acid'][0]
        purity = stream.get_mass_fraction(IDs='AceticAcid')

        return CAPEX, OPEX, purity
    # except Exception as e:
    #         print("Error occurred:", e)
    #         return 0, 0, 0
    