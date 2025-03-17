
import pdb
import optuna
import time
import matplotlib.pyplot as plt
import biosteam as bst
import numpy as np
from scipy.optimize import differential_evolution # population based optimization code
from scipy.stats import qmc # latin hypercube sampling 
from skopt import gp_minimize
from skopt.space import Integer, Real
from skopt.utils import use_named_args
import warnings
from BayesianNN import *
import traceback

class ShortCutDesign():
    def __init__(self, verbose=False):
        bst.nbtutorial() # For light-mode diagrams, ignore warnings
        self.set_base_model()
        self.nEval = 0
        self.verbose = verbose
        self.history = []  # 최적화 과정 저장
        self.error_num = 0
        print(" ##### An instance of the 'BlackBox' class  has been initialised!")
    
    def set_base_model(self):
        # relevant values based on the website example
        # _n1 = 12 # number of stages for extractor
        # _Lr1, _Hr1, _k1  = [0.95, 0.95, 1.4]
        # _Lr2, _Hr2, _k2  = [0.999, 0.999, 1.4]
        # _T_hex = 310
        # _Lr3, _Hr3, _k3  = [0.99, 0.99, 1.5]
        X = [12, 0.95, 0.95, 0.999, 0.999,  310, 0.99, 0.99, ]
        self._set(X)
        
    def _bounds(self):
        # bounds for variables. feel free to change!
        bounds = [(10, 50), # no. of stages in extractor
                  (0, 0.9999), (0, 0.9999),  # light key, heavy key, 'k' for extract distiller개ㅕ
                  (0, 0.9999), (0, 0.9999),  # light key, heavy key, 'k' for acetic_acid_purification
                  (273, 350), # temperature for 'HX'
                  (0, 0.9999), (0, 0.9999),  # light key, heavy key, 'k' for reffinate_distiller
                  ]
        return bounds
    
    def _integrality(self):
        # which varibales are integers (True if integer)
        ints = [True, 
                False, False, 
                False, False, 
                False, 
                False, False, 
                ]
        return ints
        
    
    def _set(self, X):
        
        # Define chemicals used in the process
        bst.settings.set_thermo(['Water', 'AceticAcid', 'EthylAcetate'])

        
        _n1 = X[0] # number of stages for extractor
        _Lr1, _Hr1,   = X[1], X[2]
        _Lr2, _Hr2,   = X[3:5]
        _T_hex = X[5]
        _Lr3, _Hr3  = X[6:8]
        
        _k1, _k2, _k3 = (1.4, 1.4, 1.5)
        
        # print(_Lr2)
        # ensure that integer variables are in fact integer
        _n1 = int(_n1)
        
        
        
        # Amount of ethyl-acetate to fermentation broth
        solvent_feed_ratio = 1.5

        # Fermentation broth with dilute acetic acid
        acetic_acid_broth = bst.Stream(ID='acetic_acid_broth', AceticAcid=1000, Water=9000, units='kg/hr')

        # Solvent
        ethyl_acetate = bst.Stream(ID='ethyl_acetate',  EthylAcetate=1)

        # Products
        glacial_acetic_acid = bst.Stream(ID='glacial_acetic_acid')
        wastewater = bst.Stream(ID='wastewater')

        # Recycles
        solvent_recycle = bst.Stream('solvent_rich')
        water_rich = bst.Stream('water_rich')
        distillate = bst.Stream('raffinate_distillate')
        
        
        # System and unit operations
        with bst.System('AAsep') as sys:
            print(f"stages: {_n1}")
            extractor = bst.MultiStageMixerSettlers(
                'extractor',
                ins=(acetic_acid_broth, ethyl_acetate, solvent_recycle),
                outs=('extract', 'raffinate'),
                top_chemical='EthylAcetate',
                feed_stages=(0, -1, -1),
                N_stages=_n1,
                use_cache=True,
            )

            @extractor.add_specification(run=True)
            def adjust_fresh_solvent_flow_rate():
                broth = acetic_acid_broth.F_mass
                EtAc_recycle = solvent_recycle.imass['EthylAcetate']
                EtAc_required = broth * solvent_feed_ratio
                if EtAc_required < EtAc_recycle:
                    solvent_recycle.F_mass *= EtAc_required / EtAc_recycle
                    EtAc_recycle = solvent_recycle.imass['EthylAcetate']
                EtAc_fresh = EtAc_required - EtAc_recycle
                ethyl_acetate.imass['EthylAcetate'] = max(
                    0, EtAc_fresh
                )

            HX = bst.HXutility(
                'extract_heater',
                ins=(extractor.extract),
                outs=('hot_extract'),
                rigorous=True,
                V=0,
            )
            ED = bst.ShortcutColumn(
                'extract_distiller',
                ins=HX-0,
                outs=['', 'acetic_acid'],
                LHK=('Water', 'AceticAcid'),
                Lr=_Lr1,
                Hr=_Hr1,
                k=_k1,
                partial_condenser=False,
            )
            ED2 = bst.ShortcutColumn(
                'acetic_acid_purification',
                ins=ED-1,
                outs=('', glacial_acetic_acid),
                LHK=('EthylAcetate', 'AceticAcid'),
                Lr=_Lr2,
                Hr=_Hr2,
                k=_k2,
                partial_condenser=False
            )
            ED.check_LHK = ED2.check_LHK = False
            mixer = bst.Mixer(
                ins=(ED-0, ED2-0, distillate)
            )
            HX = bst.HXutility(ins=mixer-0, T=_T_hex)
            settler = bst.MixerSettler(
                'settler',
                ins=HX-0,
                outs=(solvent_recycle, water_rich),
                top_chemical='EthylAcetate',
            )
            mixer = bst.Mixer(ins=[extractor.raffinate, water_rich])
            RD = bst.ShortcutColumn(
                'raffinate_distiller',
                LHK=('EthylAcetate', 'Water'),
                ins=mixer-0,
                outs=[distillate, wastewater],
                partial_condenser=False,
                Lr=_Lr3,
                Hr=_Hr3,
                k=_k3,
            )
            
        sys.operating_hours = 330 * 24 # annual operating hours, hr/yr
        self.ED = ED
        self.sys = sys

    def capex(self):
        # capex of equipment in MMUSD/yr
        capex = round(self.sys.installed_equipment_cost / 1e6, 4)
        
        try:
            int(capex) # checks if nan or a number is returned
            return capex
        
        except:
            print('capex_error')
            # return 10
        
    def opex(self):
        # opex of equipment in MMUSD/yr
        opex = round(self.sys.material_cost + self.sys.utility_cost / 1e6, 4)
    
        try:
            int(opex) # checks if nan or a number is returned
            return opex
        
        except:
            print('opex_error')
            # return 10
        
    def cost(self):
        return self.capex() + self.opex()
            
    def MSP(self):
        stream = [stream for stream in self.sys.streams if stream.ID == 'glacial_acetic_acid'][0]
        P_AceticAcid = 0.4 # $/kg
        F_AceticAcid = stream.F_mass * self.sys.operating_hours / 1e6 # kg/yr
        constraint = self.acetic_acid_constraint() 
        if constraint != 0: # return profit if in-specification
            msp = self.cost() / F_AceticAcid # UNits: $/kg
            penalty = 10 * (constraint ** 2)  # 패널티를 제곱으로 조정하여 완만하게 적용
            print('lower')
            return round(msp, 4) + penalty
        else:
            print('lower', F_AceticAcid)
            msp = self.cost() / F_AceticAcid # UNits: $/kg
            if msp > 100:
                msp = 50
                self.error_num += 1
            return round(msp, 4) 
        
    def simulate(self):
        self.nEval += 1
        self.sys.simulate()

    def wt_acetic_acid(self):
        stream = [stream for stream in self.sys.streams if stream.ID == 'glacial_acetic_acid'][0]
        # print(stream.get_mass_fraction(IDs='AceticAcid'))
        return stream.get_mass_fraction(IDs='AceticAcid')
        
    def acetic_acid_constraint(self):
        x_desired = 0.98 # wt%
        x_achieved = self.wt_acetic_acid()
        d_x = x_desired - x_achieved # -ve if  product is in-spec
        
        cons = max(0, d_x) # return 0 if happy or constraint violation if not
        print('acetic_acid: ', x_achieved)
        return cons
    
    def shortcut_results(self, X=None):
        print(X)
        start_time = time.time() 

        try:
            if X is None:
                self.set_base_model()
                
            else:# set and run the simulation
                self._set(X) # set the new operating parameters
                    
            self.simulate() # run the simulation
            outlet = self.ED.reboiler.outs[0]
            boilup = outlet['g'].F_mol / outlet['l'].F_mol
            distillate, condensate = self.ED.top_split.outs
            split = condensate.F_mol / self.ED.condenser.outs[0].F_mol # Or from ED.design_results['Reflux']
            N_stages = int(self.ED.design_results['Theoretical stages'])
            feed_stage = int(self.ED.design_results['Theoretical feed stage'])
            elapsed_time = time.time() - start_time  # ⏳ 경과 시간 기록

        # if failure for any reason, then reutrn a value of np.inf
        except Exception as e:
            print("Error occurred:", e)
            
            objective_function = 100
            elapsed_time = time.time() - start_time  # ⏳ 에러 발생 시에도 시간 기록
            self.history.append([self.nEval, objective_function, self.wt_acetic_acid(), elapsed_time])
            return {"CAPEX": 0,
                    "OPEX": 0,
                    "AceticAcidWt": 0,
                    "BoilupRatio": 0,
                    "Distillate": 0,
                    "SplitRatio": 0,
                    "NumberStages": 0,
                    "FeedStage": 0}

        

        if self.verbose:
            print(f"Iteration {self.nEval}: MSP = {round(objective_function, 2)}, Time = {elapsed_time:.2f} sec")

        return {"CAPEX": self.capex(),
                "OPEX": self.opex(),
                "AceticAcidWt": self.wt_acetic_acid(),
                "BoilupRatio": boilup,
                "Distillate": distillate,
                "SplitRatio": split,
                "NumberStages": N_stages,
                "FeedStage": feed_stage}
    
    def func(self, X=None):
        print(X)
        start_time = time.time() 

        try:
            if X is None:
                self.set_base_model()
                
            else:# set and run the simulation
                self._set(X) # set the new operating parameters
                    
            self.simulate() # run the simulation
            
            # assess plant financials
            objective_function = self.MSP()
            
            elapsed_time = time.time() - start_time  # ⏳ 경과 시간 기록

            self.history.append([self.nEval, objective_function, self.wt_acetic_acid(), elapsed_time])

        # if failure for any reason, then reutrn a value of np.inf
        except Exception as e:
            print("Error occurred:", e)
            # traceback.print_exc()
            
            objective_function = 100
            elapsed_time = time.time() - start_time  # ⏳ 에러 발생 시에도 시간 기록
            self.history.append([self.nEval, objective_function, self.wt_acetic_acid(), elapsed_time])
            
            return objective_function


        if self.verbose:
            print(f"Iteration {self.nEval}: MSP = {round(objective_function, 2)}, Time = {elapsed_time:.2f} sec")

        return objective_function
    
    def natural_units(self, X):
        X_natural = np.zeros((np.shape(X)))
        bounds = self._bounds()
        b = np.array(bounds)
        d_b = b[:,1] - b[:,0] # range of bounds in natural units
        
        for i, x in enumerate(X):
            X_natural[i] = b[:,0] + (x[:] * d_b)
        
        return X_natural
            