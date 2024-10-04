#!/usr/bin/env python
import argparse
import json
import mealpy
import time
from baselines.mealpy.demo_mealpy_bv import DGProblem
from utils import load_network_fn

ALL = {
    "ABC_OriginalABC": mealpy.swarm_based.ABC.OriginalABC(epoch=1000, pop_size=50, n_limits=50),
    "ACOR_OriginalACOR": mealpy.swarm_based.ACOR.OriginalACOR(epoch=1000, pop_size=50, sample_count=25, intent_factor=0.5, zeta=1.0),
    "AGTO_OriginalAGTO": mealpy.swarm_based.AGTO.OriginalAGTO(epoch=1000, pop_size=50, p1=0.03, p2=0.8, beta=3.0),
    "AGTO_MGTO": mealpy.swarm_based.AGTO.MGTO(epoch=1000, pop_size=50, pp=0.03),
    "ALO_OriginalALO": mealpy.swarm_based.ALO.OriginalALO(epoch=1000, pop_size=50),
    "ALO_DevALO": mealpy.swarm_based.ALO.DevALO(epoch=1000, pop_size=50),
    "AO_OriginalAO": mealpy.swarm_based.AO.OriginalAO(epoch=1000, pop_size=50),
    "ARO_OriginalARO": mealpy.swarm_based.ARO.OriginalARO(epoch=1000, pop_size=50),
    "ARO_LARO": mealpy.swarm_based.ARO.LARO(epoch=1000, pop_size=50),
    "ARO_IARO": mealpy.swarm_based.ARO.IARO(epoch=1000, pop_size=50),
    "AVOA_OriginalAVOA": mealpy.swarm_based.AVOA.OriginalAVOA(epoch=1000, pop_size=50, p1=0.6, p2=0.4, p3=0.6, alpha=0.8, gama=2.5),
    "BA_OriginalBA": mealpy.swarm_based.BA.OriginalBA(epoch=1000, pop_size=50, loudness=0.8, pulse_rate=0.95, pf_min=0.1, pf_max=10.0),
    "BA_AdaptiveBA": mealpy.swarm_based.BA.AdaptiveBA(epoch=1000, pop_size=50, loudness_min=1.0, loudness_max=2.0, pr_min=0.15, pr_max=0.85, pf_min=0, pf_max=10.),
    "BA_DevBA": mealpy.swarm_based.BA.DevBA(epoch=1000, pop_size=50, pulse_rate=0.95, pf_min=0., pf_max=10.),
    "BES_OriginalBES": mealpy.swarm_based.BES.OriginalBES(epoch=1000, pop_size=50, a_factor=10, R_factor=1.5, alpha=2.0, c1=2.0, c2=2.0),
    "BFO_OriginalBFO": mealpy.swarm_based.BFO.OriginalBFO(epoch=1000, pop_size=50, Ci=0.01, Ped=0.25, Nc=5, Ns=4, d_attract=0.1, w_attract=0.2, h_repels=0.1, w_repels=10),
    "BFO_ABFO": mealpy.swarm_based.BFO.ABFO(epoch=1000, pop_size=50, C_s=0.1, C_e=0.001, Ped=0.01, Ns=4, N_adapt=2, N_split=40),
    "BSA_OriginalBSA": mealpy.swarm_based.BSA.OriginalBSA(epoch=1000, pop_size=50, ff=10, pff=0.8, c1=1.5, c2=1.5, a1=1.0, a2=1.0, fc=0.5),
    "BeesA_CleverBookBeesA": mealpy.swarm_based.BeesA.CleverBookBeesA(epoch=1000, pop_size=50, n_elites=16, n_others=4),
    "BeesA_OriginalBeesA": mealpy.swarm_based.BeesA.OriginalBeesA(epoch=1000, pop_size=50, selected_site_ratio=0.5, elite_site_ratio=0.4),
    "BeesA_ProbBeesA": mealpy.swarm_based.BeesA.ProbBeesA(epoch=1000, pop_size=50, recruited_bee_ratio=0.1, dance_radius=0.1, dance_reduction=0.99),
    "COA_OriginalCOA": mealpy.swarm_based.COA.OriginalCOA(epoch=1000, pop_size=50, n_coyotes=5),
    "CSA_OriginalCSA": mealpy.swarm_based.CSA.OriginalCSA(epoch=1000, pop_size=50, p_a=0.3),
    "CSO_OriginalCSO": mealpy.swarm_based.CSO.OriginalCSO(epoch=1000, pop_size=50, mixture_ratio=0.15, smp=5, spc=False, cdc=0.8, srd=0.15, c1=0.4, w_min=0.4, w_max=0.9),
    "CoatiOA_OriginalCoatiOA": mealpy.swarm_based.CoatiOA.OriginalCoatiOA(epoch=1000, pop_size=50),
    "DMOA_OriginalDMOA": mealpy.swarm_based.DMOA.OriginalDMOA(epoch=1000, pop_size=50, n_baby_sitter=3, peep=2),
    "DMOA_DevDMOA": mealpy.swarm_based.DMOA.DevDMOA(epoch=1000, pop_size=50, peep=2),
    "DO_OriginalDO": mealpy.swarm_based.DO.OriginalDO(epoch=1000, pop_size=50),
    "EHO_OriginalEHO": mealpy.swarm_based.EHO.OriginalEHO(epoch=1000, pop_size=50, alpha=0.5, beta=0.5, n_clans=5),
    "ESOA_OriginalESOA": mealpy.swarm_based.ESOA.OriginalESOA(epoch=1000, pop_size=50),
    "FA_OriginalFA": mealpy.swarm_based.FA.OriginalFA(epoch=1000, pop_size=50, max_sparks=50, p_a=0.04, p_b=0.8, max_ea=40, m_sparks=50),
    "FFA_OriginalFFA": mealpy.swarm_based.FFA.OriginalFFA(epoch=1000, pop_size=50, gamma=0.001, beta_base=2, alpha=0.2, alpha_damp=0.99, delta=0.05, exponent=2),
    "FFO_OriginalFFO": mealpy.swarm_based.FFO.OriginalFFO(epoch=1000, pop_size=50),
    "FOA_OriginalFOA": mealpy.swarm_based.FOA.OriginalFOA(epoch=1000, pop_size=50),
    "FOA_DevFOA": mealpy.swarm_based.FOA.DevFOA(epoch=1000, pop_size=50),
    "FOA_WhaleFOA": mealpy.swarm_based.FOA.WhaleFOA(epoch=1000, pop_size=50),
    "FOX_OriginalFOX": mealpy.swarm_based.FOX.OriginalFOX(epoch=1000, pop_size=50, c1=0.18, c2=0.82),
    # "FOX_DevFOX": mealpy.swarm_based.FOX.DevFOX(epoch=1000, pop_size=50, c1=0.18, c2=0.82, pp=0.5),
    "GJO_OriginalGJO": mealpy.swarm_based.GJO.OriginalGJO(epoch=1000, pop_size=50),
    "GOA_OriginalGOA": mealpy.swarm_based.GOA.OriginalGOA(epoch=1000, pop_size=50, c_min=0.00004, c_max=1.0),
    "GTO_OriginalGTO": mealpy.swarm_based.GTO.OriginalGTO(epoch=1000, pop_size=50, A=0.4, H=2.0),
    "GTO_Matlab102GTO": mealpy.swarm_based.GTO.Matlab102GTO(epoch=1000, pop_size=50),
    "GTO_Matlab101GTO": mealpy.swarm_based.GTO.Matlab101GTO(epoch=1000, pop_size=50),
    "GWO_OriginalGWO": mealpy.swarm_based.GWO.OriginalGWO(epoch=1000, pop_size=50),
    "GWO_RW_GWO": mealpy.swarm_based.GWO.RW_GWO(epoch=1000, pop_size=50),
    "GWO_GWO_WOA": mealpy.swarm_based.GWO.GWO_WOA(epoch=1000, pop_size=50),
    "GWO_IGWO": mealpy.swarm_based.GWO.IGWO(epoch=1000, pop_size=50, a_min=0.02, a_max=2.2),
    "HBA_OriginalHBA": mealpy.swarm_based.HBA.OriginalHBA(epoch=1000, pop_size=50),
    "HGS_OriginalHGS": mealpy.swarm_based.HGS.OriginalHGS(epoch=1000, pop_size=50, PUP=0.08, LH=10000),
    "HHO_OriginalHHO": mealpy.swarm_based.HHO.OriginalHHO(epoch=1000, pop_size=50),
    "JA_DevJA": mealpy.swarm_based.JA.DevJA(epoch=1000, pop_size=50),
    "JA_OriginalJA": mealpy.swarm_based.JA.OriginalJA(epoch=1000, pop_size=50),
    "JA_LevyJA": mealpy.swarm_based.JA.LevyJA(epoch=1000, pop_size=50),
    "MFO_OriginalMFO": mealpy.swarm_based.MFO.OriginalMFO(epoch=1000, pop_size=50),
    "MGO_OriginalMGO": mealpy.swarm_based.MGO.OriginalMGO(epoch=1000, pop_size=50),
    "MPA_OriginalMPA": mealpy.swarm_based.MPA.OriginalMPA(epoch=1000, pop_size=50),
    "MRFO_OriginalMRFO": mealpy.swarm_based.MRFO.OriginalMRFO(epoch=1000, pop_size=50, somersault_range=2.0),
    # "MRFO_OriginalMRFO": mealpy.swarm_based.MRFO.OriginalMRFO(epoch=1000, pop_size=50, somersault_range=2.0, pm=0.5),
    "MSA_OriginalMSA": mealpy.swarm_based.MSA.OriginalMSA(epoch=1000, pop_size=50, n_best=5, partition=0.5, max_step_size=1.0),
    "NGO_OriginalNGO": mealpy.swarm_based.NGO.OriginalNGO(epoch=1000, pop_size=50),
    "NMRA_OriginalNMRA": mealpy.swarm_based.NMRA.OriginalNMRA(epoch=1000, pop_size=50, pb=0.75),
    "NMRA_ImprovedNMRA": mealpy.swarm_based.NMRA.ImprovedNMRA(epoch=1000, pop_size=50, pb=0.75, pm=0.01),
    "OOA_OriginalOOA": mealpy.swarm_based.OOA.OriginalOOA(epoch=1000, pop_size=50),
    "PFA_OriginalPFA": mealpy.swarm_based.PFA.OriginalPFA(epoch=1000, pop_size=50),
    "POA_OriginalPOA": mealpy.swarm_based.POA.OriginalPOA(epoch=1000, pop_size=50),
    "PSO_OriginalPSO": mealpy.swarm_based.PSO.OriginalPSO(epoch=1000, pop_size=50, c1=2.05, c2=2.05, w=0.4),
    "PSO_AIW_PSO": mealpy.swarm_based.PSO.AIW_PSO(epoch=1000, pop_size=50, c1=2.05, c2=2.05, alpha=0.4),
    "PSO_LDW_PSO": mealpy.swarm_based.PSO.LDW_PSO(epoch=1000, pop_size=50, c1=2.05, c2=2.05, w_min=0.4, w_max=0.9),
    "PSO_P_PSO": mealpy.swarm_based.PSO.P_PSO(epoch=1000, pop_size=50),
    "PSO_HPSO_TVAC": mealpy.swarm_based.PSO.HPSO_TVAC(epoch=1000, pop_size=50, ci=0.5, cf=0.1),
    "PSO_C_PSO": mealpy.swarm_based.PSO.C_PSO(epoch=1000, pop_size=50, c1=2.05, c2=2.05, w_min=0.4, w_max=0.9),
    "PSO_CL_PSO": mealpy.swarm_based.PSO.CL_PSO(epoch=1000, pop_size=50, c_local=1.2, w_min=0.4, w_max=0.9, max_flag=7),
    "SCSO_OriginalSCSO": mealpy.swarm_based.SCSO.OriginalSCSO(epoch=1000, pop_size=50),
    "SFO_OriginalSFO": mealpy.swarm_based.SFO.OriginalSFO(epoch=1000, pop_size=50, pp=0.1, AP=4.0, epsilon=0.0001),
    "SFO_ImprovedSFO": mealpy.swarm_based.SFO.ImprovedSFO(epoch=1000, pop_size=50, pp=0.1),
    "SHO_OriginalSHO": mealpy.swarm_based.SHO.OriginalSHO(epoch=1000, pop_size=50, h_factor=5.0, n_trials=10),
    "SLO_OriginalSLO": mealpy.swarm_based.SLO.OriginalSLO(epoch=1000, pop_size=50),
    "SLO_ModifiedSLO": mealpy.swarm_based.SLO.ModifiedSLO(epoch=1000, pop_size=50),
    "SLO_ImprovedSLO": mealpy.swarm_based.SLO.ImprovedSLO(epoch=1000, pop_size=50, c1=1.2, c2=1.5),
    "SRSR_OriginalSRSR": mealpy.swarm_based.SRSR.OriginalSRSR(epoch=1000, pop_size=50),
    "SSA_DevSSA": mealpy.swarm_based.SSA.DevSSA(epoch=1000, pop_size=50, ST=0.8, PD=0.2, SD=0.1),
    "SSA_OriginalSSA": mealpy.swarm_based.SSA.OriginalSSA(epoch=1000, pop_size=50, ST=0.8, PD=0.2, SD=0.1),
    "SSO_OriginalSSO": mealpy.swarm_based.SSO.OriginalSSO(epoch=1000, pop_size=50),
    "SSpiderA_OriginalSSpiderA": mealpy.swarm_based.SSpiderA.OriginalSSpiderA(epoch=1000, pop_size=50, r_a=1.0, p_c=0.7, p_m=0.1),
    "SSpiderO_OriginalSSpiderO": mealpy.swarm_based.SSpiderO.OriginalSSpiderO(epoch=1000, pop_size=50, fp_min=0.65, fp_max=0.9),
    "STO_OriginalSTO": mealpy.swarm_based.STO.OriginalSTO(epoch=1000, pop_size=50),
    "SeaHO_OriginalSeaHO": mealpy.swarm_based.SeaHO.OriginalSeaHO(epoch=1000, pop_size=50),
    "ServalOA_OriginalServalOA": mealpy.swarm_based.ServalOA.OriginalServalOA(epoch=1000, pop_size=50),
    "TDO_OriginalTDO": mealpy.swarm_based.TDO.OriginalTDO(epoch=1000, pop_size=50),
    "TSO_OriginalTSO": mealpy.swarm_based.TSO.OriginalTSO(epoch=1000, pop_size=50),
    "WOA_OriginalWOA": mealpy.swarm_based.WOA.OriginalWOA(epoch=1000, pop_size=50),
    "WOA_HI_WOA": mealpy.swarm_based.WOA.HI_WOA(epoch=1000, pop_size=50, feedback_max=10),
    "WaOA_OriginalWaOA": mealpy.swarm_based.WaOA.OriginalWaOA(epoch=1000, pop_size=50),
    "ZOA_OriginalZOA": mealpy.swarm_based.ZOA.OriginalZOA(epoch=1000, pop_size=50),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--pos', type=int, default=0)
    parser.add_argument('--model', choices=ALL.keys(), required=True)
    parser.add_argument('--store-stats')
    args = parser.parse_args()

    times, machines, (g1, g2) = load_network_fn(args.data, args.pos)
    print(times)
    print(machines)
    print(g1)
    print(g2)
    n, m = times.shape

    p = DGProblem(g1, g2)
    model = ALL[args.model]

    start = time.perf_counter_ns()
    model.solve(p, mode="swarm")
    stop = time.perf_counter_ns()

    # print(model.g_best.solution)
    print("best solution", model.g_best.target.fitness)
    print('duration', stop - start)

    if args.store_stats:
        stats = {
            'duration': stop - start,
            'best': model.g_best.target.fitness,
        }
        with open(args.store_stats, 'w') as f:
            json.dump(stats, f)


if __name__ == '__main__':
    main()
