import numpy as np
from time import perf_counter
from pyomo.environ import *
import copy
import pandas as pd
from mpi4py import MPI
# import os

from snl_progress.mod_sysdata import RASystemData
from snl_progress.mod_solar import Solar
from snl_progress.mod_wind import Wind
from snl_progress.mod_utilities import RAUtilities
from snl_progress.mod_matrices import RAMatrices
from snl_progress.mod_plot import RAPlotTools

def MCS(samples, sim_hours, system_directory, solar_directory = True, wind_directory = True) :
    """
    Performs mixed time sequential Monte Carlo Simulation (MCS) to evaluate the reliability of a power system.

    Parameters:
        samples (int): Number of samples to simulate.
        sim_hours (int): Number of hours to simulate.
        system_directory (str): Path to the directory containing system data files.
        solar_directory (str or bool): Path to the directory containing solar data files or False if not used.
        wind_directory (str or bool): Path to the directory containing wind data files or False if not used.

    Returns:
        tuple: A tuple containing indices, rank, SOC records, curtailment records, renewable records, bus names, and ESS names.
    """

    # system data
    data_gen = system_directory + '/gen.csv'
    data_branch = system_directory + '/branch.csv'
    data_bus = system_directory + '/bus.csv'
    data_load = system_directory + '/load.csv'
    data_storage = system_directory + '/storage.csv'
    BMva = 100

    rasd = RASystemData()
    genbus, ng, pmax, pmin, FOR_gen, MTTF_gen, MTTR_gen, gencost = rasd.gen(data_gen)
    nl, fb, tb, cap_trans, MTTF_trans, MTTR_trans = rasd.branch(data_branch)
    bus_name, bus_no, nz = rasd.bus(data_bus)
    load_all_regions = rasd.load(bus_name, data_load)

    essname, essbus, ness, ess_pmax, ess_pmin, ess_duration, ess_socmax, ess_socmin, ess_eff, \
        disch_cost, ch_cost, MTTF_ess, MTTR_ess, ess_units = rasd.storage(data_storage)

    raut = RAUtilities()
    mu_tot, lambda_tot = raut.reltrates(MTTF_gen, MTTF_trans, MTTR_gen, MTTR_trans, MTTF_ess, MTTR_ess)
    cap_max, cap_min = raut.capacities(nl, pmax, pmin, ess_pmax, ess_pmin, cap_trans) # calling this function to get values of cap_max and cap_min

    # wind data
    if wind_directory:

        wind_sites = wind_directory + '/wind_sites.csv'
        wind_power_curves = wind_directory + '/w_power_curves.csv'
        wind_tr_rate = wind_directory + '/t_rate.xlsx'

        wind = Wind()
        w_sites, farm_name, zone_no, w_classes, w_turbines, r_cap, p_class, out_curve2, out_curve3,\
            start_speed = wind.WindFarmsData(wind_sites, wind_power_curves)
        tr_mats = pd.read_excel(wind_tr_rate, sheet_name=None)
        tr_mats = np.array([tr_mats[sheet_name].to_numpy() for sheet_name in tr_mats])

    # solar data
    if solar_directory:
        solar_site_data = solar_directory+"/solar_sites.csv"
        solar_prob_data = solar_directory+"/solar_probs.csv"

        solar = Solar(solar_site_data, solar_directory)
        s_sites, s_zone_no, s_max, s_profiles, solar_prob = solar.GetSolarProfiles(solar_prob_data)

        # # concentrated solar
        # CSP_all_buses = solar.CSP(nh, data_CSP)

    # matrices required for optimization
    ramat = RAMatrices(nz)
    gen_mat = ramat.genmat(ng, genbus, ness, essbus)
    ch_mat = ramat.chmat(ness, essbus, nz)
    A_inc = ramat.Ainc(nl, fb, tb)
    curt_mat = ramat.curtmat(nz)

    # dictionary for storing temp. index values
    indices_rec = {"LOLP_rec": np.zeros(samples), "EUE_rec": np.zeros(samples), "MDT_rec": np.zeros(samples), \
               "LOLF_rec": np.zeros(samples), "EPNS_rec": np.zeros(samples), "LOLP_hr": np.zeros(sim_hours), "LOLE_rec": np.zeros(samples)}

    LOL_track = np.zeros((samples, sim_hours))

    for s in range(samples):

        # temp variables to be used for each sample
        var_s = {"t_min": 0, "LLD": 0, "curtailment": np.zeros(sim_hours), "label_LOLF": np.zeros(sim_hours), "freq_LOLF": 0, "LOL_days": 0, \
                 "outage_day": np.zeros(365)}

        # current states of components
        current_state = np.ones(ng + nl + ness) # all gens and TLs in up state at the start of the year

        if wind_directory:
            current_w_class = np.floor(np.random.uniform(0, 1, w_sites)*w_classes).astype(int) # starting wind speed class for each site (random)

        # record data for plotting and exporting (optional)
        renewable_rec = {"wind_rec": np.zeros((nz, sim_hours)), "solar_rec": np.zeros((nz, sim_hours)), "congen_temp": 0, \
                        "rengen_temp": 0}

        SOC_old = 0.5*(np.multiply(np.multiply(ess_pmax, ess_duration), ess_socmax))/BMva
        SOC_rec = np.zeros((ness, sim_hours))
        curt_rec = np.zeros(sim_hours)
        # gen_rec = np.zeros((sim_hours, ng))

        for n in range(sim_hours):

            # get current states(up/down) and capacities of all system components
            next_state, current_cap, var_s["t_min"] = raut.NextState(var_s["t_min"], ng, ness, nl, \
                                                                     lambda_tot, mu_tot, current_state, cap_max, cap_min, ess_units)
            current_state = copy.deepcopy(next_state)

            # update SOC based on failures in ESS
            ess_smax, ess_smin, SOC_old = raut.updateSOC(ng, nl, current_cap, ess_pmax, ess_duration, ess_socmax, ess_socmin, SOC_old)

            # calculate upper and lower bounds of gens and tls
            gt_limits = {"g_lb": np.concatenate((current_cap["min"][0:ng]/BMva, current_cap["min"][ng + nl::]/BMva)), \
                         "g_ub": np.concatenate((current_cap["max"][0:ng]/BMva, current_cap["max"][ng + nl::]/BMva)), "tl": current_cap["max"][ng:ng + nl]/BMva}

            def fb_Pg(model, i):
                return (gt_limits["g_lb"][i], gt_limits["g_ub"][i])

            def fb_flow(model,i):
                return (-gt_limits["tl"][i], gt_limits["tl"][i])

            def fb_ess(model, i):
                return(-current_cap["max"][ng + nl::][i]/BMva, current_cap["min"][ng + nl::][i]/BMva)

            def fb_soc(model, i):
                return(ess_smin[i]/BMva, ess_smax[i]/BMva)

            # get wind power output for all zones/areas
            if wind_directory:
                w_zones, current_w_class = raut.WindPower(nz, w_sites, zone_no, \
                w_classes, r_cap, current_w_class, tr_mats, p_class, w_turbines, out_curve2, out_curve3)

            # get solar power output for all zones/areas
            if solar_directory:
                s_zones = raut.SolarPower(n, nz, s_zone_no, solar_prob, s_profiles, s_sites, s_max)

            # record wind and solar profiles for plotting (optional)
            if wind_directory:
                renewable_rec["wind_rec"][:, n] = w_zones

            if solar_directory:
                s_zones_t = np.transpose(s_zones)
                renewable_rec["solar_rec"][:, n] = s_zones_t[:, n%24]

            # recalculate net load (for distribution side resources, optional)
            part_netload = 2*load_all_regions

            if solar_directory and wind_directory:
                net_load =  part_netload[n] - w_zones - s_zones[n%24]
            elif solar_directory==False and wind_directory:
                net_load = part_netload[n] - w_zones
            elif solar_directory and wind_directory==False:
                net_load = part_netload[n] - s_zones[n%24]
            elif solar_directory==False and wind_directory==False:
                net_load = part_netload[n]

            # optimize dipatch and calculate load curtailment
            load_curt, SOC_old = raut.OptDispatch(ng, nz, nl, ness, fb_ess, fb_soc, BMva, fb_Pg, fb_flow, A_inc, gen_mat, curt_mat, ch_mat, \
                                                  gencost, net_load, SOC_old, ess_pmax, ess_eff, disch_cost, ch_cost)

            # record values for visualization purposes
            SOC_rec[:, n] = SOC_old*BMva
            curt_rec[n] = load_curt*BMva
            # gen_rec[n] = gen[0:ng]

            # track loss of load states
            var_s, LOL_track = raut.TrackLOLStates(load_curt, BMva, var_s, LOL_track, s, n)

            if n%100 == 0:
                print(n)

        print(s)

        # collect indices for all samples
        indices_rec = raut.UpdateIndexArrays(indices_rec, var_s, sim_hours, s)

        # calculate reliability indices for the MCS
        indices = raut.GetReliabilityIndices(indices_rec, sim_hours, samples)

    # for parallel processing
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    index_all = raut.ParallelProcessing(indices, LOL_track, comm, rank, size, samples)

    return(index_all, rank, SOC_rec, curt_rec, renewable_rec, bus_name, essname)

if __name__ == "__main__":

    tic = perf_counter()

    # inputs
    system_directory = "path/to/system/directory"; solar_directory = "path/to/solar/directory";
    wind_directory = "path/to/wind/directory";
    samples = 1;
    sim_hours = 100;

    indices, rank, SOC_rec, curt_rec, renewable_rec, bus_name, essname = \
        MCS(samples, sim_hours, system_directory, solar_directory = solar_directory, wind_directory = wind_directory)

    print(indices)

    if rank == 0:

        # plot wind gen, solar gen, ESS SOC, load curtailment, ...
        rapt = RAPlotTools()
        rapt.PlotSolarGen(renewable_rec["solar_rec"], bus_name)
        rapt.PlotWindGen(renewable_rec["wind_rec"], bus_name)
        rapt.PlotSOC(SOC_rec, essname)
        rapt.PlotLoadCurt(curt_rec)

    toc = perf_counter()

    print("Codes finished executing in", toc - tic, " seconds")
