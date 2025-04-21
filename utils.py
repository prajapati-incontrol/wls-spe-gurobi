import pandapower as pp 
import pandas as pd 
import numpy as np 
from typing import List, Tuple, Dict 
from collections import defaultdict

def get_branch_parameters(net: pp.pandapowerNet): 
    """
    param_dict = {(i, j): [G_ij, B_ij, g_s_ij, b_s_ij, g_sh_ij, b_sh_ij]}
    """
    from_bus, to_bus = net.line.from_bus.values, net.line.to_bus.values
    hv_bus, lv_bus = net.trafo.hv_bus.values, net.trafo.lv_bus.values

    src_bus = np.concat([from_bus, hv_bus])
    tgt_bus = np.concat([to_bus, lv_bus])

    branch_tuple = [(int(u),int(v)) for u, v in zip(src_bus, tgt_bus)]

    # only lines 
    line_tuple = [(int(u),int(v)) for u, v in zip(from_bus, to_bus)]
    trafo_tuple = [(int(u),int(v)) for u, v in zip(hv_bus, lv_bus)]

    param_dict = {branch: [0.0]*6 for branch in branch_tuple}

    y_series, y_shunt = get_branch_admittance(net, line_tuple, trafo_tuple)

    for branch in param_dict.keys(): 
        # for power injection equation 
        param_dict[branch][0] = float(np.real(-y_series[branch]))
        param_dict[branch][1] = float(np.imag(-y_series[branch]))
        param_dict[branch][2] = float(np.real(y_series[branch]))
        param_dict[branch][3] = float(np.imag(y_series[branch]))
        param_dict[branch][4] = float(np.real(y_shunt[branch]))
        param_dict[branch][5] = float(np.imag(y_shunt[branch]))


    new_param_dict = dict(param_dict)  # make a copy so we don't modify while iterating

    for (i, j), value in param_dict.items():
        if (j, i) not in new_param_dict:
            new_param_dict[(j, i)] = value

    return new_param_dict 

def get_branch_admittance(net: pp.pandapowerNet, 
                                 line_tuple: List[Tuple], 
                                 trafo_tuple: List[Tuple]) -> Tuple[Dict, Dict]: 
    """
    Calculates the complex series admittance for each branch as an effective line. 
    # TODO: Tap-changers. 
    """
    branch_tuple = line_tuple + trafo_tuple
    y_series = {branch: [] for branch in branch_tuple}
    y_shunt = {branch: [] for branch in branch_tuple}

    ################ LINE PARAMETERS ########################   
    sys_freq = 50 # Hz 

    # Per-Unit values 
    # convert r_ohm_per_km to r_per_unit
    r_pu = (net.line['r_ohm_per_km'] * net.line['length_km'] / net.line['parallel']) / (net.bus['vn_kv'].loc[net.line['from_bus'].values]**2/net.sn_mva).values 

    # convert x_ohm_per_km to x_ohm
    x_pu = (net.line['x_ohm_per_km'] * net.line['length_km'] / net.line['parallel']) / (net.bus['vn_kv'].loc[net.line['from_bus'].values]**2/net.sn_mva).values

    # convert c_nf_per_km to b_pu
    b_pu = ( 2 * np.pi * sys_freq * net.line['c_nf_per_km'] * 10**(-9) * net.line['length_km'] * net.line['parallel']) * (net.bus['vn_kv'].loc[net.line['from_bus'].values]**2/net.sn_mva).values

    # convert g_us_per_km to g_pu
    g_pu = ( 2 * np.pi * sys_freq * net.line['g_us_per_km'] * 10**(-6) * net.line['length_km'] * net.line['parallel']) * (net.bus['vn_kv'].loc[net.line['from_bus'].values]**2/net.sn_mva).values 


    z_pu = r_pu + 1j*x_pu
    y_series_pu = 1 / z_pu # series admittance 


    y_shunt_pu = g_pu - 1j*b_pu # shunt admittance 

    a_1_pu = y_series_pu + y_shunt_pu/2
    a_2_pu = - y_series_pu
    a_3_pu = - y_series_pu
    a_4_pu = y_series_pu + y_shunt_pu/2

    for idx, line in enumerate(line_tuple): 
        y_series[line] = y_series_pu[idx]
        y_shunt[line] = y_shunt_pu[idx]
    
    yb_size = len(net.bus)

    Ybline = np.zeros((yb_size, yb_size)).astype(complex)

    fb_line = net.line.from_bus 
    tb_line = net.line.to_bus 

    line_idx = net.line.index 

    for (idx, fb, tb) in zip(line_idx, fb_line, tb_line): 
        Ybline[fb, fb] = complex(a_1_pu[idx])
        Ybline[fb, tb] = complex(a_2_pu[idx])
        Ybline[tb, fb] = complex(a_3_pu[idx])
        Ybline[tb, tb] = complex(a_4_pu[idx])

    # get the pandpaower internal YBus 
    pp.runpp(net)
    Ybus = np.array(net._ppc["internal"]["Ybus"].todense())

    ################# TRANSFORMER PARAMETERS ######################
    # derived from Ybus matrix. 
    Ybus_trafo = Ybus - Ybline

    if sum(net.trafo.tap_pos.isna()) > 0: 
        print("Filling nan as 0 in tap_pos, tap_neutral, tap_step_degree")
        net.trafo.loc[net.trafo.loc[:,'tap_pos'].isna(),'tap_pos'] = 0
        net.trafo.loc[net.trafo.loc[:,'tap_neutral'].isna(),'tap_neutral'] = 0
        net.trafo.loc[net.trafo.loc[:,'tap_step_degree'].isna(),'tap_step_degree'] = 0
        
    ps_s = net.trafo.shift_degree * np.pi/180
    tap_nom_s = 1 + (net.trafo.tap_pos - net.trafo.tap_neutral) * (net.trafo.tap_step_percent / 100)
    N_tap_s = tap_nom_s * np.exp(1j*ps_s)

    for id, trafo_edge in zip(net.trafo.index.values, trafo_tuple):
        hv_bus, lv_bus = trafo_edge 
        a_1t_pu = Ybus_trafo[hv_bus, hv_bus]
        a_2t_pu = Ybus_trafo[hv_bus, lv_bus]
        a_3t_pu = Ybus_trafo[lv_bus, hv_bus]
        a_4t_pu = Ybus_trafo[lv_bus, lv_bus]

        # series impedance 
        y_series[trafo_edge] = - a_3t_pu * N_tap_s[id]

        # shunt impedance 
        y_shunt[trafo_edge] = 2* (a_4t_pu - y_series[trafo_edge]) 


    return y_series, y_shunt

#####################################################################################

def get_edge_index_from_ppnet(net):
    """
    This function creates an edge-index tensor ([2, num_edges]) from pandapower net,
    considering both lines (from_bus to to_bus) and transformers (hv_bus to lv_bus).

    Args:
        net: Pandapower Net

    Returns:
        torch.Tensor: Edge-index in COO format; Shape [2, num_edges]
    """
    edges = []
    
    # Add lines (from_bus to to_bus)
    for _, line in net.line.iterrows():
        edges.append((line.from_bus, line.to_bus))
    
    # Add transformers (hv_bus to lv_bus)
    for _, trafo in net.trafo.iterrows():
        edges.append((trafo.hv_bus, trafo.lv_bus))
    
    edge_index = np.array(edges).T

    return edge_index


#####################################################################################
    
def get_neighbor_dict(edge_index: np.ndarray) -> Dict: 
    """
    # neighbor_dict = {src: [neighbors]}
    """
    neighbor_dict = defaultdict(set)
    for src, tgt in edge_index.T: 
        neighbor_dict[src.item()].add(tgt.item())
        neighbor_dict[tgt.item()].add(src.item())
    neighbor_dict = {node: sorted(list(neighbors)) for node, neighbors in neighbor_dict.items()}

    return neighbor_dict


