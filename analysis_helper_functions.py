import os 
import re
import matplotlib.pyplot as plt
import pickle 
def create_foldername_from_parameters(alpha,beta,gamma,u_dep_case):
    '''get here the file/folder names
    Careful, not checked for ints or floats (a=1, vs. a=1.0 etc is not distinguished!)'''
    if u_dep_case:
        folder_name = "u_dep_data_a="+str(alpha)+"g="+str(gamma)
    else:
        folder_name = "no_u_dep_data_a="+str(alpha)+"g="+str(gamma)
    return folder_name

def get_floats_from_filename(filename):
    return re.findall(r"[+-]? *(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", filename)


def create_dict_for_folder(folderdir):
    file_names_in_folder = os.listdir(folderdir)
    stepsize_filename_dict = dict()
    for x in file_names_in_folder:
        stepsize_filename_dict[x] = [float(y)  for y in get_floats_from_filename(x)] #assume fix order a,b,gamma,h
    return stepsize_filename_dict    
    


def calculate_conserved_quantity_evolution(q_d,lam_d,u_d_1,u_d_2,field,cofield,params):
    conserved_quantity = []
    for k in range(params["N"]+1):
        dq_val = dqk(q_d,lam_d,u_d_1,u_d_2,k,field,params)
        dlam_val = dlam_k(q_d,lam_d,u_d_1,u_d_2,k,cofield,params)
        conserved_quantity.append(calculate_conserved_quantity(q_d[k],dq_val,lam_d[k],dlam_val,params))
    return conserved_quantity


def get_fig_1_plot_data_from_dict(dict_with_data,is_explicit_u_dep,has_two_controls=False):
    
    if has_two_controls and is_explicit_u_dep: 
        q_d,lam_d,mu,nu,u_d_1,u_d_2 = dict_with_data["q_d"],dict_with_data["lam_d"],dict_with_data["mu"],dict_with_data["nu"],dict_with_data["U_d_1"],dict_with_data["U_d_2"]
    else:
        q_d,lam_d,mu,nu,u_d = dict_with_data["q_d"],dict_with_data["lam_d"],dict_with_data["mu"],dict_with_data["nu"],dict_with_data["U_d"]

    parameters = dict_with_data["parameters"]

    phase_space_evo = [[],[]]
    for x in q_d:
        tmp = x
        phase_space_evo[0].append(tmp[0,0])
        phase_space_evo[1].append(tmp[1,0])
    time_vals = parameters["times"]
    phase_space_lamevo = [[],[]]
    for x in lam_d:
        # tmp = Cartesian_trafo(x)
        tmp = x
        phase_space_lamevo[0].append(tmp[0,0])
        phase_space_lamevo[1].append(tmp[1,0])
    if is_explicit_u_dep and has_two_controls:
        u_d_1_vals = u_d_1
        u_d_2_vals = u_d_2   
        conserved_I_evo = calculate_conserved_quantity_evolution(q_d,lam_d,u_d_1,u_d_2,vector_field_u_eval,covector_field_u_eval,parameters)
        return time_vals,phase_space_evo,phase_space_lamevo,u_d_1_vals,u_d_2_vals,conserved_I_evo,parameters
    elif is_explicit_u_dep:
        print('warning, no old style needs its own conserved quantity calc funciton, but not implemented right now(copy from other or import it from there too)')
        return time_vals,phase_space_evo,phase_space_lamevo,u_d,parameters
        
    else:
        u_d_vals = [get_u_from_lambda(x,y,parameters) for (x,y) in zip(lam_d,q_d)]
        conserved_I_evo = calculate_conserved_quantity_evolution(q_d,lam_d,u_d,u_d,vector_field_eval,covector_field_eval,parameters)
        return time_vals,phase_space_evo,phase_space_lamevo,u_d_vals,conserved_I_evo,parameters


def calculate_velocity_paths(list_with_all,is_control_dependent=True):
        '''only u-dependent right now'''
        v_q_d = np.zeros([2,len(list_with_all['q_d'])])
        v_lam_d = np.zeros([2,len(list_with_all['q_d'])])

        if is_control_dependent:
            q_d_vals,lam_d_vals, u_d_1_vals,u_d_2_vals = list_with_all['q_d'],list_with_all['lam_d'],list_with_all['U_d_1'],list_with_all['U_d_2']
            vector_field = vector_field_u_eval
            covector_field = covector_field_u_eval

            for i in range(len(v_q_d[0])):
                    v_q_d[0][i], v_q_d[1][i] = dqk(q_d_vals,lam_d_vals,u_d_1_vals,u_d_2_vals,i,vector_field,list_with_all['parameters']).flatten()

            for i in range(len(v_lam_d[0])):
                    v_lam_d[0][i], v_lam_d[1][i] = dlam_k(q_d_vals,lam_d_vals,u_d_1_vals,u_d_2_vals,i,covector_field,list_with_all['parameters']).flatten()
        else:   
            q_d_vals,lam_d_vals, u_d_1_vals,u_d_2_vals = list_with_all['q_d'],list_with_all['lam_d'],None,None
            vector_field = vector_field_eval
            covector_field = covector_field_eval

            for i in range(len(v_q_d[0])):
                    v_q_d[0][i], v_q_d[1][i] = dqk(q_d_vals,lam_d_vals,u_d_1_vals,u_d_2_vals,i,vector_field,list_with_all['parameters']).flatten()

            for i in range(len(v_lam_d[0])):
                    v_lam_d[0][i], v_lam_d[1][i] = dlam_k(q_d_vals,lam_d_vals,u_d_1_vals,u_d_2_vals,i,covector_field,list_with_all['parameters']).flatten()
             
        return v_q_d, v_lam_d



def plot_curve_with_velocity_arrows(axs_obj,curve_plot,v_plot,spacing=5,furtherparamsdict={'length_includes_head':True,'head_width':0.1}): 
        for i in range(len(v_plot[0])):
                if i % spacing ==0:
                        # axs_obj.arrow(curve_plot[0][i],curve_plot[1][i],v_plot[0][i],v_plot[1][i],**furtherparamsdict)
                        axs_obj.annotate("",xytext=(curve_plot[0][i],curve_plot[1][i]),xy=(curve_plot[0][i]+v_plot[0][i],v_plot[1][i]+curve_plot[1][i]),arrowprops=dict(arrowstyle="->"))



def create_hamilton_evo(dict_with_data,is_control_dependent):
    params = dict_with_data["parameters"]
    Hamilton_vals = []
    if not is_control_dependent:
        q_d = dict_with_data["q_d"] 
        lam_d = dict_with_data["lam_d"] 
        for k in range(len(q_d)):
            Hamilton_vals.append(eval_hamilton_func_discrete_path(q_d,lam_d,k,params))
    else:
        q_d = dict_with_data["q_d"]
        lam_d = dict_with_data["lam_d"]
        U_d_1= dict_with_data["U_d_1"]
        U_d_2= dict_with_data["U_d_2"]
        for k in range(len(q_d)):
            Hamilton_vals.append(eval_hamilton_func_discrete_path(q_d,lam_d,k,params,[U_d_1,U_d_2]))
    return params['times'], np.array(Hamilton_vals)


def standard_cost_control_plot_data(datadict):
        q_d_standard,U_d_standard = datadict['q_d'], datadict['U_d']
        phase_space_evo_standard = [[],[]]
        for x in q_d_standard:
                tmp = x
                phase_space_evo_standard[0].append(tmp[0,0])
                phase_space_evo_standard[1].append(tmp[1,0])
        time_vals = datadict['parameters']["times"]

        standard_total_cost=calculate_total_running_cost(q_d_standard,U_d_standard,datadict['parameters'])
        return standard_total_cost, phase_space_evo_standard,U_d_standard.flatten(), time_vals


def evaluate_running_cost_integrand(q,u,parameters):
    return u.transpose()@(g(q,parameters)@u)

def calculate_total_running_cost(q_d,U_d,parameters):
    total_cost = 0
    alpha=parameters["alpha"]
    beta = parameters["beta"]
    gamma = parameters["gamma"]
    for k in range(parameters["N"]):
        q_wht = weighted_avg(q_d[k], q_d[k+1], gamma)
        U_wht = weighted_avg(U_d[k],U_d[k+1],beta)
        total_cost += parameters["h"]* alpha * evaluate_running_cost_integrand(q_wht,U_wht,parameters)/2
        q_wht = weighted_avg(q_d[k], q_d[k+1], 1-gamma)
        U_wht = weighted_avg(U_d[k],U_d[k+1],1-beta)
        total_cost += parameters["h"]*(1-alpha) * evaluate_running_cost_integrand(q_wht,U_wht,parameters)/2
    return total_cost  