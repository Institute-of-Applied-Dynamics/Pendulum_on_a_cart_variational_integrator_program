import sympy
import numpy as np
def weighted_sum(x1,x2,weight):
    return weight * x1 + (1-weight)*x2

def finite_diff(x1,x2,step_size):
    return (x2-x1)/step_size

# to transform the sympy vectors to numpy ones
numpier = sympy.symbols("nper")
to_numpy_transformer = lambda x : sympy.lambdify(numpier,x)
eval_sympy_to_np = lambda x:  to_numpy_transformer(x)(0).flatten()

def sympy_to_np_dict(chosen_dict,alpha_choice,gamma_choice):
    np_dict = {}
    for tmp in chosen_dict:
        if type(chosen_dict[tmp]) is  sympy.Matrix:
            np_dict[tmp] = eval_sympy_to_np(chosen_dict[tmp])
        elif tmp == "h":
            np_dict[tmp] = chosen_dict["T"]/chosen_dict["N"]
        elif tmp == "alpha":
            np_dict[tmp] = alpha_choice     
        elif tmp == "gamma":
            np_dict[tmp] = gamma_choice     
        else:
            np_dict[tmp] = chosen_dict[tmp]
    np_dict["times"] = eval_sympy_to_np(chosen_dict["times"].evalf(subs={"h":np_dict["h"]})   )
    return np_dict