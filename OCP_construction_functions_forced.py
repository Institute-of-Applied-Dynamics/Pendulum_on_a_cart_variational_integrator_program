import sympy
import helper_funcs as hfct
import numpy as np
import scipy.optimize as opt

#  discretizer standard and new

def eq_discretizer_midpoint(dgl_rhs, all_var,dgl_var_slot,params):
    '''discretizer for standard direct method, for general problems, applying general low order family
    '''
    h = params["h"]
    qk = sympy.Matrix(sympy.symbols([str(x) + "_k " for x in all_var[0]]))
    qk1 = sympy.Matrix(sympy.symbols([str(x) + "_k1 " for x in all_var[0]]))
    vqk = sympy.Matrix(sympy.symbols([str(x) + "_k " for x in all_var[1]]))
    vqk1 = sympy.Matrix(sympy.symbols([str(x) + "_k1 " for x in all_var[1]]))
    lamqk = sympy.Matrix(sympy.symbols([str(x) + "_k " for x in all_var[2]]))
    lamqk1 = sympy.Matrix(sympy.symbols([str(x) + "_k1 " for x in all_var[2]]))
    lamvk = sympy.Matrix(sympy.symbols([str(x) + "_k " for x in all_var[3]]))
    lamvk1 = sympy.Matrix(sympy.symbols([str(x) + "_k1 " for x in all_var[3]]))
    u1k = sympy.Matrix(sympy.symbols([str(x) + "1_k " for x in all_var[4]]))
    u1k1 = sympy.Matrix(sympy.symbols([str(x) + "1_k1 " for x in all_var[4]]))
    u2k = sympy.Matrix(sympy.symbols([str(x) + "2_k " for x in all_var[4]]))
    u2k1 = sympy.Matrix(sympy.symbols([str(x) + "2_k1 " for x in all_var[4]]))
    all_var_k = [qk, vqk, lamqk, lamvk,u1k,u2k ]
    all_var_k1 = [qk1, vqk1, lamqk1, lamvk1,u1k1,u2k1 ]
    alpha = params["alpha"]
    gamma = params["gamma"]
    ##### node equation
    dgl_var = all_var[dgl_var_slot]
    state_dgl_eval = dgl_var.subs([(tmp1,tmp2) for tmp1,tmp2 in zip(dgl_var,all_var_k1[dgl_var_slot])])
    state_dgl_eval -= dgl_var.subs([(tmp1,tmp2) for tmp1,tmp2 in zip(dgl_var,all_var_k[dgl_var_slot])])
    

    rhsterm = h * dgl_rhs
    rhsterm1 = rhsterm
    #nonsymplectic low thrust evaluation of q, v
    for i in range(len(all_var)-1):
        rhsterm1= rhsterm1.subs([[tmp1,(gamma*tmp2+(1-gamma)*tmp3)] for tmp1,tmp2,tmp3 in zip(all_var[i],all_var_k[i], all_var_k1[i] )])
    rhsterm1 = alpha*rhsterm1.subs([tmp1,tmp2] for tmp1,tmp2 in zip(all_var[-1],all_var_k[-2])) #u1k sub

    rhsterm2 = rhsterm
    for i in range(len(all_var)-1):
        rhsterm2= rhsterm2.subs([[tmp1,(gamma*tmp3+(1-gamma)*tmp2)] for tmp1,tmp2,tmp3 in zip(all_var[i],all_var_k[i], all_var_k1[i] )])
    rhsterm2 = (1-alpha)*rhsterm2.subs([tmp1,tmp2] for tmp1,tmp2 in zip(all_var[-1],all_var_k[-1])) #u2k sub
    
    state_dgl_eval -= rhsterm1    
    state_dgl_eval -= rhsterm2    
    return {"state_eqs":state_dgl_eval, "vars_k":all_var_k,"vars_k1":all_var_k1}

def discrete_control_Lagrangian(control_Lagrangian_func, all_var,params):
    '''generating the discrete control Lagrangian for the new approach from a given continuous function'''
    dim_q = params["dim_q"]
    dim_u = params["dim_u"]
    alpha = params["alpha"]
    gamma = params["gamma"]
    h = params["h"]
    qk = sympy.Matrix(sympy.symbols([str(x) + "_k " for x in all_var[0]]))
    qk1 = sympy.Matrix(sympy.symbols([str(x) + "_k1 " for x in all_var[0]]))
    vqk = sympy.Matrix(sympy.symbols([str(x) + "_k " for x in all_var[1]]))
    vqk1 = sympy.Matrix(sympy.symbols([str(x) + "_k1 " for x in all_var[1]]))
    lamqk = sympy.Matrix(sympy.symbols([str(x) + "_k " for x in all_var[2]]))
    lamqk1 = sympy.Matrix(sympy.symbols([str(x) + "_k1 " for x in all_var[2]]))
    vlamk = sympy.Matrix(sympy.symbols([str(x) + "_k " for x in all_var[3]]))
    vlamk1 = sympy.Matrix(sympy.symbols([str(x) + "_k1 " for x in all_var[3]]))
    U1k = sympy.Matrix(sympy.symbols([str(x) + "1_k " for x in all_var[4]]))
    U2k = sympy.Matrix(sympy.symbols([str(x) + "2_k " for x in all_var[4]]))
    U1k1 = sympy.Matrix(sympy.symbols([str(x) + "1_k1 " for x in all_var[4]]))
    U2k1 = sympy.Matrix(sympy.symbols([str(x) + "2_k1 " for x in all_var[4]]))
    all_var_k = [qk, lamqk,U1k,U2k ]
    all_var_k1 = [qk1, lamqk1,U1k1,U2k1 ]

    q_weight_g = hfct.weighted_sum(qk,qk1,gamma)
    q_weight_1_g = hfct.weighted_sum(qk,qk1,1-gamma)
    lam_weight_g = hfct.weighted_sum(lamqk,lamqk1,gamma)
    lam_weight_1_g = hfct.weighted_sum(lamqk,lamqk1,1-gamma)
    deltaq = hfct.finite_diff(qk1,qk,h)
    deltalam = hfct.finite_diff(lamqk1,lamqk,h)
    discrete_control_L = h* alpha*control_Lagrangian_func(q_weight_g,lam_weight_g,deltaq,deltalam,U1k,params)
    discrete_control_L+= h* (1-alpha)*control_Lagrangian_func(q_weight_1_g,lam_weight_1_g,deltaq,deltalam,U2k,params)
    return {"discrete_control_L":discrete_control_L, "vars_k":all_var_k,"vars_k1":all_var_k1}


###################################################################################
###################################################################################
#Discrete OCP equation generators forced Lagrangian
###################################################################################
###################################################################################
###################################################################################

# Note: new approach variables are q,xi, v_q, v_xi, but use in the code q, lam_q, v_q, v_lam

class Direct_continuous_generator_forced_L:
    '''Generate continuous eqs for direct standard and new approach. Optional I_func may be given if there is a conserved control quantity'''
    def __init__(self,parameters,Lagrangian,f_L,running_cost_func,mayer_func,g_func,I_func = None):
        self.parameters = parameters
        q_names,vq_names,u_names,lamq_names,lamv_names, vlam_names = "","","","","",""
        aq_names = ""
        pq_names, plam_names = "",""
        u_names = ""
        for tmp in range(self.parameters["dim_u"]):
            u_names += "u_"+ str(tmp)+","
        for tmp in self.parameters["variable_names"]:
            q_names += tmp +","
            vq_names += "v_"+tmp +","
            lamq_names += "lambda_"+tmp +","
            lamv_names += "lambda_v"+tmp +","
            vlam_names += "v_lambda_"+tmp +","
            pq_names += "p_"+tmp +","
            plam_names += "p_lambda_"+tmp +","
            aq_names += "a_"+tmp +","
            
        self.q = sympy.Matrix(sympy.symbols(q_names))#("r, phi "))
        self.vq = sympy.Matrix(sympy.symbols(vq_names))#("v_r, v_phi"))
        self.aq = sympy.Matrix(sympy.symbols(aq_names))#("v_r, v_phi"))
        self.u = sympy.Matrix(sympy.symbols(u_names))#("u,"))
        self.lamq = sympy.Matrix(sympy.symbols(lamq_names))#("lambda_r, lambda_phi"))
        self.lamv = sympy.Matrix(sympy.symbols(lamv_names))#("lambda_vr, lambda_vphi"))
        self.vlam = sympy.Matrix(sympy.symbols(vlam_names))#("v_lambda_r, v_lambda_phi"))
        self.pq = sympy.Matrix(sympy.symbols(pq_names))#("v_lambda_r, v_lambda_phi"))
        self.plam = sympy.Matrix(sympy.symbols(plam_names))#("v_lambda_r, v_lambda_phi"))
        
        self.Lagrangian = Lagrangian
        self.f_L = f_L
        self.calc_v_xi_from_standard_variables()
        self.f_vec = self.velocity_vector_field
        # self.rho_vec = rho_func
        self.running_cost = running_cost_func
        self.mayer_term = mayer_func
        self.g_mat = g_func
        self.new_control_H_func = self.new_control_H()
        self.u_lambda_from_newH()
        self.parameters["dim_q"] = len(self.q) 
        self.parameters["dim_u"] = len(self.u)
        # self.usub  = ((self.g_mat(self.q,self.parameters).inv()) @self.rho_vec(self.q,self.vq,self.parameters).transpose()@self.lamv)
        # self.usubdict = {tmp1:tmp2 for tmp1, tmp2 in zip(self.u,self.usub)}
        if I_func is not None:
            self.I_func = I_func(self.control_lagrangian(self.q,self.lamq,self.vq,self.vlam,self.u,self.parameters),self.q,self.lamq,self.vq,self.vlam,self.u,self.parameters)
        else:
            self.I_func = None  
   
    def mech_Euler_Lagrange_eq(self,params):
        q,vq = self.q,self.vq
        aq = self.aq
        u=self.u
        Lagrangian = self.Lagrangian(q,vq,params)
        D2L = sympy.derive_by_array(Lagrangian,vq)
        D1L = sympy.Matrix(sympy.derive_by_array(Lagrangian,q))
        f_L = self.f_L(q,vq,u,params)
        # D12L = sympy.Matrix(sympy.derive_by_array(D2L,q))
        res = sympy.derive_by_array(D2L,q)
        D12Lvq = 0*vq
        for tmp1,tmp2 in zip(res,vq):
            D12Lvq+= sympy.Matrix(sympy.flatten(tmp1*tmp2))
        D22Laq = 0*aq
        res = sympy.derive_by_array(D2L,vq)
        for tmp1,tmp2 in zip(res,aq):
            D22Laq+= sympy.Matrix(sympy.flatten(tmp1*tmp2))   
        mechEL = D22Laq +D12Lvq - D1L-f_L
        return mechEL
    
    def velocity_vector_field(self,q,vq,u,params):
        mechEL = self.mech_Euler_Lagrange_eq(params)
        u_var = self.u
        aq = self.aq
        q_var = self.q
        vq_var = self.vq
        solved_res = sympy.solve(sympy.flatten(mechEL),sympy.flatten(aq))
        field = [0 for _ in range(len(aq))]
        for i in range(len(aq)):
            field[i] = solved_res[aq[i]]
        field_vec = sympy.Matrix(field)
        return field_vec.subs([[tmp1,tmp2] for tmp1,tmp2 in zip(q_var,q)]).subs([[tmp1,tmp2] for tmp1,tmp2 in zip(vq_var,vq)]).subs([[tmp1,tmp2] for tmp1,tmp2 in zip(u_var,u)])
   
    def state_right_hand_side(self):
        '''return vector of the r.h.s. of the first-order state equations of q, with signature (q, v_q)
        in the low thrust case we thus have (r,phi, v_r, v_phi)
        '''   
        q,vq= self.q, self.vq
        params = self.parameters
        q_vec = vq
        u= self.u

        vq_vec = self.f_vec(q,vq,u,params)                 
        return q_vec, vq_vec

    def initial_conditions(self):
        q0 = self.q-self.parameters["q0"]
        dq0 = self.vq - self.parameters["dq0"]
        return q0, dq0    
    def control_lagrangian(self,q,lam,vq,vlam,u,params):
        q_var,vq_var = self.q,self.vq
        lam_var,vlam_var = self.lamq,self.vlam
        u_var=self.u
        Lagrangian = self.Lagrangian(q_var,vq_var,params)
        D2L = sympy.Matrix(sympy.derive_by_array(Lagrangian,vq_var))
        D1L = sympy.Matrix(sympy.derive_by_array(Lagrangian,q_var))
        control_Lagrangian = sympy.Matrix(sympy.flatten(D2L.T@vlam_var))
        control_Lagrangian += sympy.Matrix(sympy.flatten(sympy.Matrix(D1L + self.f_L(q_var,vq_var,u_var,params)).T@ lam_var ))
        control_Lagrangian -= sympy.Matrix(sympy.flatten(self.running_cost(q_var,u_var,params)))
        control_Lagrangian=control_Lagrangian.subs([[tmp1,tmp2] for tmp1,tmp2 in zip(q_var,q)]).subs([[tmp1,tmp2] for tmp1,tmp2 in zip(lam_var,lam)])
        control_Lagrangian=control_Lagrangian.subs([[tmp1,tmp2] for tmp1,tmp2 in zip(vq_var,vq)]).subs([[tmp1,tmp2] for tmp1,tmp2 in zip(vlam_var,vlam)])
        control_Lagrangian=control_Lagrangian.subs([[tmp1,tmp2] for tmp1,tmp2 in zip(u_var,u)])
        return control_Lagrangian
        # return vlam.T @vq + lam.T@(self.f_vec(q,vq,u,params) )- self.running_cost(q,u,params) 
    

    def u_lam_relation(self,q,lam,vq,vlam,u,params):
        control_L = self.control_lagrangian(q,lam,vq,vlam,u,params)[0]
        u_rel = sympy.derive_by_array(control_L,u)
        return u_rel
    def u_calc(self):
        identity= sympy.solve(sympy.flatten(self.u_lam_relation(self.q,self.lamq,self.vq,self.vlam,self.u,self.parameters)),self.u)
        retlist = []
        for tmp in self.u:
            retlist.append(identity[tmp])
        return sympy.Matrix(retlist)
    def u_eval(self,q,vq,lamq,vlam,u):
        u_expr = self.u_calc()
        varlist = sympy.flatten(self.q) + sympy.flatten(self.vq)+ sympy.flatten(self.lamq) + sympy.flatten(self.vlam)+sympy.flatten(self.u)
        
        u_expr = sympy.lambdify(varlist,u_expr)

        return u_expr(*q,*vq,*lamq,*vlam,*u)[0]
    def new_Lagrangian(self):
        q,vq= self.q, self.vq
        lam,vlam = self.lamq, self.vlam
        u= self.u
        return self.control_lagrangian(q,lam,vq,vlam,u,self.parameters)
    def p_y(self):
        control_L = self.new_Lagrangian()[0]
        p_q = sympy.flatten(sympy.derive_by_array(control_L,self.vq))
        p_lam = sympy.flatten(sympy.derive_by_array(control_L, self.vlam))
        return sympy.Matrix(p_q+p_lam)
    def v_y_as_p_y(self):
        p_y_var = sympy.Matrix(sympy.flatten(self.pq) +sympy.flatten(self.plam))
        v_y_var = sympy.flatten(self.vq)+sympy.flatten(self.vlam)
        inverter = sympy.solve(self.p_y()-p_y_var,v_y_var)
        v_y_as_p_y = sympy.Matrix([inverter[tmp]for tmp in v_y_var])
        return sympy.simplify(v_y_as_p_y)

    def Pontryagins_Hamiltonian(self):
        q = self.q
        vq = self.vq
        lamq = self.lamq
        lamv = self.lamv
        u = self.u
        np_params = self.parameters
        f_eval = self.f_vec(q,vq,u,np_params)

        Pontry_H = lamq.transpose()@vq + lamv.transpose()@(f_eval )- self.running_cost(q,u,np_params)
        return Pontry_H
    def Pontryagin_H_eval(self,q,vq,lamq,lamv,u,params):
        control_H = self.Pontryagins_Hamiltonian()
        
        varlist = sympy.flatten(self.q) + sympy.flatten(self.vq)+ sympy.flatten(self.lamq) + sympy.flatten(self.lamv)+sympy.flatten(self.u)

        return sympy.lambdify(varlist,control_H)(*q,*vq,*lamq,*lamv,*u)[0,0]
    def u_Pontryagin(self):
        u_vec = self.u
        u_lam_rel = sympy.flatten(sympy.derive_by_array(self.Pontryagins_Hamiltonian(),u_vec))
        sol = sympy.solve(sympy.flatten(u_lam_rel),u_vec)
        if len(sol.keys()) == 0:
            print("did not find an explicit u solution")
        return sympy.flatten(u_vec.subs([[tmp,sol[tmp]] for tmp in sympy.flatten(u_vec)]))
    def u_Pontryagin_eval(self,q,vq,lamq,lamv,np_params):
        u_expr = self.u_Pontryagin()
        all_var = sympy.flatten(self.q) + sympy.flatten(self.vq) + sympy.flatten(self.lamq) + sympy.flatten(self.lamv)
        u_lambdified = sympy.lambdify(all_var,u_expr)
        return u_lambdified(*q,*vq,*lamq,*lamv)
    
    def new_control_H(self):
        v_y = sympy.flatten(self.vq) + sympy.flatten(self.vlam)
        v_y_as_p_y = sympy.flatten(self.v_y_as_p_y())
        p_y = sympy.Matrix(sympy.flatten(self.pq) + sympy.flatten(self.plam))
        control_L = self.control_lagrangian(self.q,self.lamq,self.vq,self.vlam,self.u,self.parameters)
        control_H = (p_y.T @ sympy.Matrix(v_y)) - control_L
        control_H = control_H.subs([[tmp1,tmp2] for tmp1,tmp2 in zip(v_y,v_y_as_p_y)])
        # control_H = self.pq.transpose()@ self.plam - self.lamq.transpose()@(self.f_vec(self.q,self.plam,self.parameters)+self.rho_vec(self.q,self.plam,self.parameters)@self.u)
        # control_H += self.running_cost(self.q,self.u,self.parameters)
        varlist = sympy.flatten(self.q) + sympy.flatten(self.lamq)+ sympy.flatten(self.pq) + sympy.flatten(self.plam)+sympy.flatten(self.u)
        self.new_control_H_lambdified  = sympy.lambdify(varlist,control_H)
        return control_H
    def new_control_H_eval(self,q,lamq,pq,plam,u,params):
        control_H = self.new_control_H_func
        
        varlist = sympy.flatten(self.q) + sympy.flatten(self.lamq)+ sympy.flatten(self.pq) + sympy.flatten(self.plam)+sympy.flatten(self.u)

        return self.new_control_H_lambdified(*q,*lamq,*pq,*plam,*u)[0,0]
    def u_lambda_from_newH(self):
        u_vec = self.u
        u_lam_rel = sympy.flatten(sympy.derive_by_array(self.new_control_H(),u_vec))
        sol = sympy.solve(sympy.flatten(u_lam_rel),u_vec)
        if len(sol.keys()) == 0:
            print("did not find an explicit u solution")
        self.u_expr_from_new = sympy.flatten(u_vec.subs([[tmp,sol[tmp]] for tmp in sympy.flatten(u_vec)]))
        all_var = sympy.flatten(self.q) + sympy.flatten(self.lamq) + sympy.flatten(self.pq) + sympy.flatten(self.plam)
        self.u_expr_from_new_lambdified = sympy.lambdify(all_var,self.u_expr_from_new)
        return sympy.flatten(u_vec.subs([[tmp,sol[tmp]] for tmp in sympy.flatten(u_vec)]))
                
    def u_eval_from_new(self,q,lam,pq,plam,params):
        # u_expr = self.u_lambda_from_newH()
        # all_var = sympy.flatten(self.q) + sympy.flatten(self.lamq) + sympy.flatten(self.pq) + sympy.flatten(self.plam)
        # u_lambdified = sympy.lambdify(all_var,u_expr)
        return self.u_expr_from_new_lambdified(*q,*lam,*pq,*plam)

    def calc_xi_from_standard_variables(self,q,lamv):
        D2L = sympy.derive_by_array(self.Lagrangian(self.q,self.vq,self.parameters),self.vq)
        mass_M = sympy.Matrix([sympy.flatten(sympy.derive_by_array(D2L,tmp)) for tmp in self.vq])#,sympy.flatten(sympy.derive_by_array(D2L,self.vq[1]))])
        # inv_mass_M = ((mass_M).inv()
        mass_M_eval = mass_M.subs([[tmp1,tmp2] for tmp1,tmp2 in zip(self.q,q)])
        inv_M_eval =  mass_M_eval.inv() #inv_mass_M.subs([[tmp1,tmp2] for tmp1,tmp2 in zip(self.q,q_d_KKT[i])])
        return inv_M_eval@lamv
    def calc_lam_v_from_new_variables(self,q,xi):
        D2L = sympy.derive_by_array(self.Lagrangian(self.q,self.vq,self.parameters),self.vq)
        mass_M = sympy.Matrix([sympy.flatten(sympy.derive_by_array(D2L,tmp)) for tmp in self.vq])#,sympy.flatten(sympy.derive_by_array(D2L,self.vq[1]))])
        # inv_mass_M = ((mass_M).inv()
        mass_M_eval = mass_M.subs([[tmp1,tmp2] for tmp1,tmp2 in zip(self.q,q)])
        return mass_M_eval@xi
    
    def calc_lamq_from_new_variables(self,q,vq,xi,vxi,u):
        xi_vars = self.xi_vars
        xi_from_lambda = xi #self.calc_xi_from_standard_variables(q,lamv)
        v_xi_calc_terms = self.v_xi_as_lamq_vec.subs([[tmp1,tmp2] for tmp1,tmp2 in zip(self.q,q)])
        v_xi_calc_terms = v_xi_calc_terms.subs([[tmp1,tmp2] for tmp1,tmp2 in zip(self.vq,vq)])
        v_xi_calc_terms = v_xi_calc_terms.subs([[tmp1,tmp2] for tmp1,tmp2 in zip(xi_vars,xi_from_lambda)])
        v_xi_calc_terms = v_xi_calc_terms.subs([[tmp1,tmp2] for tmp1,tmp2 in zip(self.u,u)])
        lamq = sympy.solve(v_xi_calc_terms - sympy.Matrix(vxi), self.lamq)
        lamq_vec = sympy.Matrix([lamq[tmp] for tmp in self.lamq])
        return lamq_vec

    def calc_v_xi_from_standard_variables(self):
        '''very ugly and inefficient way to compute v_xi'''
        xi_var_names,vxi_var_names = '',''
        for tmp in self.q:
            xi_var_names += 'xi_' + str(tmp) + ','
            vxi_var_names += 'v_xi_' + str(tmp) + ','

        self.xi_vars = sympy.Matrix(sympy.symbols(xi_var_names))
        self.v_xi_vars = sympy.Matrix(sympy.symbols(vxi_var_names))
        xi_vars = self.xi_vars
        v_xi_vars = self.v_xi_vars
        D1L = sympy.derive_by_array(self.Lagrangian(self.q,self.vq,self.parameters),self.q)
        D2L = sympy.derive_by_array(self.Lagrangian(self.q,self.vq,self.parameters),self.vq)
        f_L = self.f_L(self.q,self.vq,self.u,self.parameters)
        mass_M = sympy.simplify(sympy.Matrix([sympy.flatten(sympy.derive_by_array(D2L,self.vq[0])),sympy.flatten(sympy.derive_by_array(D2L,self.vq[1]))]))
        lamv_id = mass_M@xi_vars

        Dqlamv = 0*lamv_id
        for tmpq,tmpv in zip(self.q,self.vq):
            Dqlamv += (sympy.derive_by_array(lamv_id,tmpq)*tmpv).tomatrix()
        Dvlamv = 0 *lamv_id 
        a_q_expr = sympy.Matrix(sympy.symbols("a_x,a_theta")) #here pro forma, but should vanish -> if not vanishing, then will crash due to no a-handling
        for tmpv,tmpa in zip(self.vq,a_q_expr):
            Dvlamv += (sympy.derive_by_array(lamv_id,tmpv)*tmpa).tomatrix()

        Dxilamv = 0 *lamv_id
        for tmpxi,tmpvxi in zip(xi_vars,v_xi_vars):
            Dxilamv += (sympy.derive_by_array(lamv_id,tmpxi)*tmpvxi).tomatrix()

        D12Lv = 0*D2L.tomatrix()
        for tmpq,tmpv in zip( self.q,self.vq):
            D12Lv += sympy.derive_by_array(D2L, tmpq).tomatrix()*tmpv


        xi_f = (xi_vars.T@(D1L.tomatrix() + f_L - D12Lv))[0]

        Dv_xi_f = sympy.derive_by_array(xi_f,self.vq).tomatrix()


        lam_q_expr =  - Dqlamv - Dvlamv - Dxilamv- Dv_xi_f

        v_xi_as_lamq = sympy.solve((lam_q_expr - self.lamq),sympy.flatten(v_xi_vars))
        v_xi_as_lamq_vec = sympy.Matrix([v_xi_as_lamq[tmp] for tmp in v_xi_vars])
        self.v_xi_as_lamq_vec = v_xi_as_lamq_vec
        return v_xi_as_lamq_vec
    def eval_v_xi(self,q,vq,lamq,lamv,xi,u):
        xi_vars = self.xi_vars
        xi_from_lambda = xi #self.calc_xi_from_standard_variables(q,lamv)
        v_xi_calc_terms = self.v_xi_as_lamq_vec.subs([[tmp1,tmp2] for tmp1,tmp2 in zip(self.q,q)])
        v_xi_calc_terms = v_xi_calc_terms.subs([[tmp1,tmp2] for tmp1,tmp2 in zip(self.vq,vq)])
        v_xi_calc_terms = v_xi_calc_terms.subs([[tmp1,tmp2] for tmp1,tmp2 in zip(self.lamq,lamq)])
        v_xi_calc_terms = v_xi_calc_terms.subs([[tmp1,tmp2] for tmp1,tmp2 in zip(xi_vars,xi_from_lambda)])
        v_xi_calc_terms = v_xi_calc_terms.subs([[tmp1,tmp2] for tmp1,tmp2 in zip(self.u,u)])
        return np.array(sympy.flatten(v_xi_calc_terms))

class discrete_standard_direct_eq_generator_forced_L:
    def __init__(self,PMP_low_thrust_eq ):
        self.cont_equations = PMP_low_thrust_eq
        self.N = self.cont_equations.parameters["N"]
        self.h = self.cont_equations.parameters["h"]
        mu_str_prefac,mu_str = "mu_",""
        for tmp in self.cont_equations.q:
            mu_str += mu_str_prefac+str(tmp) + ","
        self.mu =  sympy.Matrix(sympy.symbols(mu_str))
        nu_str_prefac,nu_str = "nu_",""
        for tmp in self.cont_equations.vq:
            nu_str += nu_str_prefac+str(tmp) + ","
        self.nu =  sympy.Matrix(sympy.symbols(nu_str))

        self.all_vars =  [self.cont_equations.q,self.cont_equations.vq,self.cont_equations.lamq,self.cont_equations.lamv,self.cont_equations.u]
        self.all_vars_new_approach =  [self.cont_equations.q,self.cont_equations.vq,self.cont_equations.lamq,self.cont_equations.vlam,self.cont_equations.u]
        self.k_eqs_description_q= eq_discretizer_midpoint(self.cont_equations.state_right_hand_side()[0],self.all_vars,0,self.cont_equations.parameters)
        self.k_eqs_description_vq= eq_discretizer_midpoint(self.cont_equations.state_right_hand_side()[1],self.all_vars,1,self.cont_equations.parameters)
        self.control_L_k = discrete_control_Lagrangian(self.cont_equations.control_lagrangian,self.all_vars_new_approach,self.cont_equations.parameters)
        q_d,vq_d,lamq_d,lamv_d,U_d,U1_d,U2_d = [],[],[],[],[],[],[]
        for i in range(self.N+1):
            q_d.append(sympy.Matrix(sympy.symbols([str(x) + "_"+str(i) for x in self.all_vars[0]])))
            vq_d.append(sympy.Matrix(sympy.symbols([str(x) + "_"+str(i) for x in self.all_vars[1]])))
            lamq_d.append(sympy.Matrix(sympy.symbols([str(x) + "_"+str(i) for x in self.all_vars[2]])))
            lamv_d.append(sympy.Matrix(sympy.symbols([str(x) + "_"+str(i) for x in self.all_vars[3]])))
            U_d.append(sympy.Matrix(sympy.symbols([str(x) + "_"+str(i) for x in self.all_vars[4]])))
            U1_d.append(sympy.Matrix(sympy.symbols([str(x) + "_1d_"+str(i) for x in self.all_vars[4]])))
            U2_d.append(sympy.Matrix(sympy.symbols([str(x) + "_2d_"+str(i) for x in self.all_vars[4]])))
            # vlam_d.append(sympy.Matrix(sympy.symbols([str(x) + "_"+str(i) for x in self.all_vars_new_approach[3]])))

        self.all_d = [q_d,vq_d,lamq_d, lamv_d,U1_d,U2_d]
        self.all_d_new_approach = [q_d,lamq_d,U1_d,U2_d]
    def __k_to_i_eval(self,eq_to_subs, k_sub_vector,i_sub_vector):
        return eq_to_subs.subs([(tmp1,tmp2) for tmp1,tmp2 in zip(k_sub_vector,i_sub_vector)])

    def generate_discrete_differential_eqs(self):
        '''generate here all N algebraic equations to solve for from the differential equation chosen'''
        all_eqs_q = []
        all_eqs_vq = []

        q_k_eqs = self.k_eqs_description_q["state_eqs"]
        vq_k_eqs = self.k_eqs_description_vq["state_eqs"]
        all_vars_k = self.k_eqs_description_q["vars_k"]
        all_vars_k1 = self.k_eqs_description_q["vars_k1"]
        for i in range(self.N):
            all_eqs_q.append(q_k_eqs)
            for j in range(len(all_vars_k)):
                all_eqs_q[-1] = self.__k_to_i_eval(all_eqs_q[-1],all_vars_k[j],self.all_d[j][i])
                all_eqs_q[-1] = self.__k_to_i_eval(all_eqs_q[-1],all_vars_k1[j],self.all_d[j][i+1])
            all_eqs_vq.append(vq_k_eqs)
            for j in range(len(all_vars_k)):
                all_eqs_vq[-1] = self.__k_to_i_eval(all_eqs_vq[-1],all_vars_k[j],self.all_d[j][i])
                all_eqs_vq[-1] = self.__k_to_i_eval(all_eqs_vq[-1],all_vars_k1[j],self.all_d[j][i+1])


        return all_eqs_q,all_eqs_vq
    def generate_initial_constraint(self):
        initial_q = self.all_d[0][0] - self.cont_equations.parameters["q0"]
        initial_vq = self.all_d[1][0] - self.cont_equations.parameters["dq0"]
        return initial_q, initial_vq
    def generate_discrete_running_cost(self):
        all_running_cost = []
        gamma = self.cont_equations.parameters["gamma"]
        alpha = self.cont_equations.parameters["alpha"]
        for i in range(self.N):
            mean_q_gamma = ((1-gamma)*self.all_d[0][i+1]+gamma*self.all_d[0][i])
            mean_q_1_gamma = (gamma*self.all_d[0][i+1]+(1-gamma)*self.all_d[0][i])
            u_1_k = self.all_d[-2][i]
            u_2_k = self.all_d[-1][i]
            # mean_u = self.all_d[-1][i]#(self.all_d[-1][i+1]+ self.all_d[-1][i])/2
            running_cost = self.h*alpha*self.cont_equations.running_cost(mean_q_gamma,u_1_k,self.cont_equations.parameters )
            running_cost += self.h*(1-alpha)*self.cont_equations.running_cost(mean_q_1_gamma,u_2_k,self.cont_equations.parameters )
            all_running_cost.append(running_cost)
            # all_running_cost.append(self.h*self.cont_equations.running_cost(mean_q_1_gamma,u_2_k,self.cont_equations.parameters ))
        return all_running_cost  
    def generate_terminal_cost(self):
        q_N = self.all_d[0][-1]
        vq_N = self.all_d[1][-1]
        mayer_eval = self.cont_equations.mayer_term(q_N, vq_N,self.cont_equations.parameters)
        # print(mayer_eval)
        return mayer_eval    
    def generate_discrete_objective(self):
        '''the objective and the state constraints separately'''
        terminal_cost = self.generate_terminal_cost()
        running_cost = self.generate_discrete_running_cost()
        initial_constraints = self.generate_initial_constraint()
        state_equations = self.generate_discrete_differential_eqs()   

        discrete_objective_function = terminal_cost 
        for tmp in running_cost:
            discrete_objective_function += tmp

        opt_variables = []#+list(self.mu)
        # opt_variables+= list(self.nu)
        for tmp in self.all_d[:2]:
            for t in tmp:
                opt_variables+=list(t)
        for t in self.all_d[-1]:
                opt_variables+=list(t)        
        return {"discrete_objective":discrete_objective_function, "discrete_state_constraints":state_equations,"initial_constraints":initial_constraints, "discrete_variables_flattened_list":opt_variables}

    def generate_discrete_objective_KKT(self):
        '''give here alternatively the objective that can be varied to receive the KKT'''
        terminal_cost = self.generate_terminal_cost()
        running_cost = self.generate_discrete_running_cost()
        initial_constraints = self.generate_initial_constraint()
        state_equations = self.generate_discrete_differential_eqs()   

        discrete_objective_function = terminal_cost 
        for tmp in running_cost:
            discrete_objective_function += tmp
        discrete_objective_function += self.mu.transpose()@initial_constraints[0]    
        discrete_objective_function += self.nu.transpose()@initial_constraints[1]    
        #q equations
        for tmp1, tmp2 in zip(state_equations[0],self.all_d[2][1:]):
            discrete_objective_function += tmp2.transpose()@ tmp1
        #vq equations    
        for tmp1, tmp2 in zip(state_equations[1],self.all_d[3][1:]):
            discrete_objective_function += tmp2.transpose()@ tmp1

        opt_variables = []+list(self.mu)
        opt_variables+= list(self.nu)
        for tmp in self.all_d[:-1]:
            for t in tmp:
                opt_variables+=list(t)
        for t in self.all_d[-1]:
                opt_variables+=list(t)        
        return {"discrete_objective":discrete_objective_function, "discrete_variables_flattened_list":opt_variables}  
    def calc_KKT(self,params_np):
        objective_and_vars = self.generate_discrete_objective_KKT()
        objective = objective_and_vars["discrete_objective"].subs([["alpha",params_np["alpha"]],["gamma",params_np["gamma"]],["h",params_np["h"]]])
        mu_eq = sympy.flatten(sympy.derive_by_array(objective,self.mu))
        nu_eq = sympy.flatten(sympy.derive_by_array(objective,self.nu))
        rest_eqs = []
        for tmp in self.all_d:
            for tmp1 in tmp:
                rest_eqs += sympy.flatten(sympy.derive_by_array(objective,tmp1)) #iterate through all N+1 elements
        return sympy.Array(mu_eq+ nu_eq+ rest_eqs), objective_and_vars["discrete_variables_flattened_list"]

    def vqk1_p(self):
        p_y_k1_p = self.p_y_k1p()
        p_mech_k1_p = sympy.Matrix(p_y_k1_p[2:])
        p_mech_cont = sympy.Matrix(sympy.derive_by_array(self.cont_equations.control_lagrangian(self.cont_equations.q,self.cont_equations.lamq,self.cont_equations.vq,self.cont_equations.vlam,self.cont_equations.u,self.cont_equations.parameters)[0],self.cont_equations.vlam))
        vq_as_func_disc_p = sympy.solve(p_mech_cont  - self.cont_equations.pq, self.cont_equations.vq)
        vq_discrete = sympy.Matrix([vq_as_func_disc_p[tmp] for tmp in self.cont_equations.vq])
        vq_discrete = vq_discrete.subs([[tmp1,tmp2] for tmp1,tmp2 in zip(self.cont_equations.q,self.control_L_k["vars_k1"][0])])
        return  vq_discrete.subs([[tmp1,tmp2] for tmp1,tmp2 in zip(self.cont_equations.pq,p_mech_k1_p)])
     
#  return sympy.derive_by_array(sympy.flatten(self.control_L_k["discrete_control_L"])[0], self.control_L_k["vars_k1"][1])
    def vqk_m(self):
        p_y_k_m = self.p_y_km()
        p_mech_k_m = sympy.Matrix(p_y_k_m[2:])
        p_mech_cont = sympy.Matrix(sympy.derive_by_array(self.cont_equations.control_lagrangian(self.cont_equations.q,self.cont_equations.lamq,self.cont_equations.vq,self.cont_equations.vlam,self.cont_equations.u,self.cont_equations.parameters)[0],self.cont_equations.vlam))
        vq_as_func_disc_p = sympy.solve(p_mech_cont  - self.cont_equations.pq, self.cont_equations.vq)
        vq_discrete = sympy.Matrix([vq_as_func_disc_p[tmp] for tmp in self.cont_equations.vq]).subs([[tmp1,tmp2] for tmp1,tmp2 in zip(self.cont_equations.q,self.control_L_k["vars_k"][0])])
        return vq_discrete.subs([[tmp1,tmp2] for tmp1,tmp2 in zip(self.cont_equations.pq,p_mech_k_m)])


        # return -sympy.derive_by_array(sympy.flatten(self.control_L_k["discrete_control_L"])[0], self.control_L_k["vars_k"][1])  
    def p_y_k1p(self):
            p_lam_k1=sympy.derive_by_array(sympy.flatten(self.control_L_k["discrete_control_L"])[0], self.control_L_k["vars_k1"][1])
            p_q_k1 =sympy.derive_by_array(sympy.flatten(self.control_L_k["discrete_control_L"])[0], self.control_L_k["vars_k1"][0])
            return sympy.Matrix(sympy.flatten(p_q_k1)+ sympy.flatten(p_lam_k1))
    def p_y_km(self):
            p_lam_k=-sympy.derive_by_array(sympy.flatten(self.control_L_k["discrete_control_L"])[0], self.control_L_k["vars_k"][1])
            p_q_k =-sympy.derive_by_array(sympy.flatten(self.control_L_k["discrete_control_L"])[0], self.control_L_k["vars_k"][0])
            return sympy.Matrix(sympy.flatten(p_q_k)+ sympy.flatten(p_lam_k))
    def p_v_d_from_y_d(self,q_d,lam_d,U_1_d,U_2_d,np_params):
        all_vars_to_sub = [q_d,lam_d,U_1_d,U_2_d]
        p_y_d = []
        p_y_k1p = self.p_y_k1p().subs("gamma",np_params["gamma"]).subs("alpha",np_params["alpha"]).subs("h",np_params["h"])
        p_y_km = self.p_y_km().subs("gamma",np_params["gamma"]).subs("alpha",np_params["alpha"]).subs("h",np_params["h"])
        k1_vars = self.control_L_k["vars_k1"]
        k_vars = self.control_L_k["vars_k"]
        for i in range(len(q_d)-1):
            p_y_i = p_y_km
            for j in range(len(k_vars)):
                p_y_i = self.__k_to_i_eval(p_y_i,k_vars[j],all_vars_to_sub[j][i])
                p_y_i = self.__k_to_i_eval(p_y_i,k1_vars[j],all_vars_to_sub[j][i+1])
            p_y_d.append(p_y_i)
        for j in range(len(k_vars)):
                p_y_k1p = self.__k_to_i_eval(p_y_k1p,k_vars[j],all_vars_to_sub[j][-2])
                p_y_k1p = self.__k_to_i_eval(p_y_k1p,k1_vars[j],all_vars_to_sub[j][-1])  
        p_y_d.append(np.array(sympy.flatten(p_y_k1p)))



        return p_y_d          
    def u_d_from_y_d(self,q_d,v_q_d,lam_d,v_lam_d,np_params):
        all_vars_to_sub = [q_d,v_q_d,lam_d,v_lam_d]
        u_d = []
        u_func = self.cont_equations.u_calc()
        all_vars = self.all_vars_new_approach[:-1]
        for i in range(self.cont_equations.parameters["N"]+1):
            u_d.append(sympy.Matrix(u_func))
            for tmp,tmpsub in zip(all_vars,all_vars_to_sub):
                u_d[-1] = u_d[-1].subs([[val,subval] for val,subval in zip(tmp,tmpsub[i]) ])
        return u_d        
    

    def generate_new_control_objective(self):
        '''generate here the new control objective'''


        k1_vars = self.control_L_k["vars_k1"]
        k_vars = self.control_L_k["vars_k"]

        p_y_k_m = self.p_y_km()
        p_y_k1_p = self.p_y_k1p()

        p_mech_k1_p = sympy.Matrix(p_y_k1_p[2:])
        p_mech_k_m = sympy.Matrix(p_y_k_m[2:])

        vq_k1_p = self.vqk1_p()
        vq_k_m = self.vqk_m()

        all_d = self.all_d_new_approach
        v0 = vq_k_m #.subs([[tmp1,tmp2] for tmp1,tmp2 in zip(self.cont_equations.q,all_d[0][0])])
        q_d,lam_d,U1_d,U2_d = self.all_d_new_approach
        pN = sympy.Matrix(p_mech_k1_p)
        p0 = sympy.Matrix(p_mech_k_m)
        for i in range(len(self.all_d_new_approach)):
            v0 = self.__k_to_i_eval(v0,k1_vars[i],all_d[i][1])
            v0 = self.__k_to_i_eval(v0,k_vars[i],all_d[i][0])
            p0 = self.__k_to_i_eval(p0,k1_vars[i],all_d[i][1])
            p0 = self.__k_to_i_eval(p0,k_vars[i],all_d[i][0])
        vN = vq_k1_p
        for i in range(len(self.all_d_new_approach)):
            vN = self.__k_to_i_eval(vN,k1_vars[i],all_d[i][-1])
            vN = self.__k_to_i_eval(vN,k_vars[i],all_d[i][-2])
            pN = self.__k_to_i_eval(pN,k1_vars[i],all_d[i][-1])
            pN = self.__k_to_i_eval(pN,k_vars[i],all_d[i][-2])

        #mayer term    
        discrete_objective_function = sympy.flatten(self.cont_equations.mayer_term(q_d[-1],vN,self.cont_equations.parameters))[0]
        # initial constraints
        discrete_objective_function += sympy.flatten(self.mu.transpose()@(q_d[0]- self.cont_equations.parameters["q0"])    )[0]
        discrete_objective_function += sympy.flatten(self.nu.transpose()@(v0 - self.cont_equations.parameters["dq0"]))[0]
        #partial integration boundary terms
        discrete_objective_function -= sympy.flatten(lam_d[0].transpose()@p0)[0]
       
        discrete_objective_function += sympy.flatten(lam_d[-1].transpose()@pN)[0]
        #Lagrange terms
        # print( sympy.flatten(self.control_L_k["discrete_control_L"]))
        for j in range(self.cont_equations.parameters["N"]):
            discrete_control_L_eval = self.control_L_k["discrete_control_L"]
            for i in range(len(self.all_d_new_approach)):
                discrete_control_L_eval = self.__k_to_i_eval(discrete_control_L_eval,k1_vars[i],all_d[i][j+1])
                discrete_control_L_eval = self.__k_to_i_eval(discrete_control_L_eval,k_vars[i],all_d[i][j])
            discrete_objective_function -= sympy.flatten(discrete_control_L_eval)[0]

        #terminal variation term for lambda_N definition
        D2phi =  sympy.Matrix(sympy.flatten(sympy.diff(self.cont_equations.mayer_term(self.all_vars[0],self.all_vars[1],self.cont_equations.parameters),self.all_vars[1])))
        D2phi =  D2phi.subs([[tmp1,tmp2] for tmp1,tmp2 in zip(self.all_vars[0],all_d[0][-1])]).subs(([[tmp1,tmp2] for tmp1,tmp2 in zip(self.all_vars[1],vN)]))
        opt_variables = []+list(self.mu)
        opt_variables+= list(self.nu)
        for tmp in all_d:
            for t in tmp:
                opt_variables+=list(t)
   
        return {"discrete_objective":discrete_objective_function, "discrete_variables_flattened_list":opt_variables,"D2phi":D2phi}    
    def evaluate_state_eqs(self,qkm1,qk,params):
        objective_and_vars = self.generate_new_control_objective()
        objective = objective_and_vars["discrete_objective"]
        state_eqs = sympy.derive_by_array(objective,self.all_d_new_approach[1][3]) #just get one of them
        for tmp1, tmp2 in zip (self.all_d_new_approach,self.control_L_k["vars_k"]):
            state_eqs.subs([[tmp11,tmp22] for tmp11,tmp22 in zip(sympy.flatten(tmp1[3]),sympy.flatten(tmp2))])
        all_km1=[]
        simplified_expr = state_eqs
        for tmp in self.control_L_k["vars_k"]:
            all_km1.append([sympy.Symbol(str(tmpobj)+'_m_1') for tmpobj in tmp])
        for tmp1, tmp2 in zip(self.all_d_new_approach,self.control_L_k["vars_k"]):
            simplified_expr=simplified_expr.subs([[tmp11, tmp22] for tmp11, tmp22 in zip(tmp1[3],tmp2)])
        for tmp1, tmp2 in zip(self.all_d_new_approach,self.control_L_k["vars_k1"]):
            simplified_expr=simplified_expr.subs([[tmp11, tmp22] for tmp11, tmp22 in zip(tmp1[4],tmp2)])
        for tmp1, tmp2 in zip(self.all_d_new_approach,all_km1):
            simplified_expr=simplified_expr.subs([[tmp11, tmp22] for tmp11, tmp22 in zip(tmp1[2],tmp2)])

        simplified_expr = simplified_expr.subs([["alpha",params["alpha"]],["gamma",params["gamma"]],["h",params["h"]]])
        # solve_for_k1 = sympy.solve(simplified_expr,sympy.Matrix(all_km1))

        fu = simplified_expr.subs([[tmp1,tmp2] for tmp1,tmp2 in zip(sympy.flatten(sympy.Matrix(all_km1[:-3])),qkm1)]).subs([[tmp1,tmp2] for tmp1,tmp2 in zip(sympy.flatten(sympy.Matrix(self.control_L_k["vars_k"][:-3])),qk)])
        fu = fu.subs([[tmp1,tmp2] for tmp1,tmp2 in zip(sympy.flatten(sympy.Matrix(all_km1[2:])),[0,0])]).subs([[tmp1,tmp2] for tmp1,tmp2 in zip(sympy.flatten(sympy.Matrix(self.control_L_k["vars_k"][2:])),[0,0])]).subs([[tmp1,tmp2] for tmp1,tmp2 in zip(sympy.flatten(sympy.Matrix(self.control_L_k["vars_k1"][2:])),[0,0])])
        qk1_getter = sympy.lambdify(sympy.flatten(sympy.Matrix(self.control_L_k["vars_k1"][:-3])),fu)
        qk1_getter_lambda = lambda x: qk1_getter(*x)[:,0]
        qk1_res = opt.root(qk1_getter_lambda,x0=qk)

        return qk1_res.x
    
    def calc_KKT_new(self,params_np):
        objective_and_vars = self.generate_new_control_objective()
        objective = objective_and_vars["discrete_objective"].subs([["alpha",params_np["alpha"]],["gamma",params_np["gamma"]],["h",params_np["h"]]])
        mu_eq = sympy.flatten(sympy.derive_by_array(objective,self.mu))
        nu_eq = sympy.flatten(sympy.derive_by_array(objective,self.nu))
        rest_eqs = []
        for tmp in self.all_d_new_approach:
            for tmp1 in tmp:
                rest_eqs += sympy.flatten(sympy.derive_by_array(objective,tmp1)) #iterate through all N+1 elements

        #by construction, delta lambda_0, delta lambda_N are undefined.
        #   we use therefore their definition D_2 phi = -lambda_N^T
        # and nu = lambda_0
        # in place of the 0's
        N_val = self.cont_equations.parameters["N"]
        dim_q = self.cont_equations.parameters["dim_q"]

        k1_vars = self.control_L_k["vars_k1"]
        k_vars = self.control_L_k["vars_k"]
        vq_k1_p = self.vqk1_p()
        vq_k_m = self.vqk_m()
        all_d = self.all_d_new_approach
        v0 = vq_k_m #.subs([[tmp1,tmp2] f
        q_d,lam_d,U1_d,U2_d = self.all_d_new_approach
        for i in range(len(self.all_d_new_approach)):
            v0 = self.__k_to_i_eval(v0,k1_vars[i],all_d[i][1])
            v0 = self.__k_to_i_eval(v0,k_vars[i],all_d[i][0])
        vN = vq_k1_p #.subs([[tmp1,tmp2] for tmp1,tmp2 in zip(self.cont_equations.q,all_d[0][-1])])
 
        for i in range(len(self.all_d_new_approach)):
            vN = self.__k_to_i_eval(vN,k1_vars[i],all_d[i][-1])
            vN = self.__k_to_i_eval(vN,k_vars[i],all_d[i][-2])

        dlam0term_element = 0* self.cont_equations.vq
        dlamTterm_element = 0* self.cont_equations.vq
        p_mech_cont = sympy.Matrix(sympy.derive_by_array(self.cont_equations.control_lagrangian(self.cont_equations.q,self.cont_equations.lamq,self.cont_equations.vq,self.cont_equations.vlam,self.cont_equations.u,self.cont_equations.parameters)[0],self.cont_equations.vlam))

        for tmp1,tmp2 in zip(self.cont_equations.vq,self.all_d_new_approach[1][0]):
            dlam0term_element += sympy.Matrix(sympy.derive_by_array(p_mech_cont,tmp1))*tmp2
        dlam0term_element = dlam0term_element.subs([[tmpa,tmpb] for tmpa,tmpb in zip(self.cont_equations.q,self.all_d_new_approach[0][0])])   
        dlam0term_element = dlam0term_element.subs([[tmpa,tmpb] for tmpa,tmpb in zip(self.cont_equations.vq,v0)])   
        for tmp1,tmp2 in zip(self.cont_equations.vq,self.all_d_new_approach[1][-1]):
            dlamTterm_element += sympy.Matrix(sympy.derive_by_array(p_mech_cont,tmp1))*tmp2
        dlamTterm_element = dlamTterm_element.subs([[tmpa,tmpb] for tmpa,tmpb in zip(self.cont_equations.q,self.all_d_new_approach[0][-1])])   
        dlamTterm_element = dlamTterm_element.subs([[tmpa,tmpb] for tmpa,tmpb in zip(self.cont_equations.vq,vN )])   

        dlam0term = sympy.flatten(self.nu - dlam0term_element)
        dlamTterm = sympy.flatten(objective_and_vars["D2phi"] + dlamTterm_element)
        rest_eqs[(1+N_val)*dim_q], rest_eqs[(1+N_val)*dim_q+1] = dlam0term
        rest_eqs[2*(1+N_val)*dim_q-2], rest_eqs[2*(1+N_val)*dim_q-1] = dlamTterm
        # for i in range(len(dlam0term)):
        #     rest_eqs[(1+N_val)*dim_q+i] = dlam0term[i]
        #     rest_eqs[2*(1+N_val)*dim_q-len(dlamTterm)+i]= dlamTterm[i]

        return sympy.Array(mu_eq+ nu_eq+ rest_eqs), objective_and_vars["discrete_variables_flattened_list"]


# double pendulum, removal of standard approach due to being too slow

class Direct_continuous_generator_forced_L_double_pendulum:
    '''Generate continuous eqs for direct standard and new approach. Optional I_func may be given if there is a conserved control quantity
    While the factory will try to use sympy to generate everything, it may be possible that this is not possible
    In this case use_numeric_calculation can be set if it is not possible to explicitly solve e.g. the inverse legendre transform
    '''
    def __init__(self,parameters,Lagrangian,f_L,running_cost_func,mayer_func,g_func,use_numeric_calculation=False,I_func = None):
        self.parameters = parameters
        q_names,vq_names,u_names,lamq_names,lamv_names, vlam_names = "","","","","",""
        aq_names = ""
        pq_names, plam_names = "",""
        u_names = ""
        for tmp in range(self.parameters["dim_u"]):
            u_names += "u_"+ str(tmp)+","
        for tmp in self.parameters["variable_names"]:
            q_names += tmp +","
            vq_names += "v_"+tmp +","
            lamq_names += "lambda_"+tmp +","
            lamv_names += "lambda_v"+tmp +","
            vlam_names += "v_lambda_"+tmp +","
            pq_names += "p_"+tmp +","
            plam_names += "p_lambda_"+tmp +","
            aq_names += "a_"+tmp +","
   
        self.q = sympy.Matrix(sympy.symbols(q_names))#("r, phi "))
        self.vq = sympy.Matrix(sympy.symbols(vq_names))#("v_r, v_phi"))
        self.aq = sympy.Matrix(sympy.symbols(aq_names))#("v_r, v_phi"))
        self.u = sympy.Matrix(sympy.symbols(u_names))#("u,"))
        self.lamq = sympy.Matrix(sympy.symbols(lamq_names))#("lambda_r, lambda_phi"))
        self.lamv = sympy.Matrix(sympy.symbols(lamv_names))#("lambda_vr, lambda_vphi"))
        self.vlam = sympy.Matrix(sympy.symbols(vlam_names))#("v_lambda_r, v_lambda_phi"))
        self.pq = sympy.Matrix(sympy.symbols(pq_names))#("v_lambda_r, v_lambda_phi"))
        self.plam = sympy.Matrix(sympy.symbols(plam_names))#("v_lambda_r, v_lambda_phi"))
        
        self.Lagrangian = Lagrangian
        self.f_L = f_L
        
        self.f_vec = self.velocity_vector_field

        self.running_cost = running_cost_func
        self.mayer_term = mayer_func
        self.g_mat = g_func
        self.p_y_legendre = self.p_y()
        # self.new_control_H_func = self.new_control_H()
        self.parameters["dim_q"] = len(self.q) 
        self.parameters["dim_u"] = len(self.u)
        
        self.control_L = self.control_lagrangian(self.q,self.lamq,self.vq,self.vlam,self.u,self.parameters)
        self.u_eval_from_new_velocity()
        # if I_func is not None:
        #     self.I_func = I_func(self.control_lagrangian(self.q,self.lamq,self.vq,self.vlam,self.u,self.parameters),self.q,self.lamq,self.vq,self.vlam,self.u,self.parameters)
        # else:
        #     self.I_func = None  
   
    def mech_Euler_Lagrange_eq(self,params):
        q,vq = self.q,self.vq
        aq = self.aq
        u=self.u
        Lagrangian = self.Lagrangian(q,vq,params)
        D2L = sympy.derive_by_array(Lagrangian,vq)
        D1L = sympy.Matrix(sympy.derive_by_array(Lagrangian,q))
        f_L = self.f_L(q,vq,u,params)
        res = sympy.derive_by_array(D2L,q)
        D12Lvq = 0*vq
        for tmp1,tmp2 in zip(res,vq):
            D12Lvq+= sympy.Matrix(sympy.flatten(tmp1*tmp2))
        D22Laq = 0*aq
        res = sympy.derive_by_array(D2L,vq)
        for tmp1,tmp2 in zip(res,aq):
            D22Laq+= sympy.Matrix(sympy.flatten(tmp1*tmp2))   
        mechEL = D22Laq +D12Lvq - D1L-f_L
        return mechEL
    
    def velocity_vector_field(self,q,vq,u,params):
        mechEL = self.mech_Euler_Lagrange_eq(params)
        u_var = self.u
        aq = self.aq
        q_var = self.q
        vq_var = self.vq
        solved_res = sympy.solve(sympy.flatten(mechEL),sympy.flatten(aq))

        field = [0 for _ in range(len(aq))]
        for i in range(len(aq)):
            if aq[i] in solved_res.keys():
                field[i] = solved_res[aq[i]]
            else:
                field[i] = 0
        field_vec = sympy.Matrix(field)
        return field_vec.subs([[tmp1,tmp2] for tmp1,tmp2 in zip(q_var,q)]).subs([[tmp1,tmp2] for tmp1,tmp2 in zip(vq_var,vq)]).subs([[tmp1,tmp2] for tmp1,tmp2 in zip(u_var,u)])

   
    def state_right_hand_side(self):
        '''return vector of the r.h.s. of the first-order state equations of q, with signature (q, v_q)
        in the low thrust case we thus have (r,phi, v_r, v_phi)
        '''   
        q,vq= self.q, self.vq
        params = self.parameters
        q_vec = vq
        u= self.u

        vq_vec = self.f_vec(q,vq,u,params)                 
        return q_vec, vq_vec
    
    def initial_conditions(self):
        q0 = self.q-self.parameters["q0"]
        dq0 = self.vq - self.parameters["dq0"]
        return q0, dq0    
    def control_lagrangian(self,q,lam,vq,vlam,u,params):
        q_var,vq_var = self.q,self.vq
        lam_var,vlam_var = self.lamq,self.vlam
        u_var=self.u
        Lagrangian = self.Lagrangian(q_var,vq_var,params)
        D2L = sympy.Matrix(sympy.derive_by_array(Lagrangian,vq_var))
        D1L = sympy.Matrix(sympy.derive_by_array(Lagrangian,q_var))
        control_Lagrangian = sympy.Matrix(sympy.flatten(D2L.T@vlam_var))
        control_Lagrangian += sympy.Matrix(sympy.flatten(sympy.Matrix(D1L + self.f_L(q_var,vq_var,u_var,params)).T@ lam_var ))
        control_Lagrangian -= sympy.Matrix(sympy.flatten(self.running_cost(q_var,u_var,params)))
        control_Lagrangian=control_Lagrangian.subs([[tmp1,tmp2] for tmp1,tmp2 in zip(q_var,q)]).subs([[tmp1,tmp2] for tmp1,tmp2 in zip(lam_var,lam)])
        control_Lagrangian=control_Lagrangian.subs([[tmp1,tmp2] for tmp1,tmp2 in zip(vq_var,vq)]).subs([[tmp1,tmp2] for tmp1,tmp2 in zip(vlam_var,vlam)])
        control_Lagrangian=control_Lagrangian.subs([[tmp1,tmp2] for tmp1,tmp2 in zip(u_var,u)])
        return control_Lagrangian
        # return vlam.T @vq + lam.T@(self.f_vec(q,vq,u,params) )- self.running_cost(q,u,params) 
    

    def u_lam_relation(self,q,lam,vq,vlam,u,params):
        control_L = self.control_lagrangian(q,lam,vq,vlam,u,params)[0]
        u_rel = sympy.derive_by_array(control_L,u)
        return u_rel
    def u_calc(self):
        identity= sympy.solve(sympy.flatten(self.u_lam_relation(self.q,self.lamq,self.vq,self.vlam,self.u,self.parameters)),self.u)
        retlist = []
        for tmp in self.u:
            retlist.append(identity[tmp])
        return sympy.Matrix(retlist)
    def u_eval(self,q,vq,lamq,vlam,u):
        u_expr = self.u_calc()
        varlist = sympy.flatten(self.q) + sympy.flatten(self.vq)+ sympy.flatten(self.lamq) + sympy.flatten(self.vlam)+sympy.flatten(self.u)
        
        u_expr = sympy.lambdify(varlist,u_expr)

        return u_expr(*q,*vq,*lamq,*vlam,*u)[0]
    def new_Lagrangian(self):
        q,vq= self.q, self.vq
        lam,vlam = self.lamq, self.vlam
        u= self.u
        return self.control_lagrangian(q,lam,vq,vlam,u,self.parameters)
    def p_y(self):
        control_L = self.new_Lagrangian()[0]
        p_q = sympy.flatten(sympy.derive_by_array(control_L,self.vq))
        p_lam = sympy.flatten(sympy.derive_by_array(control_L, self.vlam))
        return sympy.Matrix(p_q+p_lam)
    def calc_vy_fromq_p_new(self,q,lam,py,np_params):
        p_y_var = sympy.Matrix(sympy.flatten(self.pq) +sympy.flatten(self.plam))
        q_var = self.q
        lam_var = self.lamq
        v_y_var = sympy.flatten(self.vq)+sympy.flatten(self.vlam)
        v_p_relation = (self.p_y_legendre-p_y_var).subs([[tmp1,tmp2] for tmp1,tmp2 in zip(p_y_var,py)])
        v_p_relation = v_p_relation.subs([[tmp1,tmp2] for tmp1,tmp2 in zip(q_var,q)])
        v_p_relation = v_p_relation.subs([[tmp1,tmp2] for tmp1,tmp2 in zip(lam_var,lam)])
        
        sol  = sympy.solve(v_p_relation)
        # sol = sympy.nsolve(v_p_relation,v_y_var,[0,0,0,0,0,0])
        sol_v = np.array([sol[tmp] for tmp in v_y_var],float)
        return sol_v
         
        # inverter = sympy.solve(self.p_y()-p_y_var,v_y_var)

    def v_y_as_p_y(self):
        p_y_var = sympy.Matrix(sympy.flatten(self.pq) +sympy.flatten(self.plam))
        v_y_var = sympy.flatten(self.vq)+sympy.flatten(self.vlam)
        inverter = sympy.solve(self.p_y()-p_y_var,v_y_var)
        v_y_as_p_y = sympy.Matrix([inverter[tmp]for tmp in v_y_var])
        return v_y_as_p_y

    def Pontryagins_Hamiltonian(self):
        q = self.q
        vq = self.vq
        lamq = self.lamq
        lamv = self.lamv
        u = self.u
        np_params = self.parameters
        f_eval = self.f_vec(q,vq,u,np_params)

        Pontry_H = lamq.transpose()@vq + lamv.transpose()@(f_eval )- self.running_cost(q,u,np_params)
        return Pontry_H
    def Pontryagin_H_eval(self,q,vq,lamq,lamv,u,params):
        control_H = self.Pontryagins_Hamiltonian()
        
        varlist = sympy.flatten(self.q) + sympy.flatten(self.vq)+ sympy.flatten(self.lamq) + sympy.flatten(self.lamv)+sympy.flatten(self.u)

        return sympy.lambdify(varlist,control_H)(*q,*vq,*lamq,*lamv,*u)[0,0]
    def u_Pontryagin(self):
        u_vec = self.u
        u_lam_rel = sympy.flatten(sympy.derive_by_array(self.Pontryagins_Hamiltonian(),u_vec))
        sol = sympy.solve(sympy.flatten(u_lam_rel),u_vec)
        if len(sol.keys()) == 0:
            print("did not find an explicit u solution")
        return sympy.flatten(u_vec.subs([[tmp,sol[tmp]] for tmp in sympy.flatten(u_vec)]))
    def u_Pontryagin_eval(self,q,vq,lamq,lamv,np_params):
        u_expr = self.u_Pontryagin()
        all_var = sympy.flatten(self.q) + sympy.flatten(self.vq) + sympy.flatten(self.lamq) + sympy.flatten(self.lamv)
        u_lambdified = sympy.lambdify(all_var,u_expr)
        return u_lambdified(*q,*vq,*lamq,*lamv)
    
    def new_control_H_v_calc(self,q,lam,vy,u,params):
        v_y = sympy.flatten(self.vq) + sympy.flatten(self.vlam)
        # v_y_as_p_y = sympy.flatten(self.v_y_as_p_y())
        p_y = self.p_y_legendre
        control_L = self.control_L#self.control_lagrangian(self.q,self.lamq,self.vq,self.vlam,self.u,self.parameters)
        control_H = (p_y.T @ sympy.Matrix(v_y)) - control_L
        control_H = control_H.subs([[tmp1,tmp2] for tmp1,tmp2 in zip(self.q,q)]).subs([[tmp1,tmp2] for tmp1,tmp2 in zip(self.lamq,lam)])
        control_H = control_H.subs([[tmp1,tmp2] for tmp1,tmp2 in zip(v_y,vy)])
        control_H = control_H.subs([[tmp1,tmp2] for tmp1,tmp2 in zip(self.u,u)])
        return np.array(sympy.flatten(control_H),float)[0]

    def new_control_H(self):
        v_y = sympy.flatten(self.vq) + sympy.flatten(self.vlam)
        v_y_as_p_y = sympy.flatten(self.v_y_as_p_y())
        p_y = sympy.Matrix(sympy.flatten(self.pq) + sympy.flatten(self.plam))
        control_L = self.control_lagrangian(self.q,self.lamq,self.vq,self.vlam,self.u,self.parameters)
        control_H = (p_y.T @ sympy.Matrix(v_y)) - control_L
        control_H = control_H.subs([[tmp1,tmp2] for tmp1,tmp2 in zip(v_y,v_y_as_p_y)])
        # control_H = self.pq.transpose()@ self.plam - self.lamq.transpose()@(self.f_vec(self.q,self.plam,self.parameters)+self.rho_vec(self.q,self.plam,self.parameters)@self.u)
        # control_H += self.running_cost(self.q,self.u,self.parameters)
        varlist = sympy.flatten(self.q) + sympy.flatten(self.lamq)+ sympy.flatten(self.pq) + sympy.flatten(self.plam)+sympy.flatten(self.u)
        self.new_control_H_lambdified  = sympy.lambdify(varlist,control_H)
        return control_H
    def new_control_H_eval(self,q,lamq,pq,plam,u,params):
        control_H = self.new_control_H_func
        
        varlist = sympy.flatten(self.q) + sympy.flatten(self.lamq)+ sympy.flatten(self.pq) + sympy.flatten(self.plam)+sympy.flatten(self.u)

        return self.new_control_H_lambdified(*q,*lamq,*pq,*plam,*u)[0,0]
    def u_lambda_from_newH(self):
        u_vec = self.u
        u_lam_rel = sympy.flatten(sympy.derive_by_array(self.new_control_H(),u_vec))
        sol = sympy.solve(sympy.flatten(u_lam_rel),u_vec)
        if len(sol.keys()) == 0:
            print("did not find an explicit u solution")
        self.u_expr_from_new = sympy.flatten(u_vec.subs([[tmp,sol[tmp]] for tmp in sympy.flatten(u_vec)]))
        all_var = sympy.flatten(self.q) + sympy.flatten(self.lamq) + sympy.flatten(self.pq) + sympy.flatten(self.plam)
        self.u_expr_from_new_lambdified = sympy.lambdify(all_var,self.u_expr_from_new)
        return sympy.flatten(u_vec.subs([[tmp,sol[tmp]] for tmp in sympy.flatten(u_vec)]))
                
    def u_eval_from_new(self,q,lam,pq,plam,params):
        # u_expr = self.u_lambda_from_newH()
        # all_var = sympy.flatten(self.q) + sympy.flatten(self.lamq) + sympy.flatten(self.pq) + sympy.flatten(self.plam)
        # u_lambdified = sympy.lambdify(all_var,u_expr)
        return self.u_expr_from_new_lambdified(*q,*lam,*pq,*plam)
        
    def u_eval_from_new_velocity(self):
        control_L = self.control_L
        u_var = self.u
        u_rel = sympy.derive_by_array(control_L,u_var)
        u_sol= sympy.solve(sympy.flatten(u_rel),u_var)
        u_sol_vec = sympy.Matrix([u_sol[tmp] for tmp in u_var])
        all_var = sympy.flatten(self.q) + sympy.flatten(self.lamq) + sympy.flatten(self.vq) + sympy.flatten(self.vlam)
        self.u_expr_from_new_lambdified = sympy.lambdify(all_var,u_sol_vec)

        return u_sol_vec


class discrete_standard_direct_eq_generator_forced_L_double_pendulum:
    def __init__(self,PMP_low_thrust_eq ):
        self.cont_equations = PMP_low_thrust_eq
        self.N = self.cont_equations.parameters["N"]
        self.h = self.cont_equations.parameters["h"]
        mu_str_prefac,mu_str = "mu_",""
        for tmp in self.cont_equations.q:
            mu_str += mu_str_prefac+str(tmp) + ","
        self.mu =  sympy.Matrix(sympy.symbols(mu_str))
        nu_str_prefac,nu_str = "nu_",""
        for tmp in self.cont_equations.vq:
            nu_str += nu_str_prefac+str(tmp) + ","
        self.nu =  sympy.Matrix(sympy.symbols(nu_str))

        self.all_vars =  [self.cont_equations.q,self.cont_equations.vq,self.cont_equations.lamq,self.cont_equations.lamv,self.cont_equations.u]
        self.all_vars_new_approach =  [self.cont_equations.q,self.cont_equations.vq,self.cont_equations.lamq,self.cont_equations.vlam,self.cont_equations.u]
        # self.k_eqs_description_q= eq_discretizer_midpoint(self.cont_equations.state_right_hand_side()[0],self.all_vars,0,self.cont_equations.parameters)
        # self.k_eqs_description_vq= eq_discretizer_midpoint(self.cont_equations.state_right_hand_side()[1],self.all_vars,1,self.cont_equations.parameters)
        self.k_eqs_description_q= eq_discretizer_midpoint(self.cont_equations.state_right_hand_side()[0],self.all_vars,0,self.cont_equations.parameters)
        self.k_eqs_description_vq= eq_discretizer_midpoint(self.cont_equations.state_right_hand_side()[1],self.all_vars,1,self.cont_equations.parameters)
        self.control_L_k = discrete_control_Lagrangian(self.cont_equations.control_lagrangian,self.all_vars_new_approach,self.cont_equations.parameters)
        q_d,vq_d,lamq_d,lamv_d,U_d,U1_d,U2_d = [],[],[],[],[],[],[]
        for i in range(self.N+1):
            q_d.append(sympy.Matrix(sympy.symbols([str(x) + "_"+str(i) for x in self.all_vars[0]])))
            vq_d.append(sympy.Matrix(sympy.symbols([str(x) + "_"+str(i) for x in self.all_vars[1]])))
            lamq_d.append(sympy.Matrix(sympy.symbols([str(x) + "_"+str(i) for x in self.all_vars[2]])))
            lamv_d.append(sympy.Matrix(sympy.symbols([str(x) + "_"+str(i) for x in self.all_vars[3]])))
            U_d.append(sympy.Matrix(sympy.symbols([str(x) + "_"+str(i) for x in self.all_vars[4]])))
            U1_d.append(sympy.Matrix(sympy.symbols([str(x) + "_1d_"+str(i) for x in self.all_vars[4]])))
            U2_d.append(sympy.Matrix(sympy.symbols([str(x) + "_2d_"+str(i) for x in self.all_vars[4]])))
            # vlam_d.append(sympy.Matrix(sympy.symbols([str(x) + "_"+str(i) for x in self.all_vars_new_approach[3]])))

        self.all_d = [q_d,vq_d,lamq_d, lamv_d,U1_d,U2_d]
        self.all_d_new_approach = [q_d,lamq_d,U1_d,U2_d]
    def __k_to_i_eval(self,eq_to_subs, k_sub_vector,i_sub_vector):
        return eq_to_subs.subs([(tmp1,tmp2) for tmp1,tmp2 in zip(k_sub_vector,i_sub_vector)])

    def generate_discrete_differential_eqs(self):
        '''generate here all N algebraic equations to solve for from the differential equation chosen'''
        all_eqs_q = []
        all_eqs_vq = []

        q_k_eqs = self.k_eqs_description_q["state_eqs"]
        vq_k_eqs = self.k_eqs_description_vq["state_eqs"]
        all_vars_k = self.k_eqs_description_q["vars_k"]
        all_vars_k1 = self.k_eqs_description_q["vars_k1"]
        for i in range(self.N):
            all_eqs_q.append(q_k_eqs)
            for j in range(len(all_vars_k)):
                all_eqs_q[-1] = self.__k_to_i_eval(all_eqs_q[-1],all_vars_k[j],self.all_d[j][i])
                all_eqs_q[-1] = self.__k_to_i_eval(all_eqs_q[-1],all_vars_k1[j],self.all_d[j][i+1])
            all_eqs_vq.append(vq_k_eqs)
            for j in range(len(all_vars_k)):
                all_eqs_vq[-1] = self.__k_to_i_eval(all_eqs_vq[-1],all_vars_k[j],self.all_d[j][i])
                all_eqs_vq[-1] = self.__k_to_i_eval(all_eqs_vq[-1],all_vars_k1[j],self.all_d[j][i+1])


        return all_eqs_q,all_eqs_vq
    def generate_initial_constraint(self):
        initial_q = self.all_d[0][0] - self.cont_equations.parameters["q0"]
        initial_vq = self.all_d[1][0] - self.cont_equations.parameters["dq0"]
        return initial_q, initial_vq
    def generate_discrete_running_cost(self):
        all_running_cost = []
        gamma = self.cont_equations.parameters["gamma"]
        alpha = self.cont_equations.parameters["alpha"]
        for i in range(self.N):
            mean_q_gamma = ((1-gamma)*self.all_d[0][i+1]+gamma*self.all_d[0][i])
            mean_q_1_gamma = (gamma*self.all_d[0][i+1]+(1-gamma)*self.all_d[0][i])
            u_1_k = self.all_d[-2][i]
            u_2_k = self.all_d[-1][i]
            # mean_u = self.all_d[-1][i]#(self.all_d[-1][i+1]+ self.all_d[-1][i])/2
            running_cost = self.h*alpha*self.cont_equations.running_cost(mean_q_gamma,u_1_k,self.cont_equations.parameters )
            running_cost += self.h*(1-alpha)*self.cont_equations.running_cost(mean_q_1_gamma,u_2_k,self.cont_equations.parameters )
            all_running_cost.append(running_cost)
            # all_running_cost.append(self.h*self.cont_equations.running_cost(mean_q_1_gamma,u_2_k,self.cont_equations.parameters ))
        return all_running_cost  
    def generate_terminal_cost(self):
        q_N = self.all_d[0][-1]
        vq_N = self.all_d[1][-1]
        mayer_eval = self.cont_equations.mayer_term(q_N, vq_N,self.cont_equations.parameters)
        # print(mayer_eval)
        return mayer_eval    
    def generate_discrete_objective(self):
        '''the objective and the state constraints separately'''
        terminal_cost = self.generate_terminal_cost()
        running_cost = self.generate_discrete_running_cost()
        initial_constraints = self.generate_initial_constraint()
        state_equations = self.generate_discrete_differential_eqs()   

        discrete_objective_function = terminal_cost 
        for tmp in running_cost:
            discrete_objective_function += tmp
        # discrete_objective_function += self.mu.transpose()@initial_constraints[0]    
        # discrete_objective_function += self.nu.transpose()@initial_constraints[1]    
        # #q equations
        # for tmp1, tmp2 in zip(state_equations[0],self.all_d[2]):
        #     discrete_objective_function += tmp2.transpose()@ tmp1
        # #vq equations    
        # for tmp1, tmp2 in zip(state_equations[1],self.all_d[3]):
        #     discrete_objective_function += tmp2.transpose()@ tmp1

        opt_variables = []#+list(self.mu)
        # opt_variables+= list(self.nu)
        for tmp in self.all_d[:2]:
            for t in tmp:
                opt_variables+=list(t)
        for t in self.all_d[-1]:
                opt_variables+=list(t)        
        return {"discrete_objective":discrete_objective_function, "discrete_state_constraints":state_equations,"initial_constraints":initial_constraints, "discrete_variables_flattened_list":opt_variables}

    def generate_discrete_objective_KKT(self):
        '''give here alternatively the objective that can be varied to receive the KKT'''
        terminal_cost = self.generate_terminal_cost()
        running_cost = self.generate_discrete_running_cost()
        initial_constraints = self.generate_initial_constraint()
        state_equations = self.generate_discrete_differential_eqs()   

        discrete_objective_function = terminal_cost 
        for tmp in running_cost:
            discrete_objective_function += tmp
        discrete_objective_function += self.mu.transpose()@initial_constraints[0]    
        discrete_objective_function += self.nu.transpose()@initial_constraints[1]    
        #q equations
        for tmp1, tmp2 in zip(state_equations[0],self.all_d[2]):
            discrete_objective_function += tmp2.transpose()@ tmp1
        #vq equations    
        for tmp1, tmp2 in zip(state_equations[1],self.all_d[3]):
            discrete_objective_function += tmp2.transpose()@ tmp1

        opt_variables = []+list(self.mu)
        opt_variables+= list(self.nu)
        for tmp in self.all_d[:-1]:
            for t in tmp:
                opt_variables+=list(t)
        for t in self.all_d[-1]:
                opt_variables+=list(t)        
        return {"discrete_objective":discrete_objective_function, "discrete_variables_flattened_list":opt_variables}  
    def calc_KKT(self,params_np):
        objective_and_vars = self.generate_discrete_objective_KKT()
        objective = objective_and_vars["discrete_objective"].subs([["alpha",params_np["alpha"]],["gamma",params_np["gamma"]],["h",params_np["h"]]])
        mu_eq = sympy.flatten(sympy.derive_by_array(objective,self.mu))
        nu_eq = sympy.flatten(sympy.derive_by_array(objective,self.nu))
        rest_eqs = []
        for tmp in self.all_d:
            for tmp1 in tmp:
                rest_eqs += sympy.flatten(sympy.derive_by_array(objective,tmp1)) #iterate through all N+1 elements
        return sympy.Array(mu_eq+ nu_eq+ rest_eqs), objective_and_vars["discrete_variables_flattened_list"]

    def vqk1_p(self):
        p_y_k1_p = self.p_y_k1p()
        dim_q = len(self.cont_equations.q)
        p_mech_k1_p = sympy.Matrix(p_y_k1_p[dim_q:])
        p_mech_cont = sympy.Matrix(sympy.derive_by_array(self.cont_equations.control_lagrangian(self.cont_equations.q,self.cont_equations.lamq,self.cont_equations.vq,self.cont_equations.vlam,self.cont_equations.u,self.cont_equations.parameters)[0],self.cont_equations.vlam))
        vq_as_func_disc_p = sympy.solve(p_mech_cont  - self.cont_equations.pq, self.cont_equations.vq)
        vq_discrete = sympy.Matrix([vq_as_func_disc_p[tmp] for tmp in self.cont_equations.vq])
        vq_discrete = vq_discrete.subs([[tmp1,tmp2] for tmp1,tmp2 in zip(self.cont_equations.q,self.control_L_k["vars_k1"][0])])
        return  vq_discrete.subs([[tmp1,tmp2] for tmp1,tmp2 in zip(self.cont_equations.pq,p_mech_k1_p)])
     
        # return sympy.derive_by_array(sympy.flatten(self.control_L_k["discrete_control_L"])[0], self.control_L_k["vars_k1"][1])
    def vqk_m(self):
        p_y_k_m = self.p_y_km()
        dim_q = len(self.cont_equations.q)
        p_mech_k_m = sympy.Matrix(p_y_k_m[dim_q:])
        p_mech_cont = sympy.Matrix(sympy.derive_by_array(self.cont_equations.control_lagrangian(self.cont_equations.q,self.cont_equations.lamq,self.cont_equations.vq,self.cont_equations.vlam,self.cont_equations.u,self.cont_equations.parameters)[0],self.cont_equations.vlam))
        vq_as_func_disc_p = sympy.solve(p_mech_cont  - self.cont_equations.pq, self.cont_equations.vq)
        vq_discrete = sympy.Matrix([vq_as_func_disc_p[tmp] for tmp in self.cont_equations.vq]).subs([[tmp1,tmp2] for tmp1,tmp2 in zip(self.cont_equations.q,self.control_L_k["vars_k"][0])])
        return vq_discrete.subs([[tmp1,tmp2] for tmp1,tmp2 in zip(self.cont_equations.pq,p_mech_k_m)])

        # return -sympy.derive_by_array(sympy.flatten(self.control_L_k["discrete_control_L"])[0], self.control_L_k["vars_k"][1])  
    def p_y_k1p(self):
            p_lam_k1=sympy.derive_by_array(sympy.flatten(self.control_L_k["discrete_control_L"])[0], self.control_L_k["vars_k1"][1])
            p_q_k1 =sympy.derive_by_array(sympy.flatten(self.control_L_k["discrete_control_L"])[0], self.control_L_k["vars_k1"][0])
            return sympy.Matrix(sympy.flatten(p_q_k1)+ sympy.flatten(p_lam_k1))
    def p_y_km(self):
            p_lam_k=-sympy.derive_by_array(sympy.flatten(self.control_L_k["discrete_control_L"])[0], self.control_L_k["vars_k"][1])
            p_q_k =-sympy.derive_by_array(sympy.flatten(self.control_L_k["discrete_control_L"])[0], self.control_L_k["vars_k"][0])
            return sympy.Matrix(sympy.flatten(p_q_k)+ sympy.flatten(p_lam_k))
    def p_v_d_from_y_d(self,q_d,lam_d,U_1_d,U_2_d,np_params):
        all_vars_to_sub = [q_d,lam_d,U_1_d,U_2_d]
        p_y_d = []
        p_y_k1p = self.p_y_k1p().subs("gamma",np_params["gamma"]).subs("alpha",np_params["alpha"]).subs("h",np_params["h"])
        p_y_km = self.p_y_km().subs("gamma",np_params["gamma"]).subs("alpha",np_params["alpha"]).subs("h",np_params["h"])
        k1_vars = self.control_L_k["vars_k1"]
        k_vars = self.control_L_k["vars_k"]
        for i in range(len(q_d)-1):
            p_y_i = p_y_km
            for j in range(len(k_vars)):
                p_y_i = self.__k_to_i_eval(p_y_i,k_vars[j],all_vars_to_sub[j][i])
                p_y_i = self.__k_to_i_eval(p_y_i,k1_vars[j],all_vars_to_sub[j][i+1])
            p_y_d.append(p_y_i)
        for j in range(len(k_vars)):
                p_y_k1p = self.__k_to_i_eval(p_y_k1p,k_vars[j],all_vars_to_sub[j][-2])
                p_y_k1p = self.__k_to_i_eval(p_y_k1p,k1_vars[j],all_vars_to_sub[j][-1])  
        p_y_d.append(np.array(sympy.flatten(p_y_k1p)))



        return p_y_d          
    def u_d_from_y_d(self,q_d,v_q_d,lam_d,v_lam_d,np_params):
        all_vars_to_sub = [q_d,v_q_d,lam_d,v_lam_d]
        u_d = []
        u_func = self.cont_equations.u_calc()
        all_vars = self.all_vars_new_approach[:-1]
        for i in range(self.cont_equations.parameters["N"]+1):
            u_d.append(sympy.Matrix(u_func))
            for tmp,tmpsub in zip(all_vars,all_vars_to_sub):
                u_d[-1] = u_d[-1].subs([[val,subval] for val,subval in zip(tmp,tmpsub[i]) ])
        return u_d        
    def generate_new_control_objective(self):
        '''generate here the new control objective'''
        k1_vars = self.control_L_k["vars_k1"]
        k_vars = self.control_L_k["vars_k"]

        p_y_k_m = self.p_y_km()
        p_y_k1_p = self.p_y_k1p()
        dim_q = len(self.cont_equations.q)
        p_mech_k1_p = sympy.Matrix(p_y_k1_p[dim_q:])
        p_mech_k_m = sympy.Matrix(p_y_k_m[dim_q:])

        vq_k1_p = self.vqk1_p()
        vq_k_m = self.vqk_m()

        all_d = self.all_d_new_approach
        v0 = vq_k_m #.subs([[tmp1,tmp2] for tmp1,tmp2 in zip(self.cont_equations.q,all_d[0][0])])
        q_d,lam_d,U1_d,U2_d = self.all_d_new_approach
        pN = sympy.Matrix(p_mech_k1_p)
        p0 = sympy.Matrix(p_mech_k_m)
        for i in range(len(self.all_d_new_approach)):
            v0 = self.__k_to_i_eval(v0,k1_vars[i],all_d[i][1])
            v0 = self.__k_to_i_eval(v0,k_vars[i],all_d[i][0])
            p0 = self.__k_to_i_eval(p0,k1_vars[i],all_d[i][1])
            p0 = self.__k_to_i_eval(p0,k_vars[i],all_d[i][0])
        vN = vq_k1_p
        for i in range(len(self.all_d_new_approach)):
            vN = self.__k_to_i_eval(vN,k1_vars[i],all_d[i][-1])
            vN = self.__k_to_i_eval(vN,k_vars[i],all_d[i][-2])
            pN = self.__k_to_i_eval(pN,k1_vars[i],all_d[i][-1])
            pN = self.__k_to_i_eval(pN,k_vars[i],all_d[i][-2])

        #mayer term    
        discrete_objective_function = sympy.flatten(self.cont_equations.mayer_term(q_d[-1],vN,self.cont_equations.parameters))[0]
        # initial constraints
        discrete_objective_function += sympy.flatten(self.mu.transpose()@(q_d[0]- self.cont_equations.parameters["q0"])    )[0]
        discrete_objective_function += sympy.flatten(self.nu.transpose()@(v0 - self.cont_equations.parameters["dq0"]))[0]
        #partial integration boundary terms
        discrete_objective_function -= sympy.flatten(lam_d[0].transpose()@p0)[0]
       
        discrete_objective_function += sympy.flatten(lam_d[-1].transpose()@pN)[0]
        #Lagrange terms
        # print( sympy.flatten(self.control_L_k["discrete_control_L"]))
        for j in range(self.cont_equations.parameters["N"]):
            discrete_control_L_eval = self.control_L_k["discrete_control_L"]
            for i in range(len(self.all_d_new_approach)):
                discrete_control_L_eval = self.__k_to_i_eval(discrete_control_L_eval,k1_vars[i],all_d[i][j+1])
                discrete_control_L_eval = self.__k_to_i_eval(discrete_control_L_eval,k_vars[i],all_d[i][j])
            discrete_objective_function -= sympy.flatten(discrete_control_L_eval)[0]

        #terminal variation term for lambda_N definition
        D2phi =  sympy.Matrix(sympy.flatten(sympy.diff(self.cont_equations.mayer_term(self.all_vars[0],self.all_vars[1],self.cont_equations.parameters),self.all_vars[1])))
        D2phi =  D2phi.subs([[tmp1,tmp2] for tmp1,tmp2 in zip(self.all_vars[0],all_d[0][-1])]).subs(([[tmp1,tmp2] for tmp1,tmp2 in zip(self.all_vars[1],vN)]))
        opt_variables = []+list(self.mu)
        opt_variables+= list(self.nu)
        for tmp in all_d:
            for t in tmp:
                opt_variables+=list(t)



        return {"discrete_objective":discrete_objective_function, "discrete_variables_flattened_list":opt_variables,"D2phi":D2phi}    
    def evaluate_state_eqs(self,qkm1,qk,params):
        objective_and_vars = self.generate_new_control_objective()
        objective = objective_and_vars["discrete_objective"]
        state_eqs = sympy.derive_by_array(objective,self.all_d_new_approach[1][3]) #just get one of them
        for tmp1, tmp2 in zip (self.all_d_new_approach,self.control_L_k["vars_k"]):
            state_eqs.subs([[tmp11,tmp22] for tmp11,tmp22 in zip(sympy.flatten(tmp1[3]),sympy.flatten(tmp2))])
        all_km1=[]
        simplified_expr = state_eqs
        for tmp in self.control_L_k["vars_k"]:
            all_km1.append([sympy.Symbol(str(tmpobj)+'_m_1') for tmpobj in tmp])
        for tmp1, tmp2 in zip(self.all_d_new_approach,self.control_L_k["vars_k"]):
            simplified_expr=simplified_expr.subs([[tmp11, tmp22] for tmp11, tmp22 in zip(tmp1[3],tmp2)])
        for tmp1, tmp2 in zip(self.all_d_new_approach,self.control_L_k["vars_k1"]):
            simplified_expr=simplified_expr.subs([[tmp11, tmp22] for tmp11, tmp22 in zip(tmp1[4],tmp2)])
        for tmp1, tmp2 in zip(self.all_d_new_approach,all_km1):
            simplified_expr=simplified_expr.subs([[tmp11, tmp22] for tmp11, tmp22 in zip(tmp1[2],tmp2)])

        simplified_expr = simplified_expr.subs([["alpha",params["alpha"]],["gamma",params["gamma"]],["h",params["h"]]])
        # solve_for_k1 = sympy.solve(simplified_expr,sympy.Matrix(all_km1))
        dim_q = len(self.cont_equations.q)
        fu = simplified_expr.subs([[tmp1,tmp2] for tmp1,tmp2 in zip(sympy.flatten(sympy.Matrix(all_km1[:-3])),qkm1)]).subs([[tmp1,tmp2] for tmp1,tmp2 in zip(sympy.flatten(sympy.Matrix(self.control_L_k["vars_k"][:-3])),qk)])
        fu = fu.subs([[tmp1,tmp2] for tmp1,tmp2 in zip(sympy.flatten(sympy.Matrix(all_km1[dim_q:])),[0,0])]).subs([[tmp1,tmp2] for tmp1,tmp2 in zip(sympy.flatten(sympy.Matrix(self.control_L_k["vars_k"][dim_q:])),[0,0])]).subs([[tmp1,tmp2] for tmp1,tmp2 in zip(sympy.flatten(sympy.Matrix(self.control_L_k["vars_k1"][dim_q:])),[0,0])])
        qk1_getter = sympy.lambdify(sympy.flatten(sympy.Matrix(self.control_L_k["vars_k1"][:-3])),fu)
        qk1_getter_lambda = lambda x: qk1_getter(*x)[:,0]
        qk1_res = opt.root(qk1_getter_lambda,x0=qk)

        return qk1_res.x
    
    def calc_KKT_new(self,params_np):
        objective_and_vars = self.generate_new_control_objective()
        objective = objective_and_vars["discrete_objective"].subs([["alpha",params_np["alpha"]],["gamma",params_np["gamma"]],["h",params_np["h"]]])
        mu_eq = sympy.flatten(sympy.derive_by_array(objective,self.mu))
        nu_eq = sympy.flatten(sympy.derive_by_array(objective,self.nu))
        rest_eqs = []
        for tmp in self.all_d_new_approach:
            for tmp1 in tmp:
                rest_eqs += sympy.flatten(sympy.derive_by_array(objective,tmp1)) #iterate through all N+1 elements

        k1_vars = self.control_L_k["vars_k1"]
        k_vars = self.control_L_k["vars_k"]
        vq_k1_p = self.vqk1_p()
        vq_k_m = self.vqk_m()
        all_d = self.all_d_new_approach
        v0 = vq_k_m #.subs([[tmp1,tmp2] f
        q_d,lam_d,U1_d,U2_d = self.all_d_new_approach
        for i in range(len(self.all_d_new_approach)):
            v0 = self.__k_to_i_eval(v0,k1_vars[i],all_d[i][1])
            v0 = self.__k_to_i_eval(v0,k_vars[i],all_d[i][0])
        vN = vq_k1_p #.subs([[tmp1,tmp2] for tmp1,tmp2 in zip(self.cont_equations.q,all_d[0][-1])])
 
        for i in range(len(self.all_d_new_approach)):
            vN = self.__k_to_i_eval(vN,k1_vars[i],all_d[i][-1])
            vN = self.__k_to_i_eval(vN,k_vars[i],all_d[i][-2])

        dlam0term_element = 0* self.cont_equations.vq
        dlamTterm_element = 0* self.cont_equations.vq
        p_mech_cont = sympy.Matrix(sympy.derive_by_array(self.cont_equations.control_lagrangian(self.cont_equations.q,self.cont_equations.lamq,self.cont_equations.vq,self.cont_equations.vlam,self.cont_equations.u,self.cont_equations.parameters)[0],self.cont_equations.vlam))

        for tmp1,tmp2 in zip(self.cont_equations.vq,self.all_d_new_approach[1][0]):
            dlam0term_element += sympy.Matrix(sympy.derive_by_array(p_mech_cont,tmp1))*tmp2
        dlam0term_element = dlam0term_element.subs([[tmpa,tmpb] for tmpa,tmpb in zip(self.cont_equations.q,self.all_d_new_approach[0][0])])   
        dlam0term_element = dlam0term_element.subs([[tmpa,tmpb] for tmpa,tmpb in zip(self.cont_equations.vq,v0)])   
        for tmp1,tmp2 in zip(self.cont_equations.vq,self.all_d_new_approach[1][-1]):
            dlamTterm_element += sympy.Matrix(sympy.derive_by_array(p_mech_cont,tmp1))*tmp2
        dlamTterm_element = dlamTterm_element.subs([[tmpa,tmpb] for tmpa,tmpb in zip(self.cont_equations.q,self.all_d_new_approach[0][-1])])   
        dlamTterm_element = dlamTterm_element.subs([[tmpa,tmpb] for tmpa,tmpb in zip(self.cont_equations.vq,vN )])   

        dlam0term = sympy.flatten(self.nu - dlam0term_element)
        dlamTterm = sympy.flatten(objective_and_vars["D2phi"] + dlamTterm_element)





        #by construction, delta lambda_0, delta lambda_N are undefined.
        #   we use therefore their definition D_2 phi = -lambda_N^T
        # and nu = lambda_0
        # in place of the 0's
        N_val = self.cont_equations.parameters["N"]
        dim_q = self.cont_equations.parameters["dim_q"]
        # dlam0term = sympy.flatten(self.nu - self.all_d_new_approach[1][0])
        # dlamTterm = sympy.flatten(objective_and_vars["D2phi"]+ self.all_d_new_approach[1][-1])
        #the boundary lambda terms definitions are inserted here
        for i in range(len(dlam0term)):
            rest_eqs[(1+N_val)*dim_q+i] = dlam0term[i]
            rest_eqs[2*(1+N_val)*dim_q-len(dlamTterm)+i]= dlamTterm[i]

        return sympy.Array(mu_eq+ nu_eq+ rest_eqs), objective_and_vars["discrete_variables_flattened_list"]
    