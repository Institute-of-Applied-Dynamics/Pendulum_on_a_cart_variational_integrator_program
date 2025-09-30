import sympy

############################################################################
############################################################################
################### Pendulum on a cart functions
############################################################################
############################################################################



def pendulum_cart_lagrangian(q,vq,params):
    m1,m2 = params["m1"],params["m2"]
    l,g,I_param = params["l"],params["G"],params["I"]
    x,theta   = q
    vx,vtheta = vq
    Lfunc = (m1+m2)*vx**2/2 + l*m2*vx*vtheta*sympy.cos(theta)/2 + l**2*m2*vtheta**2/8 + I_param*vtheta**2/2 + m2*g*l*sympy.cos(theta)/2
    return Lfunc

def f_L(q,vq,u,params):
    uvec,=u 
    return sympy.Matrix([uvec,0])

def running_cost_pendulum(q,u,params):
    A_u = params["A_u"]
    return A_u *u.transpose()@u/2

def mayer_term_pendulum(q,v,params):
    x,phi = q
    vx,vphi = v
    xT,phiT = params["qT"]
    vxT,vphiT = params["dqT"]
    Phi = params["Aq"]*((x-xT)**2 + (phi-phiT)**2)
    Phi+= params["Adq"]*((vx-vxT)**2 + (vphi-vphiT)**2)
    return sympy.Matrix([Phi])

def conserved_I_control_pendulum(control_L, q,lam,vq,vlam,u,params):
    '''give here the conserved quantity for the control Lagrangian
    for pendulum on a cart that is p_x = dL_c/dv_x'''
    I_ocp = sympy.derive_by_array(control_L,vq[0])
    return I_ocp

def g_mat_pendulum(q,params):
    return sympy.Matrix([1])


############################################################################
############################################################################
################### Two-Pendulum on a cart functions
############################################################################
############################################################################



def two_pendulum_cart_lagrangian(q,vq,params):
    x,theta1,theta2   = q
    vx,vtheta1,vtheta2 = vq
    l1,l2 = sympy.symbols("l1,l2")
    m0,m1,m2 ,g= sympy.symbols("m0,m1,m2,g")
    vx0 = sympy.Matrix([vx,0])
    vt1 = sympy.Matrix([vx + l1/2*sympy.cos(theta1)*vtheta1 ,
                        l1/2*sympy.sin(theta1)*vtheta1 ])
    vt2 = sympy.Matrix([vx + l1*sympy.cos(theta1)*vtheta1 + l2/2*sympy.cos(theta1+theta2)*(vtheta1+vtheta2),
                        l1*sympy.sin(theta1)*vtheta1 + l2/2*sympy.sin(theta1+theta2)*(vtheta1+vtheta2)])
    Ekin = sympy.simplify(sympy.expand(m0*(vx0.T@vx0)/2 + m1*(vt1.T@vt1)/2 + m2*(vt2.T@vt2)/2))
    m0param,m1param,m2param = params["m0"],params["m1"],params["m2"]
    l1param,l2param,g,I1_param,I2_param = params["l1"], params["l2"],params["G"],params["I1"],params["I2"]
    Lfunc = Ekin[0] 
    Lfunc += I1_param* vtheta1**2/2 + I2_param*vtheta2**2/2
    Lfunc += (m1/2+m2)*l1*g*sympy.cos(theta1) + m2/2*l2*g*sympy.cos(theta1+theta2) 
    return Lfunc.subs([[l1,l1param],[l2,l2param],[m0,m0param],[m1,m1param],[m2,m2param]])



def f_L_two_pendulum(q,vq,u,params):
    u0,u1,u2=u 
    return sympy.Matrix([u0,u1,u2])

def running_cost_two_pendulum(q,u,params):
    A_u = params["A_u"]
    return A_u *u.transpose()@u/2

def mayer_term_two_pendulum(q,v,params):
    q_qT = q - params["qT"]  
    vq_vqT = v - params["dqT"]  

    Phi = params["Aq"]*(q_qT.T @ q_qT)
    Phi+= params["Adq"]*(vq_vqT.T @vq_vqT)
    return Phi 

def conserved_I_control_two_pendulum(control_L, q,lam,vq,vlam,u,params):
    '''give here the conserved quantity for the control Lagrangian
    for two-pendulum on a cart that is p_x = dL_c/dv_x'''
    I_ocp = sympy.derive_by_array(control_L,vq[0])
    return I_ocp

def g_mat_two_pendulum(q,params):
    return sympy.Matrix(sympy.eye(len(params["dim_u"])))
