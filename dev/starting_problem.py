import numpy as np
from scipy.optimize import brentq
import jaxprop as jxp

jxp.set_plot_options(grid=False)

fluid_name = "R245fa"
fluid = jxp.Fluid(fluid_name)
P_0 = 0.1 * fluid.critical_point.p
T_0 = 0.95 * fluid.critical_point.T
state_0 = fluid.get_state(jxp.PT_INPUTS, P_0, T_0)
h_0 = state_0.h
s_0 = state_0.s

def residual(h):
    state = fluid.get_state(jxp.HmassSmass_INPUTS, h, s_0)
    vel = np.sqrt(2 * (h_0 - h))
    return vel / state.speed_of_sound - 1

v_s = brentq(lambda v: v - fluid.get_state(jxp.HmassSmass_INPUTS, h_0 - 0.5*v**2, s_0).a, 1, 2*state_0.a)
h = h_0 - 0.5*v_s**2
sonic_state = fluid.get_state(jxp.HmassSmass_INPUTS, h, s_0)
a_star = sonic_state.a
rho_star = sonic_state.rho

state_min = fluid.get_state(jxp.SmassT_INPUTS, s_0, 175)
M_star_max = 0.999*np.sqrt(2*(h_0 - state_min.h))/a_star

def find_M_star(M_target):
    def residual_2(M_star):
        v = M_star * a_star
        h_s = h_0 - 0.5*v**2
        state = fluid.get_state(jxp.HmassSmass_INPUTS, h_s, s_0)
        a = state.speed_of_sound
        M_new = v / a
    
        return M_new - M_target
    
    return residual_2

M_1 = 2
M_star = brentq(find_M_star(M_1), 0, M_star_max, xtol=1e-6)
v_1 = M_star*a_star
h_1 = h_0 - 0.5*v_1**2

state_1 = fluid.get_state(jxp.HmassSmass_INPUTS, h_1, s_0)
p_1 = state_1.pressure
rho_1 = state_1.rho


def pressure_post_shock(v_2):
    rho_2 = rho_1*v_1/v_2
    h_2 = h_1 + 0.5*(v_1**2 - v_2**2)
    state_2 = fluid.get_state(jxp.DmassHmass_INPUTS, rho_2, h_2)
    p_2 = state_2.pressure
    return (p_1 + rho_1*v_1**2) - (p_2 + rho_2*v_2**2)

v_2 = brentq(pressure_post_shock, 0.3*a_star, a_star*(1 - 1e-6), xtol=1e-6)

C_r_is = (rho_1 * v_1) / (rho_star * a_star)

# C_r = P_0_2 / P_0_1 / C_r_is


def sonic_mass_flux(s):
    def residual_sonic(v):
        state = fluid.get_state(jxp.HmassSmass_INPUTS, h_0 - 0.5*v**2, s)
        return v - state.speed_of_sound

    v_sonic = brentq(residual_sonic, 0.5*a_star, 1.5*a_star, xtol=1e-9)
    state = fluid.get_state(jxp.HmassSmass_INPUTS, h_0 - 0.5*v_sonic**2, s)
    return state.rho * v_sonic


def contraction_ratios(M_1):
    M_star = brentq(find_M_star(M_1), 0, M_star_max, xtol=1e-9)
    v_1 = M_star*a_star
    state_1 = fluid.get_state(jxp.HmassSmass_INPUTS, h_0 - 0.5*v_1**2, s_0)
    rho_1, p_1 = state_1.rho, state_1.pressure

    def momentum(v_2):
        state_2 = fluid.get_state(jxp.DmassHmass_INPUTS, rho_1*v_1/v_2, h_0 - 0.5*v_2**2)
        return (p_1 + rho_1*v_1**2) - (state_2.pressure + rho_1*v_1*v_2)

    v_2 = brentq(momentum, 0.3*a_star, a_star*(1 - 1e-6), xtol=1e-9)
    s_2 = fluid.get_state(jxp.DmassHmass_INPUTS, rho_1*v_1/v_2, h_0 - 0.5*v_2**2).s

    C_r_is = (rho_1*v_1) / (rho_star*a_star)
    C_r = (rho_1*v_1) / sonic_mass_flux(s_2)
    return C_r_is, C_r


M_1_range = np.linspace(1.02, 3.5, 60)
ratios = np.array([contraction_ratios(M) for M in M_1_range])

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6,5))
# ax.plot(M_1_range, ratios[:, 0], "k:", label=r"$C_{r,is}$ isentropic")
ax.plot(M_1_range, ratios[:, 1], "k-", label=r"$C_r$ Kantrowitz (normal shock)")
ax.set_xlabel(r"Inlet Mach number $M_1$ [-]")
ax.set_ylabel(r"Area ratio $A_t/A_1$ [-]")
ax.set_xlim(1, 3.5)
ax.set_ylim(0.1, 1.01)
ax.legend()
plt.tight_layout()
plt.show()
