import numpy as np
from scipy.optimize import brentq
import CoolProp
import jaxprop as jxp


fluid_name = "R245fa"
CoolProp.CoolProp.set_config_bool(CoolProp.CoolProp.DONT_CHECK_PROPERTY_LIMITS, True)
T_MIN = 175

r = 0.5  # leading edge radius
s = r / 0.05  # pitch
beta_s = np.pi / 3  # stagger angle
x_O_adj = s * np.sin(beta_s)
y_O_adj = s * np.cos(beta_s)


def state_hs(h, s):
    """(h, s) state; above the EOS temperature limit the HS flash refuses, so solve T along the isentrope."""
    try:
        return fluid.get_state(jxp.HmassSmass_INPUTS, h, s)
    except (ValueError, RuntimeError):
        T = brentq(lambda T: fluid.get_state(jxp.SmassT_INPUTS, s, T).h - h, T_MIN, 2 * state_0.T)
        return fluid.get_state(jxp.SmassT_INPUTS, s, T)


def v_limit(s):
    state_min = fluid.get_state(jxp.SmassT_INPUTS, s, T_MIN)
    return 0.999 * np.sqrt(2 * (h_0 - state_min.h))


def sonic_state(s):
    v = brentq(lambda v: v - state_hs(h_0 - 0.5*v**2, s).a, 1, v_limit(s))
    return v, state_hs(h_0 - 0.5*v**2, s)


def find_M_star(M_target, a_star, s):
    def residual_2(M_star):
        v = M_star * a_star
        h_s = h_0 - 0.5*v**2
        state = state_hs(h_s, s)
        a = state.speed_of_sound
        M_new = v / a
        return M_new - M_target

    return brentq(residual_2, 0, v_limit(s) / a_star, xtol=1e-9)


def set_inlet(p_r, T_r):
    """Total inlet state from reduced pressure and temperature; sets the module-level inlet state."""
    global fluid, state_0, h_0, s_0, a_star, rho_star
    fluid = jxp.Fluid(fluid_name)
    critical = fluid.critical_point
    state_0 = fluid.get_state(jxp.PT_INPUTS, p_r * critical.p, T_r * critical.T)
    h_0, s_0 = state_0.h, state_0.s
    a_star, sonic_0 = sonic_state(s_0)
    rho_star = sonic_0.rho


def unique_incidence(M_1):
    v_1 = find_M_star(M_1, a_star, s_0) * a_star
    h_1 = h_0 - 0.5 * v_1**2
    state_1 = state_hs(h_1, s_0)
    p_1 = state_1.pressure
    rho_1 = state_1.rho

    def pressure_post_shock(u_2, u_1):
        rho_2 = rho_1*u_1/u_2
        h_2 = h_1 + 0.5*(u_1**2 - u_2**2)
        state_2 = fluid.get_state(jxp.DmassHmass_INPUTS, rho_2, h_2)
        p_2 = state_2.pressure
        return (p_1 + rho_1*u_1**2) - (p_2 + rho_2*u_2**2)

    def oblique_shock(beta):
        u_1 = v_1 * np.sin(beta)
        u_2 = brentq(pressure_post_shock, 0.1*u_1, 0.999*u_1, args=(u_1,))
        rho_2 = rho_1*u_1/u_2
        h_2 = h_1 + 0.5*(u_1**2 - u_2**2)
        state_2 = fluid.get_state(jxp.DmassHmass_INPUTS, rho_2, h_2)
        theta = beta - np.arctan(u_2 * np.tan(beta) / u_1)
        return u_1, u_2, state_2, theta

    # normal shock (beta = 90 deg)
    _, v_2, state_2, _ = oblique_shock(np.pi/2)
    s_2 = state_2.s

    # find theta and beta
    M_2_target = 1

    def residuals_theta(beta):
        u_1, u_2, state_2, theta = oblique_shock(beta)
        M_2n = u_2/state_2.a
        M_2 = M_2n / np.sin(beta - theta)
        return M_2 - M_2_target

    beta = brentq(residuals_theta, np.arcsin(1/M_1) + 1e-3, np.pi/2 - 1e-6, xtol=1e-6)
    u_1, u_2, state_beta, theta = oblique_shock(beta)

    # find geometrical parameters of the bow shock
    x_s_le = - r * np.sin(theta)
    y_s_le = r * np.cos(theta)

    a_2_star, sonic_2 = sonic_state(s_2)
    C_r = (rho_1 * v_1) / (sonic_2.rho * a_2_star)
    beta_M = np.sqrt(M_1**2 - 1)
    C_beta = beta_M * (beta_M * np.tan(beta) - np.sqrt(beta_M**2 * np.tan(beta)**2 - 1))
    y_s = y_s_le * (1 - C_r * np.cos(theta))**(-1)
    L = y_s_le * (y_s/y_s_le * (C_beta + np.tan(theta)) - np.tan(theta))
    standoff = r * np.sin(theta) + L

    # Find point E, which depends on the inlet flow angle and has to be coupled,
    # hence, to the solution of the unique incidence problem
    def residual_phi_1(x_E, details=False):

        if x_E < 0:
            phi_E = np.arcsin(np.abs(x_E) / r)
            y_E = r * np.cos(phi_E)
        else:
            phi_E = 0
            y_E = r

        # 2
        x_C = x_O_adj - standoff * np.cos(phi_E)
        y_C = y_O_adj - standoff * np.sin(phi_E)
        l_E = np.sqrt((y_C - y_E)**2 + (x_C - x_E)**2)
        alpha_E = np.arctan2(y_C - y_E, x_C - x_E) - phi_E
        M_E = 1 / np.sin(alpha_E)

        # 3
        M_E_star = find_M_star(M_E, a_2_star, s_2)
        v_E = M_E_star * a_2_star
        h_E = h_0 - 0.5 * v_E**2
        state_E = state_hs(h_E, s_2)
        rho_E = state_E.rho
        phi_1 = np.arcsin(rho_E * (v_E * np.sin(alpha_E)) * l_E / (rho_1 * v_1 * s))

        A_E = np.sqrt(M_E**2 - 1) / v_E
        A_1 = np.sqrt(M_1**2 - 1) / v_1
        A_bar = (A_1 + A_E) / 2
        beta_i_1 = phi_E + A_bar * (v_E - v_1)  # incidence angle
        phi_1_test = np.pi / 2 - beta_s - beta_i_1

        if details:
            return phi_1, beta_i_1
        return phi_1 - phi_1_test

    x_C = x_O_adj - standoff
    x_grid = np.concatenate([np.linspace(x_s_le, 0, 60, endpoint=False), np.linspace(0, x_C, 30)])
    values = []
    for x in x_grid:
        try:
            values.append(residual_phi_1(x))
        except (ValueError, RuntimeError):
            values.append(np.nan)
    values = np.array(values)
    k = next(i for i in range(len(x_grid) - 1)
             if np.isfinite(values[i]) and np.isfinite(values[i + 1]) and values[i] * values[i + 1] < 0)
    x_E = brentq(residual_phi_1, x_grid[k], x_grid[k + 1], xtol=1e-9)
    phi_1, beta_i_1 = residual_phi_1(x_E, details=True)
    return phi_1, beta_i_1


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    jxp.set_plot_options(grid=False)
    M_1_range = np.linspace(1.2, 2.0, 17)
    cases = {"Dense gas": (0.1, 0.95, "k-"), "Dilute gas": (0.01, 1.18, "k:")}

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for label, (p_r, T_r, style) in cases.items():
        set_inlet(p_r, T_r)
        results = np.array([unique_incidence(M) for M in M_1_range])
        axes[0].plot(M_1_range, np.degrees(results[:, 0]), style, label=label)
        axes[1].plot(M_1_range, np.degrees(results[:, 1]), style, label=label)
    axes[0].set_ylabel(r"$\phi_1$ [deg]")
    axes[1].set_ylabel(r"$\beta_{i,1}$ [deg]")
    for ax in axes:
        ax.set_xlabel(r"$M_1$ [-]")
        ax.legend()
    plt.tight_layout()
    plt.show()
