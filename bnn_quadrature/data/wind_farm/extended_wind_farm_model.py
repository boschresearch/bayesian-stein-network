"""Calculate C_T^* as a function
of ct_prime, theta
"""

import numpy as np
from py_wake import YZGrid
from py_wake.deficit_models import NiayifarGaussianDeficit
from py_wake.rotor_avg_models import GQGridRotorAvg
from py_wake.site import UniformSite
from py_wake.superposition_models import LinearSum
from py_wake.turbulence_models import CrespoHernandez
from py_wake.wind_farm_models.engineering_models import PropagateDownwind
from py_wake.wind_turbines import OneTypeWindTurbines
from pydantic import BaseModel


def wake_model(
    ct_prime: float,
    wake1: float,
    wake2: float,
    turbulence_intensity: float,
    theta: float,
    hub_height: float,
    diameter: float,
) -> float:
    """Use a low order wake model_type to
    calculate C_T^*

    :param ct_prime: float turbine resistance coefficient describing turbing operating conditions.
    Suggested distribution: Gaussian with mean 1.33 and variance 0.1.
    :param wake1: float first coefficient describing wake expansion. Small non-negative values.
    Suggested distribution: Gaussian with mean 0.38 and variance 0.001.
    :param wake2: float second coefficient describing wake expansion. Small non-negative values.
    Suggested distribution: Gaussian with mean 4e-3 and variance 1e-8.
    :param turbulence_intensity: float turbulence intensity. Non-negative values.
    Suggested distribution: Gaussian with mean 1.33 and variance 0.1.
    :param theta: float wind direction with respect to x direction. Values between 0 and 45 (degrees).
    Suggested distribution: mixture of von-Mises distributions, truncated to (0,45).
    :param hub_height:
    Suggested distribution: Gaussian with mean 100 and variance 0.5 truncated at 0
    :param diameter:
    Suggested distribution: Gaussian with mean 100 and variance 0.1 truncated at 0

    :returns ct_star: float local turbine thrust coefficient (dimensionless)
    """

    # check if all the data is of correct type
    if not check_inputs_wind_farm(ct_prime, theta, turbulence_intensity, wake1, wake2, hub_height, diameter):
        raise ValueError("Input check failed")
    # estimate thrust coefficient ct
    a = ct_prime / (4 + ct_prime)
    ct = 4 * a * (1 - a)

    # define a farm site with a background turbulence intensity of 0.1 (10%)
    site = UniformSite([1], turbulence_intensity)

    # calculate turbine coordinates
    S_x = 500  # Todo ask FX about this
    S_y = 500
    x = np.hstack((np.arange(0, -10000, -S_x), np.arange(S_x, 1500, S_x)))
    y = np.hstack((np.arange(0, 8500, S_y), np.arange(-S_y, -2000, -S_y)))
    xx, yy = np.meshgrid(x, y)
    x = xx.flatten()
    y = yy.flatten()

    # only consider turbines 10km upstream or 1km in the cross stream direction
    streamwise_cond = (
        -x * np.cos(theta * np.pi / 180) + y * np.sin(theta * np.pi / 180) < 10000
    )
    spanwise_cond = (
        abs(-y * np.cos(theta * np.pi / 180) - x * np.sin(theta * np.pi / 180)) < 2000
    )
    total_cond = np.logical_and(streamwise_cond, spanwise_cond)
    x_rot = x[total_cond]
    y_rot = y[total_cond]

    # create ideal turbines with constant thrust coefficients
    my_wt = OneTypeWindTurbines(
        name="MyWT",
        diameter=diameter,
        hub_height=hub_height,
        ct_func=lambda ws: np.interp(ws, [0, 30], [ct, ct]),
        power_func=lambda ws: np.interp(ws, [0, 30], [2, 2]),
        power_unit="kW",
    )
    windTurbines = my_wt

    # select models to calculate wake deficits behind turbines
    wake_deficit = PropagateDownwind(
        site,
        windTurbines,
        NiayifarGaussianDeficit(a=[wake1, wake2], use_effective_ws=True),
        superpositionModel=LinearSum(),
        rotorAvgModel=GQGridRotorAvg(4, 3),
        turbulenceModel=CrespoHernandez(),
    )

    # run wind farm simulation
    simulationResult = wake_deficit(x_rot, y_rot, ws=10, wd=270 + theta)

    # calculate turbine disk velocity
    U_T = (1 - a) * simulationResult.WS_eff[0]

    # calculate velocity in wind farm layer (0-250m above the surface)
    U_F = 0
    for i in np.linspace(-S_x, 0, 200):
        grid = YZGrid(
            x=i, y=np.linspace(-S_y / 2, S_y / 2, 200), z=np.linspace(0, 250, 20)
        )
        flow_map = simulationResult.flow_map(grid=grid, ws=10, wd=270 + theta)
        U_F += np.mean(flow_map.WS_eff)
    U_F = U_F / 200

    # calculate local turbine thrust coefficient
    ct_star = float(ct_prime * (U_T / U_F) ** 2)
    return ct_star


def check_inputs_wind_farm(ct_prime, theta, turbulence_intensity, wake1, wake2, hub_height, diameter):
    isinstance(ct_prime, float)
    isinstance(wake1, float)
    if wake1 <= 0:
        return False
    isinstance(wake2, float)
    if wake2 <= 0:
        return False
    isinstance(turbulence_intensity, float)
    if turbulence_intensity <= 0:
        return False
    if hub_height <= 0:
        return False
    if diameter <= 0:
        return False
    if not check_theta_in_limits(theta):
        return False
    return True


def check_theta_in_limits(theta):
    isinstance(theta, float)
    if 0 < theta < 45:
        return True
    return False


class WindData(BaseModel):
    ct_prime: float = 1.33
    wake1: float = 0.38
    wake2: float = 4e-3
    turbulence_intensity: float = 1.33
    theta: float = 0
    hub_height: float = 100
    diameter: float = 100
