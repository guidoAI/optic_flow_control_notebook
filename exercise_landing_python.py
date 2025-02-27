# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 19:12:15 2018

Different types of simple simulations that show the properties of optical flow control for landing.

Based on:
de Croon, G.C.H.E. (2016). Monocular distance estimation with optical flow maneuvers 
and efference copies: a stability-based strategy. Bioinspiration & biomimetics, 11(1), 016004.

@author: Guido de Croon.
"""

from scipy.integrate import ode
import numpy as np
import matplotlib.pyplot as plt

# supporting functions:

def f_continuous(t, x):
    """ f_continuous receives the time t and the state x
        It returns x_dot, i.e., the change in the state x over time
        at the time instant t.
    """
    
    # the error is the difference between the desired divergence and the measured divergence
    # the "measured divergence" is equal to the actual divergence in this function
    err = desired_div - (x[1] / x[0])
    
    # u is the control input, the vertical acceleration here.
    u = P * err
    
    # return x_dot
    return [x[1], u]

def f_ZOH(t, x, arg1):
    """ f_ZOH receives the time, state, and the control input as argument.
        It returns x_dot. 
    """
    
    # the control input is determined outside of the function
    # and stays constant for the time interval:
    u = arg1
    
    return [x[1], u]

def free_fall(t,x):
    """ free_fall models a free fall with earth gravity (no drag or anything)
    """
    return[x[1], -9.8]

def continuous_control():
    """ Control a drone to land with optical flow.
        The measurements are perfect and without any delay.
    """
    # set up an ordinary differential equation (ode) with the function f_continuous
    r = ode(f_continuous).set_integrator('zvode', method='bdf')
    r.set_initial_value(x0, t0)
    
    # simulate the controlled landing:
    states_over_time = np.zeros([n_time_steps, n_states])
    for t_index, t in enumerate(time_steps):
        # determine the next state using the ode:
        x = r.integrate(r.t+dt)
        # store the states over time for plotting:
        states_over_time[t_index, :] = x
        
    return states_over_time

def control_with_ZOH():
    """ Control a drone to land with optical flow.
        The measurements are perfect but determined with a zero-order hold (ZOH)
        as in a digital system.
    """
    
    # set up the ode:
    r = ode(f_ZOH).set_integrator('zvode', method='bdf')
    r.set_initial_value(x0, t0)
    
    # simulate the system:
    states_over_time = np.zeros([n_time_steps, n_states])
    for t_index, t in enumerate(time_steps):
        # determine the error = desired divergence - observed divergence:
        if(t_index >= 1):
            err = desired_div - (states_over_time[t_index-1, 1] / states_over_time[t_index-1, 0])
        else:
            err = 0
        # determine the control input = vertical acceleration with a P-gain:
        u = P * err
        # set the argument passed to f_ZOH to the determined control input:
        r.set_f_params(u)
        # determine the next state using the ode:
        x = r.integrate(r.t+dt)
        # store the states over time for plotting:
        states_over_time[t_index, :] = x
        
    return states_over_time

def control_with_delay(time_steps_delay=3):
    """ Control a drone to land with optical flow.
        The measurements are perfect but determined with a zero-order hold (ZOH)
        as in a digital system and with an additional delay.
    """
    
    # set up the ode:
    r = ode(f_ZOH).set_integrator('zvode', method='bdf')
    r.set_initial_value(x0, t0)
    
    # simulate the system:
    states_over_time = np.zeros([n_time_steps, n_states])
    u = np.zeros([n_time_steps, 1])
    for t_index, t in enumerate(time_steps):
        
        # determine the error = desired divergence - observed divergence
        # the observed divergence is the actual divergence of a few time steps ago.
        if(t_index >= time_steps_delay+1):
            err = desired_div - (states_over_time[t_index-time_steps_delay-1, 1] / states_over_time[t_index-time_steps_delay-1, 0])
        else:
            err = 0
        # determine the control input = vertical acceleration with a P-gain:
        u[t_index] = P * err
        # set the argument passed to f_ZOH to the determined control input:
        r.set_f_params(u[t_index][0])
        # determine the next state using the ode:
        x = r.integrate(r.t+dt)
        # store the states over time for plotting:
        states_over_time[t_index, :] = x
        
    return states_over_time, u

def control_with_delay_and_noise(noise_std= 0.02, time_steps_delay=3, new_strategy=False, time_window_fit = 30):
    """ Control a drone to land with optical flow.
        The measurements are noisy, determined with a zero-order hold (ZOH)
        as in a digital system and have an additional delay.
    """
    
    # set up the ode:
    r = ode(f_ZOH).set_integrator('zvode', method='bdf')
    r.set_initial_value(x0, t0)
    
    # simulate the system:
    states_over_time = np.zeros([n_time_steps, n_states])
    observations_over_time = np.zeros([n_time_steps, 1])
    effectiveness_over_time = np.zeros([n_time_steps, 1])
    error_fit_over_time = np.zeros([n_time_steps, 1])
    effectiveness = 0
    u = np.zeros([n_time_steps, 1])

    for t_index, t in enumerate(time_steps):
        
        # determine the error = desired divergence - observed divergence
        # the observed divergence is the actual divergence of a few time steps ago.
        if(t_index >= time_steps_delay+1):
            measured_div = states_over_time[t_index-time_steps_delay-1, 1] / states_over_time[t_index-time_steps_delay-1, 0]
            measured_div += np.random.normal(0, noise_std)
            observations_over_time[t_index] = measured_div
            err = desired_div - measured_div
        else:
            observations_over_time[t_index] = 0
            err = 0

        # Determine the effectiveness of control input changes
        eps = 1E-3
        if(t_index >= time_window_fit):
            
            v = np.polyfit(time_steps[t_index-time_window_fit:t_index], observations_over_time[t_index-time_window_fit:t_index], 1, full=True)
            params_div = v[0]
            residuals = v[1][0]
            
            error_fit_over_time[t_index] = residuals
            slope_div = params_div[0][0]

            mean_u = np.mean(u[t_index-time_window_fit:t_index])
            # This will just give the inverse of the control gain.
            # params_u = np.polyfit(time_steps[t_index-time_window_fit:t_index], u[t_index-time_window_fit:t_index], 1)
            # slope_u = params_u[0][0]
            # print(f'Slope u = {slope_u}, slope div = {slope_div}')
            #if(abs(8-t) < (dt/2) or abs(2-t) < (dt/2)):
            if(t_index % int(round(1/dt)) == 0):
                plot_fit = False
                if(plot_fit):
                    plt.figure()
                    plt.plot(time_steps[t_index-time_window_fit:t_index], observations_over_time[t_index-time_window_fit:t_index], 'b')
                    plt.plot(time_steps[t_index-time_window_fit:t_index], params_div[1] + params_div[0]*time_steps[t_index-time_window_fit:t_index], 'k--')
                    plt.plot(time_steps[t_index-time_window_fit:t_index], u[t_index-time_window_fit:t_index], 'r')
                    # plt.plot(time_steps[t_index-time_window_fit:t_index], params_u[1] + params_u[0]*time_steps[t_index-time_window_fit:t_index], 'k--')
                    # plot mean u:
                    plt.plot(time_steps[t_index-time_window_fit:t_index], mean_u*np.ones(time_window_fit), 'k--')
                    plt.legend(['divergence', 'divergence fit', 'control input', 'control input fit'])
                    plt.show()
                #print('slope_div: ', slope_div)
                # print('slope_u: ', slope_u)
                #print('mean_u: ', mean_u)
            if(abs(mean_u) > eps):
                effectiveness = abs(slope_div / mean_u)
                #print('Effectiveness: ', effectiveness)
        effectiveness_over_time[t_index] = effectiveness
        
        # determine the control input = vertical acceleration with a P-gain:
        if(new_strategy):
            
            if(effectiveness > 1E-5 and t_index >= time_window_fit):
                effectiveness_lp = np.mean(effectiveness_over_time[t_index-time_window_fit:t_index])
                factor = 1.0 / effectiveness_lp
                factor = min(factor, 1.0) 
            else:
                factor = 1.0
            P_eff = P * factor
            print(f'P_eff = {P_eff}, factor = {factor}')
            u[t_index] = P_eff * err
        else:
            u[t_index] = P * err
        # set the argument passed to f_ZOH to the determined control input:
        r.set_f_params(u[t_index][0])
        # determine the next state using the ode:
        x = r.integrate(r.t+dt)
        # store the states over time for plotting:
        states_over_time[t_index, :] = x
        
    return states_over_time, u, observations_over_time, effectiveness_over_time, error_fit_over_time

def low_pass_filter_array(v, alpha):
    v_lp = v[0]
    v_lps = np.zeros(len(v))
    for i in range(len(v)):
        v_lp = alpha*v[i] + (1-alpha)*v_lp
        v_lps[i] = v_lp
    return v_lps

def running_average_array(v, window=30):
    v_ra = np.mean(v[0:window])
    v_ras = np.zeros(len(v))
    for i in range(len(v)):
        if(i >= window):
            v_ra = v_ra + (v[i] - v[i-window])/window
        v_ras[i] = v_ra
    return v_ras

def windowed_variance(v, window=30):
    # Windowed variance:
    v_wvs = np.zeros(len(v))
    for i in range(len(v)):
        if(i >= window):
            v_wvs[i] = np.var(v[i-window:i])
    return v_wvs

def plot_states_over_time(states_over_time, time_steps, plot_title='', u=[], 
                          observations_over_time=[], effectiveness_over_time=[],
                          error_fit_over_time=[]):
    """ Plot the states over time.
    """
    
    # Height and vertical velocity over time:
    plt.figure()
    plt.plot(time_steps, states_over_time[:,0], label='z')
    plt.plot(time_steps, states_over_time[:,1], label='v_z')
    plt.xlabel('Time [s]')
    plt.ylabel('z [m], v_z [m/s]')
    plt.legend()
    if(plot_title != ''):
        plt.title(plot_title)

    # divergence over time with respect to the desired divergence:
    plt.figure()
    plt.plot(time_steps, np.divide(states_over_time[:,1],states_over_time[:,0]), label='divergence')
    plt.plot([time_steps[0], time_steps[-1]], [desired_div, desired_div], '--', label='desired divergence')
    plt.xlabel('Time [s]')
    plt.ylabel('Divergence [/s]')
    plt.legend()
    if(plot_title != ''):
        plt.title(plot_title)
    
    # Height and vertical velocity over time:
    fig = plt.figure()
    plt.plot(states_over_time[:,1], states_over_time[:,0])
    ax = fig.gca()
    grid_step_z = 0.5
    grid_step_vz = 0.25
    ax.set_xticks(np.arange(round(min(states_over_time[:,1])), round(max(states_over_time[:,1]))+1.0, grid_step_vz))
    ax.set_yticks(np.arange(round(min(states_over_time[:,0])), round(max(states_over_time[:,0]))+1.0, grid_step_z))
    #ax.axis('equal')
    plt.xlabel('v_z [m/s]')
    plt.ylabel('z [m]')
    plt.grid()
    if(plot_title != ''):
        plt.title(plot_title)
    
    if len(u) > 0:
        fig_u = plt.figure()
        plt.plot(time_steps, u, label='u')    
        ax = fig_u.gca()
        grid_step_t = 0.5
        ax.set_xticks(np.arange(round(min(time_steps)), round(max(time_steps))+1.0, grid_step_t))
        plt.xlabel('Time [s]')
        plt.ylabel('u [m/s^2]')
        plt.legend()
        plt.grid()
        if(plot_title != ''):
            plt.title(plot_title)

    if len(observations_over_time) > 0:
        fig_obs = plt.figure()
        plt.plot(time_steps, observations_over_time, label='observed divergence')    
        ax = fig_obs.gca()
        grid_step_t = 0.5
        ax.set_xticks(np.arange(round(min(time_steps)), round(max(time_steps))+1.0, grid_step_t))
        plt.xlabel('Time [s]')
        plt.ylabel('Divergence [/s]')
        plt.legend()
        plt.grid()
        if(plot_title != ''):
            plt.title(plot_title)

    if len(effectiveness_over_time) > 0:

        alpha = 0.1
        eff_lp = low_pass_filter_array(effectiveness_over_time, alpha)
        window = 30
        eff_ra = running_average_array(effectiveness_over_time, window)
        eff_wv = windowed_variance(observations_over_time, window)

        fig_eff = plt.figure()
        plt.subplot(311)
        plt.plot(time_steps, effectiveness_over_time, 'k', label='effectiveness')
        plt.plot(time_steps, eff_lp, 'g--', label='effectiveness low pass')
        plt.plot(time_steps, eff_ra, 'b--', label='effectiveness running average')
        # plot the z coordinate over time:
        plt.plot(time_steps, 1.0/states_over_time[:,0], 'r:', label='1/z')
        ax = fig_eff.gca()
        grid_step_t = 0.5
        ax.set_xticks(np.arange(round(min(time_steps)), round(max(time_steps))+1.0, grid_step_t))
        plt.xlabel('Time [s]')
        plt.ylabel('Effectiveness [?]')
        plt.legend()
        plt.grid()
        if(plot_title != ''):
            plt.title(plot_title)
        plt.subplot(312)
        plt.plot(time_steps, error_fit_over_time, 'r', label='error fit')
        plt.xlabel('Time [s]')
        plt.ylabel('Error fit [?]')
        plt.subplot(313)
        plt.plot(time_steps, eff_wv, 'm--', label='effectiveness windowed variance')
        plt.xlabel('Time [s]')
        plt.ylabel('Variance divergence [?]') 

    
if __name__ == '__main__':
    # ********************************************************
    # TODO: Try out all three different types of 'simulations'
    # You can do this by setting them one by one to True
    # ********************************************************
    
    PERFECT_MEASUREMENTS = False
    ZOH = False
    DELAY = False
    NOISE = True
    
    # *****************************************************************************
    # TODO: play around with initial state x0 and desired divergence (desired_div):
    # *****************************************************************************
    
    # state x = [height, vertical velocity]
    x0 = [10.0, -2.0]
    # desired divergence = the desired (velocity / height)
    desired_div = x0[1] / x0[0]
    
    # ******************************************
    # TODO: play around with the control gain P:
    # ******************************************
    P = 20
    
    # *************************************
    # TODO: play around with the time step:
    # *************************************
    dt = 0.033
    
    # global variables:
    n_states = 2
    t0 = 0
    t1 = 13
    time_steps = np.arange(t0, t1, dt)
    n_time_steps = len(time_steps)
    
    if(PERFECT_MEASUREMENTS):
        # control with perfect measurements:
        states_over_time = continuous_control()
        plot_states_over_time(states_over_time, time_steps, plot_title='control with perfect measurements')
    
    if(ZOH):
        # control with ZOH:
        states_over_time = control_with_ZOH()
        plot_states_over_time(states_over_time, time_steps, plot_title='control with zero-order-hold')
    
    if(DELAY):
        # control with delay:
        states_over_time, u = control_with_delay(time_steps_delay = 3)
        plot_states_over_time(states_over_time, time_steps, plot_title='control with delay', u = u)
        print(f'Mean absolute control effort =  {np.mean(np.abs(u))}')
    
    if(NOISE):
        # control with noise:
        states_over_time, u, observations_over_time, effectiveness_over_time, error_fit_over_time = \
            control_with_delay_and_noise(noise_std = 0.05, time_window_fit = 30, new_strategy = True)
        plot_states_over_time(states_over_time, time_steps, plot_title='control with noise', u = u,
                               observations_over_time = observations_over_time, 
                               effectiveness_over_time = effectiveness_over_time,
                               error_fit_over_time = error_fit_over_time)
        print(f'Mean absolute control effort =  {np.mean(np.abs(u))}')

    plt.show()
    print('Done')