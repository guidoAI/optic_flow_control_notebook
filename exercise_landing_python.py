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
    err = desired_div - (x[1] / x[0]);
    
    # u is the control input, the vertical acceleration here.
    u = P * err;
    
    # return x_dot
    return [x[1], u];

def f_ZOH(t, x, arg1):
    """ f_ZOH receives the time, state, and the control input as argument.
        It returns x_dot. 
    """
    
    # the control input is determined outside of the function
    # and stays constant for the time interval:
    u = arg1;
    
    return [x[1], u];

def free_fall(t,x):
    """ free_fall models a free fall with earth gravity (no drag or anything)
    """
    return[x[1], -9.8];

def continuous_control():
    """ Control a drone to land with optical flow.
        The measurements are perfect and without any delay.
    """
    # set up an ordinary differential equation (ode) with the function f_continuous
    r = ode(f_continuous).set_integrator('zvode', method='bdf')
    r.set_initial_value(x0, t0)
    
    # simulate the controlled landing:
    states_over_time = np.zeros([n_time_steps, n_states]);
    for t_index, t in enumerate(time_steps):
        # determine the next state using the ode:
        x = r.integrate(r.t+dt);
        # store the states over time for plotting:
        states_over_time[t_index, :] = x;
        
    return states_over_time;

def control_with_ZOH():
    """ Control a drone to land with optical flow.
        The measurements are perfect but determined with a zero-order hold (ZOH)
        as in a digital system.
    """
    
    # set up the ode:
    r = ode(f_ZOH).set_integrator('zvode', method='bdf')
    r.set_initial_value(x0, t0)
    
    # simulate the system:
    states_over_time = np.zeros([n_time_steps, n_states]);
    for t_index, t in enumerate(time_steps):
        # determine the error = desired divergence - observed divergence:
        if(t_index >= 1):
            err = desired_div - (states_over_time[t_index-1, 1] / states_over_time[t_index-1, 0]);
        else:
            err = 0;
        # determine the control input = vertical acceleration with a P-gain:
        u = P * err;
        # set the argument passed to f_ZOH to the determined control input:
        r.set_f_params(u);
        # determine the next state using the ode:
        x = r.integrate(r.t+dt);
        # store the states over time for plotting:
        states_over_time[t_index, :] = x;
        
    return states_over_time;

def control_with_delay(time_steps_delay=3):
    """ Control a drone to land with optical flow.
        The measurements are perfect but determined with a zero-order hold (ZOH)
        as in a digital system and with an additional delay.
    """
    
    # set up the ode:
    r = ode(f_ZOH).set_integrator('zvode', method='bdf')
    r.set_initial_value(x0, t0)
    
    # simulate the system:
    states_over_time = np.zeros([n_time_steps, n_states]);
    u = np.zeros([n_time_steps, 1]);
    for t_index, t in enumerate(time_steps):
        
        # determine the error = desired divergence - observed divergence
        # the observed divergence is the actual divergence of a few time steps ago.
        if(t_index >= time_steps_delay+1):
            err = desired_div - (states_over_time[t_index-time_steps_delay-1, 1] / states_over_time[t_index-time_steps_delay-1, 0]);
        else:
            err = 0;
        # determine the control input = vertical acceleration with a P-gain:
        u[t_index] = P * err;
        # set the argument passed to f_ZOH to the determined control input:
        r.set_f_params(u[t_index]);
        # determine the next state using the ode:
        x = r.integrate(r.t+dt);
        # store the states over time for plotting:
        states_over_time[t_index, :] = x;
        
    return states_over_time, u;

def plot_states_over_time(states_over_time, time_steps, plot_title='', u=[]):
    """ Plot the states over time.
    """
    
    # Height and vertical velocity over time:
    plt.figure();
    plt.plot(time_steps, states_over_time[:,0], label='z');
    plt.plot(time_steps, states_over_time[:,1], label='v_z');
    plt.xlabel('Time [s]');
    plt.ylabel('z [m], v_z [m/s]')
    plt.legend();
    if(plot_title != ''):
        plt.title(plot_title);

    # divergence over time with respect to the desired divergence:
    plt.figure();
    plt.plot(time_steps, np.divide(states_over_time[:,1],states_over_time[:,0]), label='divergence');
    plt.plot([time_steps[0], time_steps[-1]], [desired_div, desired_div], '--', label='desired divergence');
    plt.xlabel('Time [s]');
    plt.ylabel('Divergence [/s]')
    plt.legend();
    if(plot_title != ''):
        plt.title(plot_title);
    
    # Height and vertical velocity over time:
    fig = plt.figure();
    plt.plot(states_over_time[:,1], states_over_time[:,0]);
    ax = fig.gca();
    grid_step_z = 0.5;
    grid_step_vz = 0.25;
    ax.set_xticks(np.arange(round(min(states_over_time[:,1])), round(max(states_over_time[:,1]))+1.0, grid_step_vz));
    ax.set_yticks(np.arange(round(min(states_over_time[:,0])), round(max(states_over_time[:,0]))+1.0, grid_step_z));
    #ax.axis('equal');
    plt.xlabel('v_z [m/s]');
    plt.ylabel('z [m]')
    plt.grid();
    if(plot_title != ''):
        plt.title(plot_title);
    
    if len(u) > 0:
        fig_u = plt.figure();
        plt.plot(time_steps, u, label='u');    
        ax = fig_u.gca();
        grid_step_t = 0.5;
        ax.set_xticks(np.arange(round(min(time_steps)), round(max(time_steps))+1.0, grid_step_t));
        plt.xlabel('Time [s]');
        plt.ylabel('u [m/s^2]')
        plt.legend();
        plt.grid();
        if(plot_title != ''):
            plt.title(plot_title);
            
    
if __name__ == '__main__':
    # ********************************************************
    # TODO: Try out all three different types of 'simulations'
    # You can do this by setting them one by one to True
    # ********************************************************
    
    PERFECT_MEASUREMENTS = False;
    ZOH = True;
    DELAY = False;
    
    # *****************************************************************************
    # TODO: play around with initial state x0 and desired divergence (desired_div):
    # *****************************************************************************
    
    # state x = [height, vertical velocity]
    x0 = [10.0, -2.0]
    # desired divergence = the desired (velocity / height)
    desired_div = x0[1] / x0[0];
    
    # ******************************************
    # TODO: play around with the control gain P:
    # ******************************************
    P = 25;
    
    # *************************************
    # TODO: play around with the time step:
    # *************************************
    dt = 0.033;
    
    
    # global variables:
    n_states = 2;
    t0 = 0;
    t1 = 40;
    time_steps = np.arange(t0, t1, dt);
    n_time_steps = len(time_steps);
    
    
    if(PERFECT_MEASUREMENTS):
        # control with perfect measurements:
        states_over_time = continuous_control();
        plot_states_over_time(states_over_time, time_steps, plot_title='control with perfect measurements')
    
    if(ZOH):
        # control with ZOH:
        states_over_time = control_with_ZOH();
        plot_states_over_time(states_over_time, time_steps, plot_title='control with zero-order-hold')
    
    if(DELAY):
        # control with delay:
        states_over_time, u = control_with_delay(time_steps_delay = 3);
        plot_states_over_time(states_over_time, time_steps, plot_title='control with delay', u = u);
    
