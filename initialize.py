# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 16:09:34 2015
@author: Peter Meisrimel
"""
import os
import json

## get dictionary with default parameters for initialization\n
# for adding new parameters, make sure they are ALL LOWER CASE (!!!)
# @return dictionary with default parameters for input file
def initialize_parameters_default():
    dic = {# Run type
           'run_mode'          : 1,
           'tol_step_start'    : 0,
           'tol_steps'         : 4,
           'dg'                : 1, # 1 for DG, 0 for RK
           'time_int_type'     : 0, # type of time integration scheme for RK case, see RK_get_class in schemes_RK.py
           
           # Problem spcifications
           'problem_type'      : 0,
           'problem_t_end'     : None, # default must be given by problem
           'problem_stiff_para': -1, # stiffness parameter
           'problem_meshsize'  : 32, # spatial resolution
           'problem_extended'  : 0, # do quadrature using extended system approach, i.e. using the time-integration method
           
           # Parameters for specific problems
           'direction'         : 1, # direction of velocity for heat advection problem
           'fsi_d1'            : 0.1, # Capital D in licentiate thesis/paper
           'fsi_d2'            : 1,
           'fsi_k1'            : 0.1, # lambda in licentiate thesis/paper
           'fsi_k2'            : 1,
           'fsi_tol_fixpoint'  : 1e-4, # tolerance for fixpoint iteration
           'fsi_tol_fp_fac'    : None, # if None, use fixpoint tolerance, else scaling based on actual tolerance
           'fsi_flux'          : 1, # base flux calculation on which subdomain, 1 for left, 2 for right
           'fsi_u0'            : 5, # initial condition for 2-D FSI problem
           
           # DWR stuff
           'dwr_scheme'        : 0, # 1 = DWR, 0 = no DWR
           'dwr_refine_perc'   : 0.8,
           'dwr_start_cells'   : 10, # number of cells in initial grid
           
           # timestep controller
           'dt_cont_type'      : 0, # deadbeat controller
           'dt_cont_f_min'     : 1e-2,
           'dt_cont_f_max'     : 3.,
           'dt_cont_f_safety'  : 1, # commonly used safety factor used for timestep controllers
           'dt_cont_tol'       : 1e-3, # used in creating reference solutions
           
           # choice of initial timestep
           'dt_cont_dt0'       : 0, # scheme used for determining the initial timestep
           
           # error estimation
           'err_est_type'      : 0, # type of error estimator used, e.g. specific areas or components
           
           # functionals used
           'func_type'         : 0,
           
           # number of steps for adjoint solution (run mode 101), will take 'dwr_start_cells' if not set
           'adjoint_steps'     : None,
           
           # printing & plotting options
           'print'             : 1, # save plots
           'legend'            : 'missing legend entry', # legend text
           'path'              : None, # determined by input parameters in main.py
           'saving_path'       : None, # determined by input parameters in main.py
           'filename_self'     : None, # determined by input parameters in main.py
           'save_results'      : 0, # save results to txt file
           'file_ending'       : '.eps',
           
           # auxiliary, for naming files
           'aux'               : 0
           }
    return dic

## Function to create multiple input files by taking in keys to vary\n
# see examples below on how to conviniently create input files
# @param filename template name of output file
# @param dic dictionary of input data, min. one value needs to be a list
# @param *args keys to be varied, iterates through lists given in dic, succ. filling {}s in filename, lists for parallel list iteration
# @return None, creates lots of files though10
def input_file_generator(filename, dic, *args):
    if len(args) == 0:
        # check if filename contains subdirectory and create those if needed
        if '/' in filename:
            direct = filename[:filename.rfind('/')]
            if not os.path.exists(direct):
                os.makedirs(direct)
        
        default = initialize_parameters_default()
        default_keys = list(default.keys())
        
        print('creating : ' + filename)
        with open(filename, 'w') as myfile:
            for i in dic.keys(): # overwrite desired inputs 
                assert i in default_keys, '{} is invalid input parameter'.format(i)
                default[i] = dic[i]
            myfile.write(json.dumps(default, sort_keys = True, indent = 4))
    else:
        lista = list(args[0])
        iter_list = dic[lista[0]]
        dic_new = dic.copy()
        for i in range(len(iter_list)):
            for j in lista:
                dic_new[j] = dic[j][i]
            input_file_generator(filename.replace('{}', str(dic[lista[0]][i]), 1), dic_new, *args[1:])

if __name__ == '__main__':
    ############# BASIC VERIFICATION
    input_file_generator('input_files/verification/run_mode_-1.json',  # run_mode = -1 test
                         {'run_mode': -1, 'problem_type': 12, 'problem_t_end': 1,
                         'problem_meshsize': 32, 'func_type': 1, 'dt_cont_tol': 1e-4})
    input_file_generator('input_files/verification/run_mode_0.json',   # run_mode = 0 test
                        {'run_mode': 0, 'problem_t_end': 1, 'problem_stiff_para': -1,
                         'func_type': 1, 'legend': r'$u_1$', 'save_results': 1, 'tol_steps': 7})
    input_file_generator('input_files/verification/run_mode_2_{}.json',   # run_mode = 2 test
                        {'run_mode': 2, 'problem_t_end': 1, 'problem_stiff_para': -1,
                         'func_type': 1, 'legend': ['norm', 'dwr', 'goal'], 'save_results': 1, 'tol_steps': [4,4,6],
                         'dwr_scheme': [0,1,0], 'err_est_type': [0,2,1]}, ['err_est_type', 'dwr_scheme', 'legend', 'tol_steps'])
    ## RK
    input_file_generator('input_files/verification/RK/run_mode_0_{}.json',   # run_mode = 0 test RK
                        {'run_mode': 0, 'problem_t_end': 1, 'problem_stiff_para': -1, 'problem_extended': 1,
                         'func_type': 2, 'legend': r'$u_1$', 'save_results': 1, 'tol_steps': [8,8,12,8,8], 'dg': 0,
                         'time_int_type': [21, 32, 541, -21, -32], 'problem_type': 0}, ['time_int_type', 'tol_steps'])
    ##  PROBLEMS VERIFICATION
    input_file_generator('input_files/verification/{}/run_mode_0.json', 
                        {'run_mode': 0, 'func_type': 1, 'problem_stiff_para': -100,
                         'save_results': 1, 'legend': 'test_leg',
                         'problem_type': [0,-1,-2,12], 'aux': ['toy', 'ver', 'ver_time', 'heat_src_adv'],
                         'problem_t_end': [2,1,1,6], 'tol_steps': [4,4,4,3]},
                         ['aux', 'problem_type', 'problem_t_end', 'tol_steps'])
    input_file_generator('input_files/verification/FSI_2D/run_mode_0_ext_{}.json', # run_mode = 0 FSI 2D problem
                        {'tol_steps': 4, 'problem_type': 21, 'run_mode': 0, 'legend': '$u$', 'problem_t_end': 0.2,
                         'dg': 0, 'problem_meshsize': 20, 'fsi_tol_fixpoint': 1e-9, 'fsi_u0': 7,
                         'fsi_d1': 0.1, 'fsi_d2': 1, 'fsi_k1': 0.01, 'fsi_k2': 1,
                         'problem_extended' : [0,1]}, ['problem_extended'])
    ## ADJOINT VERIFICATION
    input_file_generator('input_files/verification/adjoints/problem_{}.json',  # run_mode = 101 test
                        {'run_mode': 101, 'problem_t_end': 2, 'func_type': 1, 'problem_stiff_para': -1,
                         'problem_type': [-2, -1, 0, 1], 'adjoint_steps': 100}, ['problem_type'])
    input_file_generator('input_files/verification/adjoints/problem_12.json',  # run_mode = 101 test
                        {'run_mode': 101, 'problem_t_end': 6, 'func_type': 1,
                         'problem_type': 12, 'adjoint_steps': 100})
    ############# BASIC VERIFICATION END

    ############################
    ############# 0  TOY PROBLEM
    ############################
    ## RUN_MODE   2, tolerance comparison of defined input files
    input_file_generator('input_files/2_tols/0_test/k_{}/func_{}/est_{}.json',
                        {'run_mode': 2, 'problem_type': 0, 'problem_t_end': 2, 'file_ending': '.eps',
                         'save_results': 1, 'tol_steps': [9]*4 + [7], 'dg': 1,
                         'problem_stiff_para': [-1, -100], 'func_type': [1, 2, 10, 21, 22, 31, 32],
                         'err_est_type': [0,1,2,3,4], 'dwr_scheme': [0]*4 + [1],
                         'legend': ['Norm', 'Goal T', 'Goal Q', 'Goal T+Q', 'DWR']},
                         ['problem_stiff_para'], ['func_type'],
                         ['err_est_type', 'legend', 'dwr_scheme', 'tol_steps'])
    
    ##############################################
    ############# 12 HEAT SOURCE PROBLEM ADVECTION
    ##############################################
    ## RUN_MODE  -1, reference solution
    input_file_generator('input_files/-1_reference/12_conv_diff/dx_32_t_6/tol_{}.json',
                        {'run_mode': -1, 'problem_type': 12, 'problem_t_end': 6,
                         'problem_meshsize': 32, 'func_type': 1, 'dt_cont_tol': [10**(-i) for i in range(5,9)],
                         'save_results': 1}, ['dt_cont_tol'])
                         
    input_file_generator('input_files/-1_reference/12_conv_diff/dx_32_t_3/tol_{}.json',
                        {'run_mode': -1, 'problem_type': 12, 'problem_t_end': 3,
                         'problem_meshsize': 32, 'func_type': 1, 'dt_cont_tol': [10**(-i) for i in range(5,12)],
                         'save_results': 1, 'direction': -1}, ['dt_cont_tol'])
    ## RUN_MODE   2, tolerance comparison of defined input files
    # FORWARD
    input_file_generator('input_files/2_tols/12_conv_diff/forward_DWR/est_{}.json',
                        {'run_mode': 2, 'problem_type': 12, 'problem_t_end': 6, 
                         'dg': 1, 'direction': 1, 'problem_meshsize': 32, 'err_est_type': [0,1,2,3,4],
                         'func_type': 1, 'file_ending': '.eps','save_results': 1, 
                         'dwr_scheme': [0]*4+[1], 'tol_steps': [7,9,9,9,5],
                         'legend': ['Norm', 'Goal T', 'Goal Q', 'Goal T+Q', 'DWR']},
                          ['err_est_type', 'legend', 'dwr_scheme', 'tol_steps'])
    # BACKWARD
    input_file_generator('input_files/2_tols/12_conv_diff/backward_DWR/est_{}.json',
                        {'run_mode': 2, 'problem_type': 12, 'problem_t_end': 3, 
                         'dg': 1, 'direction': -1, 'problem_meshsize': 32, 'err_est_type': [0,1,2,3,4],
                         'func_type': 1, 'file_ending': '.eps','save_results': 1, 
                         'dwr_scheme': [0]*4+[1], 'tol_steps': [8,11,11,11,7],
                         'legend': ['Norm', 'Goal T', 'Goal Q', 'Goal T+Q', 'DWR']},
                          ['err_est_type', 'legend', 'dwr_scheme', 'tol_steps'])
    
    ########################################
    ############# 21 2 D THERMAL FSI PROBLEM
    ########################################
    ## RUN_MODE  -1, reference solution
    input_file_generator('input_files/-1_reference/21_FSI_2D/u0_7_dx_{}/ref_tol_{}.json',
                        {'problem_type': 21, 'run_mode': -1, 'problem_t_end': 0.2,
                         'dg': 0, 'problem_meshsize': [20, 40, 60, 80], 'fsi_tol_fixpoint': 1e-12, 'dt_cont_dt0': 1,
                         'fsi_d1': 0.1, 'fsi_d2': 1, 'fsi_k1': 0.01, 'fsi_k2': 1, 'fsi_u0': 7,
                         'dt_cont_tol': [10**(-i) for i in range(6, 10)], 'aux': [i for i in range(6,10)]},
                         ['problem_meshsize'], ['aux', 'dt_cont_tol'])
    ## RUN_MODE   2, tolerance comparison of defined input files
    input_file_generator('input_files/2_tols/21_FSI_2D/u0_7_dx_{}/est_{}.json',
                        {'tol_steps': [6,8,8,8], 'problem_type': 21, 'run_mode': 2, 'problem_t_end': 0.2,
                         'dg': 0, 'problem_meshsize': [20,40,60,80], 'fsi_tol_fp_fac': 0.01, 'dt_cont_dt0': 1,
                         'fsi_d1': 0.1, 'fsi_d2': 1, 'fsi_k1': 0.01, 'fsi_k2': 1,
                         'save_results': 1, 'file_ending': '.eps', 'fsi_u0': 7, 'err_est_type': [0,1,2,3],
                         'legend': ['Norm', 'Goal T', 'Goal Q', 'Goal T+Q']},
                         ['problem_meshsize'], ['err_est_type', 'legend', 'tol_steps'])