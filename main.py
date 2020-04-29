# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 16:09:34 2015
@author: Peter Meisrimel
"""
from pylab import close
from main_run import run_files
import sys
close('all')

"""
How to:
Enter input files as a list to be run (run as a group)
or 
Enter input files as arguments when starting from console (run in sequence)
"""
if len(sys.argv) == 1:
    print('performing input from main.py')
    ############# VERIFICATION RUN MODES AND PROBLEMS
    run_files('input_files/verification/adjoints/problem_{}{}.json', [0, 12], [''])
    run_files('input_files/verification/run_mode_{}{}.json', [-1,0], [''])
    run_files('input_files/verification/run_mode_2_{}.json', [0,1,2])
    run_files('input_files/verification/RK/run_mode_0_{}{}.json', [21, 32, 541, -21, -32], [''])
    run_files('input_files/verification/{}/run_mode_0{}.json',
              ['toy', 'ver', 'ver_time', 'heat_src_adv'], [''])
    run_files('input_files/verification/FSI_2D/run_mode_0_ext_{}.json', [0, 1])
    ############################
    ############# 0  TOY PROBLEM
    ############################
    """ RUN_MODE   2, tolerance comparison of defined input files """
    #run_files('input_files/2_tols/0_test/k_{}/func_{}/est_{}.json', [-1, -100], [1,2], [0,1,2,3,4])
    
    ##############################################
    ############# 12 HEAT SOURCE ADVECTION PROBLEM
    ##############################################
    """ RUN_MODE  -1, reference solution """
    ##run_files('input_files/-1_reference/12_conv_diff/dx_32_t_6/tol_{}.json', [10**(-i) for i in range(5,9)])
    ##run_files('input_files/-1_reference/12_conv_diff/dx_32_t_3/tol_{}.json', [10**(-i) for i in range(5,12)])
    """ RUN_MODE   2, DWR """
    #run_files('input_files/2_tols/12_conv_diff/forward_DWR/est_{}.json', [0,1,2,3,4])
    #run_files('input_files/2_tols/12_conv_diff/backward_DWR/est_{}.json', [0,1,2,3,4])
    
    ############################################
    ############# 21  2 D COUPLED HEAT EQUATIONS
    ############################################
    """ RUN_MODE  -1, reference solution """
    #run_files('input_files/-1_reference/21_FSI_2D/u0_7_dx_20/ref_tol_{}.json', [i for i in range(6, 10)])
    #run_files('input_files/-1_reference/21_FSI_2D/u0_7_dx_40/ref_tol_{}.json', [i for i in range(6, 10)])
    #run_files('input_files/-1_reference/21_FSI_2D/u0_7_dx_60/ref_tol_{}.json', [i for i in range(6, 10)])
    #run_files('input_files/-1_reference/21_FSI_2D/u0_7_dx_80/ref_tol_{}.json', [i for i in range(6, 10)])
    """ RUN_MODE   2, tolerance comparison of defined input files """
    #run_files('input_files/2_tols/21_FSI_2D/u0_7_dx_20/est_{}.json', [0,1,2,3])
    #run_files('input_files/2_tols/21_FSI_2D/u0_7_dx_40/est_{}.json', [0,1,2,3])
    #run_files('input_files/2_tols/21_FSI_2D/u0_7_dx_60/est_{}.json', [0,1,2,3])
    #run_files('input_files/2_tols/21_FSI_2D/u0_7_dx_80/est_{}.json', [0,1,2,3])
    
else:
    print('performing input from terminal')
    for i in range(1, len(sys.argv)):
        run_files(sys.argv[i])