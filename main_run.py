# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 16:09:34 2015
@author: Peter Meisrimel
"""
from __future__               import division
from pylab                    import close
from run_tolerances           import run_tolerances
from run_basic                import run_basic
from run_reference            import run_reference
from run_adjoint              import run_adjoint
from dolfin import set_log_level
import os
import time
import json

set_log_level(30) # to suppress massive ammounts of output

## seperates file and path to file for given file with input path\n
# criteria for seperation is the last '/', so don't use '/' in filenames
# @param input_file input file including path
# @return input_file file path
# @return filename name of file
def get_folder_path(input_file):
    filename = ''
    while input_file[-1] != '/':
        filename = input_file[-1] + filename
        input_file = input_file[:-1]
    return input_file, filename

## Function for runtime measurement, combining start and end in one function
# @param start (bool) if True, starts measurement, if False, ends measurement using start_time
# @param text info text to print when starting measurement
# @param start_time start time for calculating passed time
# @return None, outputs here are only console based, actual runtime measurements take place in the time-integration functions
def runtime(start, text = 'missing text entry', start_time = 0):
    if start:
        print('...starting evaluation of ' + text)
        return time.time() # runtime measurement
    else:
        end_time = time.time() # runtime measurement
        print('...done')
        print('runtime : %2i minutes'%((end_time - start_time)/60) +
              '  %2.3f second'%((end_time - start_time)%60))

## Runs program for specified input file(s)
# @param *args Either enter single file or to be formatted string to file, which will be formated filling the {} consecutively
# @return None
def run_files(*args):
    close('all')
    # allow compact input, as filenames will likely have little variance
    if len(args) == 1:
        if type(args[0]) is list:
            input_file = args[0]
        else:
            input_file = [args[0]]
    elif len(args) == 2:
        input_file = []
        for i in args[1]:
            input_file.append(args[0].replace('{}', str(i), 1))
    elif len(args) >= 3:
        for i in args[1]:
            run_files(args[0].replace('{}', str(i), 1), *args[2:])
        return 0
    
    file_readings, filenames = [], []
    problems_num, run_types_num = set([]), set([])
    
    # reading of files 
    print('...reading files')
    for file in input_file:
        file_read = {}
        with open(file, 'r') as myfile:
            file_read = json.load(myfile)
        file_read['path'] = file
        file_read['saving_path'], file_read['filename_self'] = get_folder_path(file)
        file_read['filename_self'] = file_read['filename_self'][:-5] # cutting the .json
        if file_read['run_mode'] == -1: # Reference
            file_read['saving_path'] += 'results_reference/'
        elif file_read['run_mode'] == 0: # Basic run
            file_read['saving_path'] += 'plots_basic/'
        elif file_read['run_mode'] == 2: # Tolerance comparison individual
            file_read['saving_path'] += 'plots_tols/'
        elif file_read['run_mode'] == 101: # adjoint visualization 
            file_read['saving_path'] += 'plots_' + file_read['filename_self'] + '/'
        if not os.path.exists(file_read['saving_path']): # create patch if needed
            os.makedirs(file_read['saving_path'])
        filenames.append(file_read['filename_self'])
        file_readings.append(file_read)
        problems_num.add(file_read['problem_type'])
        run_types_num.add(file_read['run_mode'])
    # check if run_types and problem_types align
    if len(problems_num) != 1:
        raise RuntimeError('Problem types do not match, check input files')
    if len(run_types_num) != 1:
        raise RuntimeError('run modes do not match, check input files')
    run_mode = file_readings[0]['run_mode']
    print('...files read')
    tols_min = 16
    tols_max = 0
    for file_read in file_readings:
        tols_min = min(tols_min, file_read['tol_step_start'])
        tols_max = max(tols_max, file_read['tol_step_start'] + file_read['tol_steps'])
    
    if run_mode in [-1, 0, 2]: # single folder output for several input files
        if run_mode == 0:      run_instance = run_basic                 (file_readings[0])
        elif run_mode == 2:    run_instance = run_tolerances            (file_readings[0], tols_min, tols_max)
        elif run_mode == -1:   run_instance = run_reference             (file_readings[0])
        else:                  raise RuntimeError('run mode not available')
        for file_read in file_readings:
            start_time = runtime(1, file_read['path'])
            run_instance.simulate(file_read)
            runtime(0, file_read['path'], start_time)
        if file_readings[0]['print']:
            run_instance.print_plot(file_read)
        close('all')
    elif run_mode in [101]: # multiple folder output, one for each input file
        for file_read in file_readings:
            if run_mode == 101: run_instance = run_adjoint                (file_read)
            else:                raise RuntimeError('run mode not available')
            start_time = runtime(1, file_read['path'])
            run_instance.simulate(file_read)
            runtime(0, file_read['path'], start_time)
            if file_readings[0]['print']:
                run_instance.print_plot(file_read)
            close('all')
    else:
        raise RuntimeError('run mode not available')
    print('...done')