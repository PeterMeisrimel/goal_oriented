# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 14:03:21 2016
@author: Peter Meisrimel
"""
from __future__ import division
from run_simulation import run_simulation

## Reference running case, create reference solution, no plotting, will always save results
# \n run_mode = -1
class run_reference(run_simulation):
    ## main function for executing a run
    # @param self .
    # @param file_read parameters read from file
    # @return None
    def simulate(self, file_read):
        import time
        file_t = time.strftime("%Y-%m-%d-%H-%M-%S")    
        self.myfile = open(file_read['saving_path'] + 'reference_' + file_read['filename_self'] + '_{}.txt'.format(file_t), 'w')
        self.save_file_header(file_read)
            
        Time_integ, u0, dt0 = self.init_parameters(file_read, tol = file_read['dt_cont_tol'])
        
        _, _, _, res = Time_integ.run(dt0, u0, j_ref = 0)
        
        print('Reference value : %10.50f' % res)
        self.myfile.writelines('\n' + 'dt_cont_tol' + ' = %5.2e' % file_read['dt_cont_tol'])
        self.myfile.writelines('\n' + 'dt_cont_dt0 : {}'.format(file_read['dt_cont_dt0']))
        self.myfile.writelines('\n' + 'result' + ' = %10.50f' % res)
        self.save_file_end()