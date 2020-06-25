# goal_oriented

Written by Peter Meisrimel, Azahar Monge (coupled heat eq. test case), credit for help on DWR to Patrick Farrell

contact: Peter.Meisrimel@na.lu.se
License: see gpl-3.0.txt

Related publications:

Meisrimel, Peter, and Philipp Birken. "On Goal Oriented Time Adaptivity using Local Error Estimates." PAMM 17.1 (2017): 849-850.

Meisrimel, Peter, and Philipp Birken. "Goal Oriented Time Adaptivity Using Local Error Estimates." Algorithms 13.5 (2020): 113.

Developed using Python 3.6.6 and dolfin v. 2018.1.0
Documented using doxygen (ver. 1.8.13)

After unpacking run:
doxygen doxygenconfig (to create documentation)
python initialize.py (to create input files)
python main.py (to verify the test cases work without errors)

The main code supports the trapezoidal rule and time-integration for quadrature.
The main test cases can be run by uncommenting the according parts in main.py.
Reference results have already been calculated and are in the according problem_x.py files.
Documentation is not 100% up to date, variable names in the code are largely similar to the ones used in the paper.

The verfication tests consist of testing most functions and producing plots. 
For the 2D problem the adjoint solution is plotted using the .pvd format and can, for example, be viewed using Paraview.

Quick overview over code structure:
Filenames are mostly self explaining, FSI is for the coupled heat equations test case

initialize.py is for generating input files based on a list of possible parameters. Any non specified parameter takes the default value.
What specifically is computed can be found in the simulate(...) function of the according run_x.py file. 
Visualization uses the according __init__(...) and print_plot(...) functions in the same files. 
Initialization of problem/controller/time integration classes is done in run_simulation.py.
