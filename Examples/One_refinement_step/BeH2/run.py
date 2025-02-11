import subprocess as sp
    
# Execute orbital optimization
programm = sp.call("/workspaces/MRA-OrbitalOptimization/build/OrbitalOptimization madness_input.json", stdout=open('log', 'w'), stderr=open('err_log', 'w'), shell = True)
