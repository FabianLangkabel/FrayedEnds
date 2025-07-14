from pathlib import Path
import subprocess


# Define the root directory to search
root_dir = Path('/workspaces/MRA-OrbitalOptimization/automate_process')

# Define the common ending of the file
file_ending = 'HeH_SpinorbOpt.json'

# Traverse through all subdirectories
for folder in root_dir.iterdir():
    if folder.is_dir():
        # Look for the file with the specific ending
        for file in folder.iterdir():
            if file.name.endswith(file_ending):
                print('Start spinorbital refinement with ', file)
                SpinorbOpt_cmd = ["/workspaces/MRA-OrbitalOptimization/build/madness_programs/spinorb_opt/SpinorbOpt", file]
                try: 
                    run_spinorbopt = subprocess.Popen(SpinorbOpt_cmd, stdout=subprocess.PIPE, text=True)
                    for line in run_spinorbopt.stdout:
                        print(line, end='')
                    run_spinorbopt.wait()
                except subprocess.CalledProcessError as e:
                    print("Error occurred:")
                    print(e.stderr)