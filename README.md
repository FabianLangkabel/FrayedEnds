## Compile/Execute
- To compile the optimizer, the line "set(MADNESS_DIR "/home/fabian/MW-VQE/madness_install/opt/lib/cmake/madness")" in the “CMakeLists.txt” must be modified
- To execute the examples, the path to the OrbitalOptimizer must be adapted in the “run.py” scripts (program = sp.call(...)).
- The corresponding run.py script must be executed for the examples. An HPC start script is provided for Benzene.