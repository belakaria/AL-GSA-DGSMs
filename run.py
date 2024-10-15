from run_loop import run_loop

import pickle
import sys

function_name = str(sys.argv[1])
method_name = str(sys.argv[2])
num_al_iter = int(sys.argv[3])
n_init = int(sys.argv[4])
seed = int(sys.argv[5])


results = run_loop(
    function_name=function_name,
    method_name=method_name,
    n_init=n_init,
    num_al_iter=num_al_iter,
    seed=seed,
)

file_path = "results/" + function_name + '/' + function_name + "_" + method_name + "_"+ str(seed) + ".pickle"
pickle.dump(results, open(file_path, "wb"))
