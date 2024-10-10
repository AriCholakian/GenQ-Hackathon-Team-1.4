import numpy as np
import matplotlib.pyplot as plt
from pulser import Pulse, Sequence, Register
from pulser_simulation import QutipEmulator
from pulser.devices import DigitalAnalogDevice
from pulser.waveforms import InterpolatedWaveform, RampWaveform, ConstantWaveform
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
import json
from pulser_pasqal import PasqalCloud
import os
import pulser
def evaluate_mapping(new_coords, *args):
    """Cost function to minimize. Ideally, the pairwise
    distances are conserved"""
    Q, shape = args
    new_coords = np.reshape(new_coords, shape)
    new_Q = squareform(
        DigitalAnalogDevice.interaction_coeff / pdist(new_coords) ** 6
    )
    return np.linalg.norm(new_Q - Q)

#%%
import json
from time import sleep

#%%
import numpy as np

def loss_func(dist_abs, theta) -> float:
    if 270 > theta > 90: # prevent wind effects upstream
        return 0
    k = 15 # constant from literature of wind turbine effect distances
    return (k/dist_abs ** 2) * np.cos(np.deg2rad(theta)) ** 2


# could be vectorized, for faster runtimes
def lookup_loss(P1: tuple, P2: tuple, power, direction) -> float:
    x1 = P1[0]
    x2 = P2[0]
    y1 = P1[1]
    y2 = P2[1]

    if P1 == P2: # on the diagonal
        return power[P2[2]]

    mean_power = (power[P2[2]] + power[P1[2]])/2
    
    delta_x = x2 - x1
    delta_y = y2 - y1
    
    theta_rad = np.arctan2(delta_y, delta_x)
    vector_angle = np.degrees(theta_rad)
    
    theta = (direction - vector_angle) % 360
    grid_spacing = 5 # Standard minimum separation of 5 blade radii as the minimum distance
    return (abs(mean_power) * loss_func(grid_spacing * np.sqrt((x1-x2) ** 2 + (y1-y2) ** 2), theta)) ** 3 # distance between P1 and P2
        

def lookup_xy(cell_ID, grid_size) -> tuple:
    return cell_ID % grid_size, cell_ID // grid_size, cell_ID  #x,y
    

def genQ(direction, power, grid_size) -> np.ndarray:
    Q = np.array([[0.0] * grid_size ** 2] * grid_size ** 2)
    for cellID_outer in range(grid_size ** 2):
        P1 = lookup_xy(cellID_outer, grid_size)
        for cellID_inner in range(grid_size ** 2):
            P2 = lookup_xy(cellID_inner, grid_size)
            loss = lookup_loss(P1, P2, power, direction)
            Q[cellID_outer, cellID_inner] = loss
            
    Q += Q.T - np.diag(Q.diagonal())   
    return np.array(Q)


def wind_prob_genQ(prob_dist: list[dict], grid_size: int) -> np.ndarray:
    Q_arr = np.zeros((grid_size ** 2, grid_size ** 2))
    for case in prob_dist:
        Q_arr += genQ(case["direction"], case["wind_speed"], grid_size=grid_size) * case["frequency"]
    
    return Q_arr

    
def find_optimal(results: list[dict], desired_mills) -> str:
    print(results)
    for detuning_run in results.values():
        print(f'{detuning_run=}')
        top_three = list(detuning_run.values())
        top_three.sort(reverse=True)
        threshold = top_three[min(3, len(top_three)-1)]
        for placement_map, hits in detuning_run.items():
            if placement_map.count("1") == desired_mills and hits >= threshold:
                return placement_map

    print("No m-mills solutions found")
    return ""

def visualize_result(optimal: str):
    size = round(len(optimal) ** 0.5)
    array_2d = np.array([int(bit) for bit in optimal]).reshape((size, size))
    plt.imshow(array_2d, cmap='pink')
    for i in range(size):
        for j in range(size):
            if array_2d[i,j]:
                text = plt.text(j, i, "Selected Site\nX",
                               ha="center", va="center", color="b")
    plt.xlabel("East-West ($5r_w$)")
    plt.ylabel("North-South ($5r_w$)")
    plt.title("Wind Farm Siting Results")
    plt.savefig("OUTPUT.png", dpi=500)
    plt.show()
    return array_2d


def run (input_data, solver_params, extra_arguments):
    Q = wind_prob_genQ(input_data["distribution"], input_data["metaparams"]["grid_size"])
    with open("t4dataset.json", 'w') as f:
        f.write(json.dumps(Q.tolist(), indent=4))
    
    bitstrings = [np.binary_repr(i, len(Q)) for i in range(2 ** len(Q))]
    costs = []
    # this takes exponential time with the dimension of the QUBO
    for b in bitstrings:
        z = np.array(list(b), dtype=int)
        cost = z.T @ Q @ z
        costs.append(cost)
    zipped = zip(bitstrings, costs)
    sort_zipped = sorted(zipped, key=lambda x: x[1])
    shape = (len(Q), 2)
    costs = []
    np.random.seed(0)
    # x0 = np.random.random(shape).flatten()
    grid = np.zeros((input_data["metaparams"]["grid_size"], input_data["metaparams"]["grid_size"]))
    initial_coords = []
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            x_0, y_0 = i, j  
            initial_coords.append([5*x_0,  5*y_0])
    x0 = np.array(initial_coords).flatten()
    
    res = minimize(
        evaluate_mapping,
        x0,
        args=(Q, shape),
        method="Nelder-Mead",
        tol=1e-6,
        options={"maxiter": 200000, "maxfev": None},
    )
    coords = np.reshape(res.x, (len(Q), 2))
    coords *= 5/np.min(pdist(coords))
    qubits = dict(enumerate(coords))
    reg = Register(qubits)
    Omega = 2*2*np.pi
    delta_0 = -10  # just has to be negative
    T = 5000  # time in ns, we choose a time long enough to ensure the propagation of information in the system

    seq = Sequence(reg, pulser.MockDevice)
    seq.declare_channel("ising", "rydberg_global")
    delta_param = seq.declare_variable('delta_f', dtype = int)
    # seq.declare_variable("delta_f", dtype=int)
    rise = Pulse.ConstantDetuning(
        RampWaveform(1000, 0.0, Omega), delta_0, 0.0
    )
    sweep = Pulse.ConstantAmplitude(
        Omega, RampWaveform(3000, delta_0, delta_param), 0.0
    )
    fall = Pulse.ConstantDetuning(
        RampWaveform(1000, Omega, 0.0), delta_param, 0.0
    )
    
    seq.add(rise, "ising")
    seq.add(sweep, "ising")
    seq.add(fall, "ising")

    seq.measure("ground-rydberg")
    ############################ FOR LOCAL SIMULATION ############################
    # all_results = {}
    # k=0
    # for dval in [1,2,4,8,16,32]:
    #     simul = QutipEmulator.from_sequence(seq.build(delta_f = dval))
    #     results = simul.run()
    #     all_results[k] = results.sample_final_state(N_samples = 100)
    #     k += 1


    
    ##############################################################################
    ########### THE CURRENT SOLVER IS CREATED FOR ONLY LOCAL SIMULATION ##########     
    ### PLEASE, VISIT THE EXAMPLES AND THE DOCUMENTATION FOR REMOTE SIMULATION ###
    ##############################################################################
    
    ########################### FOR REMOTE SIMULATION ############################   
    connection = PasqalCloud(
        username=os.environ.get('PASQAL_USERNAME'),  # Your username or email address for the Pasqal Cloud Platform
        project_id=os.environ.get('PASQAL_PROJECTID'),  # The ID of the project associated to your account
        password=os.environ.get('PASQAL_PASSWORD'),  # The password for your Pasqal Cloud Platform account
    )
    #setting options of the EmuTN backend
    backend_options = {'max_bond_dim' : 200} #this makes the emulator faster but limits how accurate it is
    #do not go beyond 500!
    
    #then we define the Emulator Configuration
    emu_tn_config = EmulatorConfig(backend_options=backend_options, 
                                   sampling_rate=0.1, 
                                   evaluation_times='Final')
    
    #we initialize the emu_tn backend, using our connection and our sequence (the same as before!)
    tn_bknd = EmuTNBackend(
        seq, connection=connection, config=emu_tn_config
    )
    
    # Remote execution, requires job_params
    job_params = [
        {"runs": 100, "variables": {"delta_f": 1}},
        {"runs": 100, "variables": {"delta_f": 2}},
        {"runs": 100, "variables": {"delta_f": 4}},
        {"runs": 100, "variables": {"delta_f": 8}},
        {"runs": 100, "variables": {"delta_f": 16}},
        {"runs": 100, "variables": {"delta_f": 32}}
    ]
    results = tn_bknd.run(job_params=job_params, wait = True)
    #############################################################################
    all_results = {}
    for k in range(6):
        all_results[k] = results[k].bitstring_counts

    optimal = find_optimal(all_results, input_data["metaparams"]["m_desired_mills"])
    visualize_result(optimal)

    with open("output.json", 'w') as f:
        final_data = {'selected': optimal}
        f.write(json.dumps(final_data))
        
    # final = results.get_final_state()
    # count_dict = results.sample_final_state()
    # return {x:int(count_dict[x]) for x in count_dict}
    return all_results