import concurrent.futures
import random
import time
import json

# Example: Simulation function
def run_simulation(hyperparams):
    """Simulates a task with the given hyperparameters."""
    param1, param2 = hyperparams
    print(f"Starting simulation with params: {param1}, {param2}")
    time.sleep(random.uniform(0.5, 2))  # Simulate computation time
    result = {"param1": param1, "param2": param2, "accuracy": random.random()}
    print(f"Finished simulation with params: {param1}, {param2}, result: {result['accuracy']:.3f}")
    return result

# Grid of hyperparameters
hyperparameter_grid = [(x, y) for x in range(1, 5) for y in range(10, 15)]

# Results storage
results = []

def save_results_to_file(results):
    """Save results to a file."""
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)

# Thread pool execution
def multithreaded_grid_search(hyperparameter_grid, max_threads=4):
    with concurrent.futures.ThreadPoolExecutor(max_threads) as executor:
        # Submit all tasks
        future_to_params = {executor.submit(run_simulation, params): params for params in hyperparameter_grid}
        
        # Process as tasks complete
        for future in concurrent.futures.as_completed(future_to_params):
            params = future_to_params[future]
            try:
                result = future.result()
                results.append(result)
                print(f"Saved result for params: {params}")
            except Exception as exc:
                print(f"Simulation with params {params} generated an exception: {exc}")
    
    # Save all results to a file after completion
    save_results_to_file(results)
    print("All simulations completed and results saved.")

# Run the grid search
#print(hyperparameter_grid)
multithreaded_grid_search(hyperparameter_grid, max_threads=10)
