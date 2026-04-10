import json
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import stats

# --- Configuration ---
# Update these paths to point to your actual 3 json files
space = "nasbench101"
start_seed = 10500
num_submissions = 10
seeds = [start_seed + i for i in range(num_submissions)]
dataset = "cifar100"
if dataset == "cifar10":
    seeds = [10500, 10501, 10516, 10517] # cifar 10
else:
    seeds = [10502, 10503, 10518, 10519] # cifar 100
trials = 25
all_seeds = []
for seed in seeds:
    for trial in range(trials):
        all_seeds.append(seed * (trial+1))
all_seeds = list(set(all_seeds))
surrogate = "mlp"

surrogate_name_map = {
    "xgboost": "CustomXGBoost",
    "mlp": "CustomMLP"
}

if space == "nasbench201":
    llm_pred = f"LLM_NB201_Predictor_{surrogate_name_map[surrogate]}_bananas"
    run = "run_nb201"
    if dataset == "cifar10":
        ylims = (84, 92)
    else:
        ylims = (62, 74)
        # ylims = (72.5, 73.5)
elif space == "nasbench301":
    llm_pred = f"LLM_NB301_Predictor_{surrogate_name_map[surrogate]}_bananas"
    run = "run_nb301"
    ylims = (90, 96)
elif space == "nasbench101":
    llm_pred = f"LLM_NB101_Predictor_{surrogate_name_map[surrogate]}_bananas"
    run = "run_nb101"
    ylims = (87, 96)

formal_name = {'nasbench201': 'NAS-Bench-201', 'nasbench301': 'NAS-Bench-301', 'nasbench101': 'NAS-Bench-101'}

surrogate_folder = f"{surrogate_name_map[surrogate]}_bananas"

# get user environment variable

user = os.getenv("USER")

file_paths = {
    "BANANAS (Path Encoding)": f"/storage/ice-shared/vip-vvk/data/AOT/skravtsov3/{run}/{space}/{dataset}/{surrogate_folder}/{{seed}}/errors.json",
    "BANANAS (COLE)": f"/storage/ice-shared/vip-vvk/data/AOT/skravtsov3/{run}/{space}/{dataset}/{llm_pred}/{{seed}}/errors.json"
}

# Will update this with actual seeds loaded after loading data
output_filename_template = f"results/PAPER_search_trajectory_comparison_{space}_{dataset}_avg_{{n_seeds}}seeds_startseed_{start_seed}.pdf"

def load_data(filepath):
    """Parses the NASLib results JSON."""
    if not os.path.exists(filepath):
        print(f"Warning: File not found {filepath}")
        return None, None
        
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # The JSON is a list: [config_dict, results_dict]
    config = data[0]['search']
    results = data[1]
    
    # Extract Metric (Test Accuracy of the Incumbent)
    # Using test_acc to see true performance, or valid_acc if you strictly want search progression
    y_vals = results['valid_acc']
    
    # Convert to "best so far" (cumulative maximum)
    y_vals = np.maximum.accumulate(y_vals)
    
    # Calculate X-axis (Number of Architectures Evaluated)
    # Queries = iteration_index
    num_init = config.get('num_init', 10)
    k = config.get('k', 10) # Batch size per step
    
    x_vals = range(len(y_vals))
    
    return x_vals, y_vals

def load_all_seeds(file_template, seeds):
    """Load data from multiple seeds and return aggregated results."""
    all_y_vals = []
    max_length = 0
    
    for seed in seeds:
        filepath = file_template.format(seed=seed)
        x, y = load_data(filepath)
        
        if x is not None and y is not None:
            all_y_vals.append(y)
            max_length = max(max_length, len(y))
    
    if not all_y_vals:
        print(f"Warning: No valid data found for template {file_template}")
        return None, None
    
    # Pad shorter sequences with their last value
    padded_y_vals = []
    for y in all_y_vals:
        if len(y) < max_length:
            padded = list(y) + [y[-1]] * (max_length - len(y))
        else:
            padded = y
        padded_y_vals.append(padded)
    
    # Calculate mean and std across seeds
    y_mean = np.mean(padded_y_vals, axis=0)
    y_std = np.std(padded_y_vals, axis=0)
    y_sem = y_std / np.sqrt(len(all_y_vals))  # Standard error of mean
    
    # 95% confidence interval (1.96 * SEM for normal distribution)
    # Using t-distribution would be more accurate: stats.t.ppf(0.975, len(all_y_vals)-1)
    ci_multiplier = stats.t.ppf(0.975, len(all_y_vals)-1) if len(all_y_vals) > 1 else 1.96
    y_ci = ci_multiplier * y_sem
    
    x_vals = range(len(y_mean))
    
    print(f"Loaded {len(all_y_vals)} seeds for template {file_template}")
    
    return x_vals, y_mean, y_std, np.array(padded_y_vals), len(all_y_vals), y_ci

# --- Plotting ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [2, 1]})

colors = ['#d62728', '#2ca02c'] # Red, Green
markers = ['s', '^']

print(f"\nProcessing {len(all_seeds)} seeds: {all_seeds}\n")

# Store mean, std, and raw values for comparison and t-tests
method_means = {}
method_stds = {}
method_raw_data = {}
actual_seeds_loaded = 0  # Track actual number of seeds successfully loaded

for i, (label, file_template) in enumerate(file_paths.items()):
    result = load_all_seeds(file_template, all_seeds)
    
    if result[0] is not None:
        x, y_mean, y_std, y_raw, n_seeds_loaded, y_ci = result
        
        # Update actual seeds count (use max across methods)
        actual_seeds_loaded = max(actual_seeds_loaded, n_seeds_loaded)
        
        # Store for comparison
        method_means[label] = y_mean
        method_stds[label] = y_std
        method_raw_data[label] = y_raw
        
        # Plot mean line
        ax1.plot(x, y_mean, label=label, color=colors[i], marker=markers[i], 
                 markersize=5, markevery=max(1, len(x)//20), linewidth=2, alpha=0.8)
        
        # Plot shaded 95% confidence interval (not std - this matches the t-test)
        ax1.fill_between(x, y_mean - y_ci, y_mean + y_ci, 
                        color=colors[i], alpha=0.2)

# Set output filename with actual seeds loaded
output_filename = output_filename_template.format(n_seeds=actual_seeds_loaded)

# Add optimal architecture line for NASBench201
if space == "nasbench201":
    if dataset == "cifar100":
        optimal_acc = 73.49
    elif dataset == "cifar10":
        optimal_acc = 91.61
    else:
        optimal_acc = None
elif space == "nasbench101":
    optimal_acc = 95
    
if optimal_acc is not None:
    ax1.axhline(y=optimal_acc, color='black', linestyle='--', linewidth=1.5, 
                label=f'Optimal Architecture ({optimal_acc}%)', alpha=0.7)
        
# Calculate and print statistics
if len(method_means) >= 1:
    methods = list(method_means.keys())
    
    # Prepare text file path
    txt_filename = output_filename.replace('.pdf', '.txt')
    
    # Open file for writing
    with open(txt_filename, 'w') as f:
        # Get optimal performance if available
        
        # Write optimal performance
        if optimal_acc is not None:
            f.write(f"=== Optimal Architecture Performance ===\n")
            f.write(f"Optimal: {optimal_acc:.2f}%\n\n")
            print(f"\n=== Optimal Architecture Performance ===")
            print(f"Optimal: {optimal_acc:.2f}%\n")
        
        # Write method statistics (best individual = last data point)
        f.write("=== Method Statistics - Best Individual (% Validation Accuracy) ===\n")
        print("=== Method Statistics - Best Individual (% Validation Accuracy) ===")
        
        for method in methods:
            mean_acc = method_means[method][-1]  # Mean of final point across seeds
            std_acc = method_stds[method][-1]     # Std of final point across seeds
            
            # Print to console
            print(f"{method}: Mean = {mean_acc:.3f}%, Std = {std_acc:.3f}%")
            
            # Write to file
            f.write(f"{method}: Mean = {mean_acc:.3f}%, Std = {std_acc:.3f}%\n")

        print()
        f.write("\n")

        f.write("=== Method Statistics - Cumulative (% Validation Accuracy) ===\n")
        print("=== Method Statistics - Cumulative (% Validation Accuracy) ===")
        for method in methods:
            cumulative_mean = np.mean(method_means[method])
            print(f"{method}: Mean = {cumulative_mean:.3f}%")
            f.write(f"{method}: Mean = {cumulative_mean:.3f}%\n")

        
        print()
        f.write("\n")
        
        # Calculate AUC (Area Under Curve) for each method
        f.write("=== Method Statistics - AUC (Area Under Curve) ===\n")
        print("=== Method Statistics - AUC (Area Under Curve) ===")
        method_aucs = {}
        for method in methods:
            auc = np.trapz(method_means[method])
            method_aucs[method] = auc
            print(f"{method}: AUC = {auc:.2f}")
            f.write(f"{method}: AUC = {auc:.2f}\n")
        
        print()
        f.write("\n")
        
        # Calculate earliest iteration to reach multiple thresholds
        threshold_percentages = [0.98, 0.985, 0.9875, 0.99, 0.9925, 0.995]  # 90%, 95%, 99%, 99.5% of optimal
        
        f.write(f"=== Convergence Speed (Iterations to reach thresholds) ===\n")
        print(f"=== Convergence Speed (Iterations to reach thresholds) ===")
        
        for threshold_pct in threshold_percentages:
            threshold = optimal_acc * threshold_pct
            
            f.write(f"\nThreshold: {threshold:.2f}% ({threshold_pct*100:.1f}% of optimal)\n")
            print(f"\nThreshold: {threshold:.2f}% ({threshold_pct*100:.1f}% of optimal)")
            
            for method in methods:
                # Find first iteration where method reaches threshold
                iterations_to_threshold = np.where(method_means[method] >= threshold)[0]
                if len(iterations_to_threshold) > 0:
                    first_iter = iterations_to_threshold[0]
                    print(f"  {method}: {first_iter} iterations")
                    f.write(f"  {method}: {first_iter} iterations\n")
                else:
                    print(f"  {method}: Never reached threshold")
                    f.write(f"  {method}: Never reached threshold\n")
        
        print()
        f.write("\n")
        
        # Calculate and write pairwise differences
        if len(method_means) >= 2:
            f.write("=== Average Pairwise Differences (% Validation Accuracy) ===\n")
            print("=== Average Pairwise Differences (% Validation Accuracy) ===")
            
            for i in range(len(methods)):
                for j in range(i + 1, len(methods)):
                    method1, method2 = methods[i], methods[j]
                    max_common_length = min(len(method_means[method1]), len(method_means[method2]))
                    diff_cumulative = np.mean(method_means[method2] - method_means[method1])
                    diff_best = method_means[method2][max_common_length-1] - method_means[method1][max_common_length-1]
                    diff_auc = method_aucs[method2] - method_aucs[method1]
                    
                    # Print to console
                    print(f"Cumulative: {method2} - {method1}: {diff_cumulative:+.3f}%")
                    print(f"Best individual: {method2} - {method1}: {diff_best:+.3f}%")
                    print(f"AUC: {method2} - {method1}: {diff_auc:+.2f}")
                    
                    # Write to file
                    f.write(f"Cumulative: {method2} - {method1}: {diff_cumulative:+.3f}%\n")
                    f.write(f"Best individual: {method2} - {method1}: {diff_best:+.3f}%\n")
                    f.write(f"AUC: {method2} - {method1}: {diff_auc:+.2f}\n")
    
    print(f"\nStatistics saved to {txt_filename}")
    print()

# Compute t-test between BANANAS and BANANAS (COLE) at each generation
if "BANANAS (Path Encoding)" in method_raw_data and "BANANAS (COLE)" in method_raw_data:
    BANANAS_data = method_raw_data["BANANAS (Path Encoding)"]
    BANANAS_llm_data = method_raw_data["BANANAS (COLE)"]
    
    # Compute t-test at each iteration
    p_values = []
    num_iterations = BANANAS_data.shape[1]
    
    for iter_idx in range(num_iterations):
        # Get data for this iteration across all seeds
        BANANAS_scores = BANANAS_data[:, iter_idx]
        llm_scores = BANANAS_llm_data[:, iter_idx]
        
        # Perform two-sample t-test
        t_stat, p_val = stats.ttest_ind(llm_scores, BANANAS_scores)
        p_values.append(p_val)
    
    # Plot p-values
    ax2.plot(range(num_iterations), p_values, color='purple', linewidth=2, label='P-value')
    ax2.axhline(y=0.05, color='red', linestyle='--', linewidth=1, label='p=0.05 (significance threshold)', alpha=0.7)
    ax2.axhline(y=0.01, color='darkred', linestyle=':', linewidth=1, label='p=0.01', alpha=0.7)
    ax2.set_xlabel('Number of Architectures Evaluated', fontsize=12)
    ax2.set_ylabel('P-value (t-test)', fontsize=12)
    ax2.set_title('Statistical Significance: BANANAS (COLE) differs from BANANAS (Path Encoding)', fontsize=12)
    ax2.set_yscale('log')
    ax2.grid(True, which='both', linestyle='--', alpha=0.5)
    ax2.legend(fontsize=9)
    
    print(f"\nT-test results:")
    for it in [60, 120, 250, 500]:
        idx = min(it - 1, len(p_values) - 1)
        if idx >= 0:
            data_std = BANANAS_data[:, idx]
            data_llm = BANANAS_llm_data[:, idx]
            mean_std = np.mean(data_std)
            mean_llm = np.mean(data_llm)
            n_std = len(data_std)
            n_llm = len(data_llm)
            ci_std = (stats.t.ppf(0.975, n_std-1) if n_std > 1 else 1.96) * (np.std(data_std) / np.sqrt(n_std))
            ci_llm = (stats.t.ppf(0.975, n_llm-1) if n_llm > 1 else 1.96) * (np.std(data_llm) / np.sqrt(n_llm))
            print(f"  Iteration {it}: p={p_values[idx]:.4e} | BANANAS (Path Encoding): {mean_std:.2f}% ± {ci_std:.2f}% | BANANAS (COLE): {mean_llm:.2f}% ± {ci_llm:.2f}%")
    print(f"  P-value at final iteration: {p_values[-1]:.4e}")
    print(f"  Number of iterations with p<0.05: {sum(1 for p in p_values if p < 0.05)}/{len(p_values)}")
    print()
    
    if optimal_acc is not None:
        print("Convergence Speed Significance Testing (Iterations to threshold):")
        for threshold_pct in threshold_percentages:
            threshold = optimal_acc * threshold_pct
            
            iters_std = []
            for trial_data in BANANAS_data:
                idx = np.where(trial_data >= threshold)[0]
                iters_std.append(idx[0] if len(idx) > 0 else num_iterations)
                
            iters_llm = []
            for trial_data in BANANAS_llm_data:
                idx = np.where(trial_data >= threshold)[0]
                iters_llm.append(idx[0] if len(idx) > 0 else num_iterations)
                
            t_stat, p_val = stats.ttest_ind(iters_llm, iters_std)
            n_std_iters = len(iters_std)
            n_llm_iters = len(iters_llm)
            ci_std_iters = (stats.t.ppf(0.975, n_std_iters-1) if n_std_iters > 1 else 1.96) * (np.std(iters_std) / np.sqrt(n_std_iters))
            ci_llm_iters = (stats.t.ppf(0.975, n_llm_iters-1) if n_llm_iters > 1 else 1.96) * (np.std(iters_llm) / np.sqrt(n_llm_iters))
            print(f"  Threshold {threshold_pct*100:.2f}%: p={p_val:.4e} | BANANAS (Path Encoding): {np.mean(iters_std):.2f} ± {ci_std_iters:.2f} iters | BANANAS (COLE): {np.mean(iters_llm):.2f} ± {ci_llm_iters:.2f} iters")
        print()

# Styling for main plot
ax1.set_title(f"NAS Trajectory: {dataset.upper()} ({formal_name[space]}) - Average of {actual_seeds_loaded} Trials (95% CI)", fontsize=14)
ax1.set_xlabel("Number of Architectures Evaluated", fontsize=12)
ax1.set_ylabel("Validation Accuracy (%)", fontsize=12)
# hardcode y lower and upper limit
ax1.set_ylim(ylims[0], ylims[1])

ax1.grid(True, which='both', linestyle='--', alpha=0.5)
ax1.legend(fontsize=11)
plt.tight_layout()

# Save
os.makedirs("results", exist_ok=True)
fig.savefig(output_filename)
print(f"\nPlot saved to {output_filename}")