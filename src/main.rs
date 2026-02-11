use rand::{SeedableRng, Rng};
use clap::Parser;
use serde::{Deserialize, Serialize};
use serde_yaml;
use rayon::prelude::*;
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;
use std::process::Command;
use value_envelopes::*;
use ndarray::prelude::*;

mod plotting;

#[derive(clap::Args, Debug, Serialize, Deserialize, Clone)]
pub struct Args {
    #[arg(short, long, default_value = "regret_curves")]
    pub experiment: String,

    #[arg(short = 'H', long, default_value_t = 4)]
    pub h: usize,

    #[arg(short, long, default_value_t = 12)]
    pub s: usize,

    #[arg(short, long, default_value_t = 3)]
    pub a: usize,

    #[arg(short, long, default_value_t = 10000)]
    pub t: usize,

    #[arg(short, long, default_value_t = 10)]
    pub n_seeds: usize,

    #[arg(long, default_value_t = true)]
    pub compile: bool,

    #[arg(long)]
    pub no_h_split: bool,

    #[arg(long, default_value_t = 42)]
    pub seed: u64,

    #[arg(long, default_value_t = 0.1)]
    pub reward_window_width: f64,

    #[arg(long, default_value_t = 200)]
    pub plot_resolution: usize,

    #[arg(long, default_value_t = 5)]
    pub num_points: usize,

    #[arg(long, default_value_t = true)]
    pub show_progress: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentConfig {
    pub experiment: String,
    pub h: usize,
    pub s: usize,
    pub a: usize,
    pub t: usize,
    pub n_seeds: usize,

    #[serde(default = "default_layered")]
    pub layered: bool,

    #[serde(default = "default_intermediate_reward")]
    pub intermediate_reward: (f64, f64),

    #[serde(default = "default_terminal_reward")]
    pub terminal_reward: (f64, f64),

    #[serde(default = "default_delta")]
    pub delta: f64,

    pub k_values: Option<Vec<usize>>,

    #[serde(default = "default_k")]
    pub k: usize,

    #[serde(default = "default_num_points")]
    pub num_points: usize,

    #[serde(default = "default_compile")]
    pub compile: bool,

    #[serde(default = "default_use_h_split")]
    pub use_h_split: bool,

    #[serde(default = "default_seed")]
    pub seed: u64,

    #[serde(default = "default_reward_window_width")]
    pub reward_window_width: f64,

    #[serde(default = "default_plot_resolution")]
    pub plot_resolution: usize,

    #[serde(default = "default_baseline_agents")]
    pub baseline_agents: Option<Vec<String>>,

    #[serde(default = "default_shaping_agents")]
    pub shaping_agents: Option<Vec<String>>,

    #[serde(default = "default_show_progress")]
    pub show_progress: bool,
}

fn default_layered() -> bool { true }
fn default_intermediate_reward() -> (f64, f64) { (0.0, 0.0) }
fn default_terminal_reward() -> (f64, f64) { (0.0, 1.0) }
fn default_delta() -> f64 { 0.05 }
fn default_k() -> usize { 60000 }
fn default_num_points() -> usize { 5 }
fn default_compile() -> bool { true }
fn default_use_h_split() -> bool { true }
fn default_seed() -> u64 { 42 }
fn default_reward_window_width() -> f64 { 0.1 }
fn default_plot_resolution() -> usize { 200 }
fn default_baseline_agents() -> Option<Vec<String>> { Some(vec!["standard_hoeffding".to_string()]) }
fn default_shaping_agents() -> Option<Vec<String>> { Some(vec!["v_shaping".to_string(), "q_shaping".to_string(), "count_init_hoeffding".to_string()]) }
fn default_show_progress() -> bool { true }

impl From<Args> for ExperimentConfig {
    fn from(args: Args) -> Self {
        Self {
            experiment: args.experiment,
            h: args.h,
            s: args.s,
            a: args.a,
            t: args.t,
            n_seeds: args.n_seeds,
            layered: true,
            intermediate_reward: (0.0, 0.0),
            terminal_reward: (0.0, 1.0),
            delta: 0.05,
            k_values: None,
            k: 6000, 
            num_points: args.num_points,
            compile: args.compile,
            use_h_split: !args.no_h_split,
            seed: args.seed,
            reward_window_width: args.reward_window_width,
            plot_resolution: args.plot_resolution,
            baseline_agents: Some(vec!["standard_hoeffding".to_string()]),
            shaping_agents: Some(vec!["v_shaping".to_string(), "q_shaping".to_string(), "count_init_hoeffding".to_string()]),
            show_progress: args.show_progress,
        }
    }
}

#[derive(Parser)]
struct Cli {
    /// Path to a YAML config file containing a list of experiments
    #[arg(short, long)]
    config: Option<String>,

    /// Fallback arguments for single-run mode
    #[command(flatten)]
    args: Args,
}

fn main() {
    // Create output directories if they don't exist
    fs::create_dir_all("data").unwrap();
    fs::create_dir_all("png").unwrap();
    fs::create_dir_all("pdf").unwrap();

    let cli = Cli::parse();

    if let Some(config_path) = cli.config {
        let configs = load_configs(&config_path);
        for config in configs {
            println!("--- Running experiment: {} ---", config.experiment);
            execute_experiment(&config);
        }
    } else {
        execute_experiment(&ExperimentConfig::from(cli.args));
    }
}

fn load_configs(path: &str) -> Vec<ExperimentConfig> {
    let content = fs::read_to_string(path).unwrap_or_else(|e| {
        eprintln!("Error: Failed to read config file '{}': {}", path as &str, e);
        if path.ends_with(".json") && !Path::new(path).exists() {
            let yaml_path = path.replace(".json", ".yaml");
            if Path::new(&yaml_path).exists() {
                eprintln!("Hint: A file named '{}' exists. Did you mean to use that?", yaml_path);
            }
        }
        std::process::exit(1);
    });
    
    // Try parsing as a list of experiment configs first
    if let Ok(configs) = serde_yaml::from_str::<Vec<ExperimentConfig>>(&content) {
        return configs;
    }

    // Try parsing as a list of paths to other config files
    if let Ok(paths) = serde_yaml::from_str::<Vec<String>>(&content) {
        let mut all_configs = Vec::new();
        for sub_path in paths {
            all_configs.extend(load_configs(&sub_path));
        }
        return all_configs;
    }

    // Try parsing as a single experiment config
    if let Ok(config) = serde_yaml::from_str::<ExperimentConfig>(&content) {
        return vec![config];
    }

    panic!("Failed to parse config YAML at {}. It must be a single ExperimentConfig, a list of ExperimentConfigs, or a list of paths (strings).", path);
}

fn execute_experiment(config: &ExperimentConfig) {
    let result = match config.experiment.as_str() {
        "regret_curves" => run_regret_curves(config),
        "expanding_reward" => run_mdp_trials(config, "ExpandingReward"),
        "sliding_window" => run_mdp_trials(config, "SlidingWindow_width0p10"),
        "convergence" => run_convergence_experiment(config),
        "compile_only" => compile_and_move_pdfs(config),
        _ => {
            println!("Unknown experiment: {}", config.experiment);
            return;
        }
    };

    if let Err(e) = result {
        eprintln!("Experiment '{}' failed: {}", config.experiment, e);
    }
}

fn run_regret_curves(config: &ExperimentConfig) -> Result<(), Box<dyn std::error::Error>> {
    let h = config.h;
    let s = config.s;
    let a = config.a;
    let t = config.t;
    let n_seeds = config.n_seeds;
    let delta = config.delta;
    let master_seed = config.seed;

    let mdp = TabularMDP::random(h, s, a, config.layered, Some(config.intermediate_reward), Some(config.terminal_reward), master_seed);
    
    let layered_str = if config.layered { "Layered" } else { "Standard" };
    let folder_name = format!("data/RegretCurves_H{}_S{}_A{}_T{}_{}", h, s, a, t, layered_str);
    fs::create_dir_all(&folder_name).unwrap();

    let mut k_values = config.k_values.clone().unwrap_or_else(|| vec![20000, 40000, 80000]); 
    for k in k_values.iter_mut() {
        if *k % h != 0 {
            *k = *k - (*k % h);
            println!("Adjusting K to {} to be a multiple of H={}", k, h);
        }
    }
    
    for &k in &k_values {
        fs::create_dir_all(format!("{}/K={}", folder_name, k)).unwrap();
    }

    let mut rng = rand::rngs::StdRng::seed_from_u64(master_seed + 1);
    let mut agent_seeds = Vec::new();
    for _ in 0..n_seeds {
        agent_seeds.push(rng.r#gen::<u64>());
    }

    let mut baseline_agents = config.baseline_agents.clone().unwrap_or_default();
    let mut shaping_agents = config.shaping_agents.clone().unwrap_or_default();
    
    // Deduplicate agents
    baseline_agents.sort();
    baseline_agents.dedup();
    shaping_agents.sort();
    shaping_agents.dedup();

    // Standard Agents (No Offline Data)
    if baseline_agents.contains(&"standard_hoeffding".to_string()) {
        println!("Running Standard UCBVI (Hoeffding)...");
        let standard_results: Vec<(Vec<f64>, Vec<f64>)> = agent_seeds.par_iter().enumerate().map(|(i, &seed)| {
            run_standard_ucbvi(&mdp, t, delta, seed, i == 0)
        }).collect();
        save_regret_data(&format!("{}/Standard_UCBVI_Hoeffding.dat", folder_name), &standard_results, t, config.plot_resolution);
    }

    if baseline_agents.contains(&"standard_bernstein".to_string()) {
        println!("Running Standard UCBVI (Bernstein)...");
        let standard_results: Vec<(Vec<f64>, Vec<f64>)> = agent_seeds.par_iter().enumerate().map(|(i, &seed)| {
            run_standard_ucbvi_bernstein(&mdp, t, delta, seed, i == 0)
        }).collect();
        save_regret_data(&format!("{}/Standard_UCBVI_Bernstein.dat", folder_name), &standard_results, t, config.plot_resolution);
    }

    if !shaping_agents.is_empty() {
        for &k in &k_values {
            println!("Running shaping agents for K={}...", k);
            let dataset = generate_dataset(&mdp, k, Some(master_seed + k as u64), config.show_progress);
            let offline_bounds = compute_offline_bounds(&mdp, &dataset, delta, Some(master_seed + k as u64 + 1), config.use_h_split);

            for agent in &shaping_agents {
                let (run_agent, filename) = match agent.as_str() {
                    "v_shaping" => (true, "V_Shaping.dat"),
                    "q_shaping" => (true, "Q_Shaping.dat"),
                    "count_init_hoeffding" => (true, "Count_Init_UCBVI_Hoeffding.dat"),
                    "count_init_bernstein" => (true, "Count_Init_UCBVI_Bernstein.dat"),
                    _ => (false, ""),
                };

                if run_agent {
                    let results: Vec<(Vec<f64>, Vec<f64>)> = agent_seeds.par_iter().enumerate().map(|(i, &seed)| {
                        match agent.as_str() {
                            "v_shaping" => run_v_shaping(&mdp, &offline_bounds, t, delta, seed, i == 0),
                            "q_shaping" => run_q_shaping(&mdp, &offline_bounds, t, delta, seed, i == 0),
                            "count_init_hoeffding" => run_count_initialized_ucbvi(&mdp, &offline_bounds, t, delta, seed, false, i == 0),
                            "count_init_bernstein" => run_count_initialized_ucbvi(&mdp, &offline_bounds, t, delta, seed, true, i == 0),
                            _ => unreachable!(),
                        }
                    }).collect();
                    save_regret_data(&format!("{}/K={}/{}", folder_name, k, filename), &results, t, config.plot_resolution);
                }
            }
        }
    }

    println!("Generating plots...");
    if shaping_agents.contains(&"v_shaping".to_string()) {
        if let Err(e) = plotting::plot_regret_curves(&folder_name, "V_Shaping", "png/rust_v_shaping.png") {
            eprintln!("Error plotting V_Shaping: {}", e);
        }
    }
    if shaping_agents.contains(&"q_shaping".to_string()) {
        if let Err(e) = plotting::plot_regret_curves(&folder_name, "Q_Shaping", "png/rust_q_shaping.png") {
            eprintln!("Error plotting Q_Shaping: {}", e);
        }
    }

    // Compile LaTeX files and move PDFs
    if config.compile {
        compile_and_move_pdfs(config)?;
    }

    Ok(())
}

fn save_regret_data(path: &str, results: &[(Vec<f64>, Vec<f64>)], t: usize, plot_resolution: usize) {
    let n_seeds = results.len();
    let mut cumulative_regrets = Array2::<f64>::zeros((n_seeds, t));
    for (i, (regrets, _)) in results.iter().enumerate() {
        let mut sum = 0.0;
        for (j, &r) in regrets.iter().enumerate() {
            sum += r;
            cumulative_regrets[[i, j]] = sum;
        }
    }

    let mean = cumulative_regrets.mean_axis(Axis(0)).unwrap();
    let std = cumulative_regrets.std_axis(Axis(0), 0.0);

    let num_points = plot_resolution.min(t);
    let mut file = File::create(path).unwrap();
    writeln!(file, "Episode CumulativeRegret StdDev").unwrap();

    for i in 0..num_points {
        let idx = (i * t) / num_points;
        if idx >= t { continue; }
        writeln!(file, "{} {:.6} {:.6}", idx + 1, mean[idx], std[idx]).unwrap();
    }
    println!("Saved {}", path);
}

fn run_mdp_trials(config: &ExperimentConfig, mode: &str) -> Result<(), Box<dyn std::error::Error>> {
    let h = config.h;
    let s = config.s;
    let a = config.a;
    let t = config.t;
    let n_seeds = config.n_seeds;
    let delta = config.delta;
    let mut k = config.k;
    if k % h != 0 {
        k = k - (k % h);
        println!("Adjusting K to {} to be a multiple of H={}", k, h);
    }
    let num_points = config.num_points;

    let folder_name = format!("data/{}", mode);
    // Purge old results to avoid "ghost" curves in plots
    if Path::new(&folder_name).exists() {
        fs::remove_dir_all(&folder_name)?;
    }
    fs::create_dir_all(&folder_name).unwrap();

    let x_values: Vec<f64> = if mode == "ExpandingReward" {
        (0..num_points).map(|i| 0.01 * (1.0 / 0.01 as f64).powf(i as f64 / (num_points - 1) as f64)).collect()
    } else {
        let width = config.reward_window_width;
        (0..num_points).map(|i| i as f64 * (1.0 - width) / (num_points - 1) as f64).collect()
    };

    let default_shaping_algos = vec!["v_shaping".to_string(), "q_shaping".to_string(), "count_init_hoeffding".to_string()];
    let shaping_agents = config.shaping_agents.as_ref().unwrap_or(&default_shaping_algos);

    let baseline_agent = config.baseline_agents.as_ref()
        .and_then(|v| v.first())
        .map(|s| s.as_str())
        .unwrap_or("standard_hoeffding");
    
    println!("Reference Baseline: {}", baseline_agent);

    let mut final_performance: std::collections::HashMap<String, Vec<(f64, f64)>> = std::collections::HashMap::new();
    for algo in shaping_agents {
        final_performance.insert(algo.to_string(), Vec::new());
    }

    for &x in &x_values {
        let terminal_range = if mode == "ExpandingReward" {
            (1.0 - x, 1.0)
        } else {
            let width = config.reward_window_width;
            (x, x + width)
        };

        println!("Running for x={:.4}, R_term=({:.3}, {:.3})", x, terminal_range.0, terminal_range.1);
        
        let mdp_seed = config.seed + (x * 100000.0) as u64;
        let mdp = TabularMDP::random(h, s, a, config.layered, Some(config.intermediate_reward), Some(terminal_range), mdp_seed);
        
        let mut rng = rand::rngs::StdRng::seed_from_u64(mdp_seed + 1);
        let mut agent_seeds = Vec::new();
        for _ in 0..n_seeds {
            agent_seeds.push(rng.r#gen::<u64>());
        }

        let dataset = generate_dataset(&mdp, k, Some(mdp_seed + 2), config.show_progress);
        let offline_bounds = compute_offline_bounds(&mdp, &dataset, delta, Some(mdp_seed + 3), config.use_h_split);

        let baseline_results: Vec<f64> = agent_seeds.par_iter().enumerate().map(|(i, &seed)| {
            let (regrets, _) = match baseline_agent {
                "standard_hoeffding" => run_standard_ucbvi(&mdp, t, delta, seed, i == 0),
                "standard_bernstein" => run_standard_ucbvi_bernstein(&mdp, t, delta, seed, i == 0),
                _ => {
                    if i == 0 { eprintln!("Warning: Unknown baseline agent '{}'. Falling back to standard_hoeffding.", baseline_agent); }
                    run_standard_ucbvi(&mdp, t, delta, seed, i == 0)
                }
            };
            regrets.iter().sum::<f64>()
        }).collect();

        for algo in shaping_agents {
            let algo_results: Vec<f64> = agent_seeds.par_iter().enumerate().map(|(i, &seed)| {
                let (regrets, _) = match algo.as_str() {
                    "v_shaping" => run_v_shaping(&mdp, &offline_bounds, t, delta, seed, i == 0),
                    "q_shaping" => run_q_shaping(&mdp, &offline_bounds, t, delta, seed, i == 0),
                    "Bonus_Shaping_Only" => run_bonus_shaping_only(&mdp, &offline_bounds, t, delta, seed, i == 0),
                    "Upper_Bonus_Shaping" => run_upper_bonus_shaping(&mdp, &offline_bounds, t, delta, seed, i == 0),
                    "Count_Init_UCBVI" | "count_init_hoeffding" => run_count_initialized_ucbvi(&mdp, &offline_bounds, t, delta, seed, false, i == 0),
                    "count_init_bernstein" => run_count_initialized_ucbvi(&mdp, &offline_bounds, t, delta, seed, true, i == 0),
                    _ => (vec![0.0; t], vec![0.0; t])
                };
                regrets.iter().sum::<f64>()
            }).collect();

            let mut improvements = Vec::new();
            for i in 0..n_seeds {
                let base = baseline_results[i].max(1e-6);
                improvements.push((base - algo_results[i]) / base);
            }

            let mean_imp = improvements.iter().sum::<f64>() / n_seeds as f64;
            let std_imp = (improvements.iter().map(|v| (v - mean_imp).powi(2)).sum::<f64>() / n_seeds as f64).sqrt();
            final_performance.get_mut(algo).unwrap().push((mean_imp, std_imp));
        }
    }

    for (algo, stats) in final_performance {
        let path = format!("{}/{}.dat", folder_name, algo);
        let mut file = File::create(&path).unwrap();
        writeln!(file, "X_Value Mean_Improvement Std_Improvement").unwrap();
        for (i, &x) in x_values.iter().enumerate() {
            let (mean, std) = stats[i];
            writeln!(file, "{:.8} {:.8} {:.8}", x, mean, std).unwrap();
        }
        println!("Saved {}", path);
    }

    println!("Generating plots...");
    if let Err(e) = plotting::plot_mdp_trials(&folder_name, mode, &format!("png/{}.png", mode), mode == "ExpandingReward") {
        eprintln!("Error plotting MDP trials ({}): {}", mode, e);
    }

    // Compile LaTeX files and move PDFs
    if config.compile {
        compile_and_move_pdfs(config)?;
    }

    Ok(())
}

fn run_convergence_experiment(config: &ExperimentConfig) -> Result<(), Box<dyn std::error::Error>> {
    let h_total = config.h;
    let s_total = config.s;
    let a_total = config.a;
    let delta = config.delta;
    let master_seed = config.seed;
    let h_specific = 2; // Fixed as in python script

    let mdp = TabularMDP::random(h_total, s_total, a_total, config.layered, Some(config.intermediate_reward), Some(config.terminal_reward), master_seed);
    let (v_opt, _) = value_iteration(&mdp);
    
    let s_layer = s_total / h_total;
    let s_start = h_specific * s_layer;
    let s_end = s_start + s_layer;
    
    let v_slice = v_opt.slice(s![h_specific, s_start..s_end]);
    let mut min_v = f64::INFINITY;
    let mut max_v = -f64::INFINITY;
    for &v in v_slice {
        if v < min_v { min_v = v; }
        if v > max_v { max_v = v; }
    }
    let span_h_layer = max_v - min_v;

    let k_min = 1000;
    let k_max = 100000;
    let num_points = config.num_points;
    let mut k_values = Vec::new();
    for i in 0..num_points {
        let k_raw = k_min + i * (k_max - k_min) / (num_points - 1);
        let k = k_raw - (k_raw % h_total);
        if k > 0 && !k_values.contains(&k) {
            k_values.push(k);
        }
    }

    println!("Generating dataset (K_max={})...", k_max);
    let large_dataset = generate_dataset(&mdp, k_max, Some(master_seed + 1), config.show_progress);

    let mut file = File::create("data/R_vs_D.dat").unwrap();
    let mut header = String::from("K");
    for s in s_start..s_end {
        header.push_str(&format!(" D_s{}", s));
    }
    header.push_str(" R_h_layer span_h_layer");
    writeln!(file, "{}", header).unwrap();

    for &k in &k_values {
        print!("\rProcessing K={}...          ", k);
        std::io::stdout().flush().unwrap();
        let subset = &large_dataset[0..k];
        let bounds = compute_offline_bounds(&mdp, subset, delta, Some(master_seed + k as u64), config.use_h_split);
        
        let u_slice = bounds.u_values.slice(s![h_specific, s_start..s_end]);
        let w_slice = bounds.w_values.slice(s![h_specific, s_start..s_end]);
        
        let mut min_w = f64::INFINITY;
        let mut max_u = -f64::INFINITY;
        for &v in u_slice { if v > max_u { max_u = v; } }
        for &v in w_slice { if v < min_w { min_w = v; } }
        let r_h_layer = max_u - min_w;

        let mut line = format!("{}", k);
        for s in s_start..s_end {
            line.push_str(&format!(" {:.6}", bounds.d[[h_specific, s]]));
        }
        line.push_str(&format!(" {:.6} {:.6}", r_h_layer, span_h_layer));
        writeln!(file, "{}", line).unwrap();
    }
    println!("\nSaved data/R_vs_D.dat");

    println!("Generating plots...");
    if let Err(e) = plotting::plot_convergence("data/R_vs_D.dat", "png/rust_R_vs_D.png") {
        eprintln!("Error plotting convergence: {}", e);
    }

    // Compile LaTeX files and move PDFs
    if config.compile {
        compile_and_move_pdfs(config)?;
    }

    Ok(())
}

fn compile_and_move_pdfs(config: &ExperimentConfig) -> Result<(), Box<dyn std::error::Error>> {

    let tex_dir = Path::new("tex files");
    let pdf_dir = Path::new("pdf");
    fs::create_dir_all(tex_dir)?;

    let default_baseline_agents = vec!["standard_hoeffding".to_string()];
    let default_shaping_agents = vec!["v_shaping".to_string(), "q_shaping".to_string()];
    let _baseline_agents = config.baseline_agents.as_ref().unwrap_or(&default_baseline_agents);
    let shaping_agents = config.shaping_agents.as_ref().unwrap_or(&default_shaping_agents);

    let tex_files = match config.experiment.as_str() {
        "regret_curves" => {
            let mut files = Vec::new();
            if shaping_agents.contains(&"v_shaping".to_string()) || shaping_agents.contains(&"count_init_hoeffding".to_string()) || shaping_agents.contains(&"count_init_bernstein".to_string()) {
                files.push("plot_v_shaping_vs_K.tex");
            }
            if shaping_agents.contains(&"q_shaping".to_string()) || shaping_agents.contains(&"count_init_hoeffding".to_string()) || shaping_agents.contains(&"count_init_bernstein".to_string()) {
                files.push("plot_q_shaping_vs_K.tex");
            }
            // If only baseline agents are selected, maybe we want a plot too? 
            // The logic above ensures we at least try to compile them if shaping agents are present. 
            // Let's just always include them for now if we want PDFs.
            if files.is_empty() {
                files.push("plot_v_shaping_vs_K.tex");
                files.push("plot_q_shaping_vs_K.tex");
            }
            files
        }
        "expanding_reward" | "sliding_window" => vec!["mdp_trials.tex"],
        "convergence" => vec!["R_vs_D.tex"],
        "compile_only" => vec![
            "plot_v_shaping_vs_K.tex",
            "plot_q_shaping_vs_K.tex",
            "mdp_trials.tex",
            "R_vs_D.tex",
        ],
        _ => vec![],
    };

    for tex_file in tex_files {
        let tex_path = tex_dir.join(tex_file);

        // Generate the tex content
        let content = match tex_file {
            "plot_v_shaping_vs_K.tex" => generate_plot_v_tex(config),
            "plot_q_shaping_vs_K.tex" => generate_plot_q_tex(config),
            "mdp_trials.tex" => generate_mdp_tex(config),
            "R_vs_D.tex" => generate_r_vs_d_tex(config),
            _ => continue,
        };
        fs::write(&tex_path, content)?;

        // Compile with pdflatex
        let output = Command::new("pdflatex")
            .arg("-output-directory")
            .arg(tex_dir)
            .arg(&tex_path)
            .output()?;

        if !output.status.success() {
            eprintln!("pdflatex failed for {}: stdout: {}, stderr: {}", tex_file, String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
            continue;
        }

        // Move the generated PDF to pdf/ directory
        let pdf_file = tex_path.with_extension("pdf");
        if pdf_file.exists() {
            let target_name = if tex_file == "mdp_trials.tex" {
                format!("{}_mdp_trials.pdf", config.experiment)
            } else {
                pdf_file.file_name().unwrap().to_str().unwrap().to_string()
            };
            let target_pdf = pdf_dir.join(target_name);
            fs::rename(&pdf_file, &target_pdf)?;
            println!("Moved {} to {}", pdf_file.display(), target_pdf.display());
        }

        // Cleanup auxiliary files
        for ext in &["aux", "log", "out"] {
            let aux_file = tex_path.with_extension(ext);
            if aux_file.exists() {
                let _ = fs::remove_file(aux_file);
            }
        }
    }

    if tex_dir.exists() {
        fs::remove_dir_all(tex_dir)?;
    }

    Ok(())
}

fn generate_plot_v_tex(config: &ExperimentConfig) -> String {
    let layered_str = if config.layered { "Layered" } else { "Standard" };
    let k_values = config.k_values.clone().unwrap_or_else(|| vec![20000, 40000, 80000]);
    let default_baseline_agents = vec!["standard_hoeffding".to_string()];
    let default_shaping_agents = vec!["v_shaping".to_string(), "q_shaping".to_string()];
    let baseline_agents = config.baseline_agents.as_ref().unwrap_or(&default_baseline_agents);
    let shaping_agents = config.shaping_agents.as_ref().unwrap_or(&default_shaping_agents);
    
    let mut add_plots = String::new();
    let colors = vec!["red", "orange", "blue", "green", "purple"];
    for (i, &k) in k_values.iter().enumerate() {
        let color = colors[i % colors.len()];
        // V_Shaping
        if shaping_agents.contains(&"v_shaping".to_string()) {
            add_plots.push_str(&format!(r#"    \addplot[{}, mark=square*, error bars/.cd, y dir=both, y explicit] 
        table[x expr=\thisrow{{Episode}}/ {}, y=CumulativeRegret, y error=StdDev] {{\mainfolder/K={}/V_Shaping.dat}};
    \addlegendentry{{V-Shaping ($K={}$k)}}
"#, color, config.t, k, k/1000));
        }
        
        // Count Init Hoeffding
        if shaping_agents.contains(&"count_init_hoeffding".to_string()) {
            add_plots.push_str(&format!(r#"    \addplot[{}, mark=o, dashed, error bars/.cd, y dir=both, y explicit] 
        table[x expr=\thisrow{{Episode}}/ {}, y=CumulativeRegret, y error=StdDev] {{\mainfolder/K={}/Count_Init_UCBVI_Hoeffding.dat}};
    \addlegendentry{{Count-Init Hoeffding ($K={}$k)}}
"#, color, config.t, k, k/1000));
        }

         // Count Init Bernstein
        if shaping_agents.contains(&"count_init_bernstein".to_string()) {
            add_plots.push_str(&format!(r#"    \addplot[{}, mark=*, solid, error bars/.cd, y dir=both, y explicit] 
        table[x expr=\thisrow{{Episode}}/ {}, y=CumulativeRegret, y error=StdDev] {{\mainfolder/K={}/Count_Init_UCBVI_Bernstein.dat}};
    \addlegendentry{{Count-Init Bernstein ($K={}$k)}}
"#, color, config.t, k, k/1000));
        }
    }

    let mut standard_plots = String::new();
    if baseline_agents.contains(&"standard_hoeffding".to_string()) {
        standard_plots.push_str(&format!(r#"    % Standard Hoeffding
    \addplot[
        black,
        dashed,
        thick,
        no marks,
        error bars/.cd, y dir=both, y explicit
    ] table[x expr=\thisrow{{Episode}}/ {}, y=CumulativeRegret, y error=StdDev] {{\mainfolder/Standard_UCBVI_Hoeffding.dat}};
    \addlegendentry{{Standard Hoeffding}}
"#, config.t));
    }

    if baseline_agents.contains(&"standard_bernstein".to_string()) {
        standard_plots.push_str(&format!(r#"    % Standard Bernstein
    \addplot[
        black,
        solid,
        very thick,
        no marks,
        error bars/.cd, y dir=both, y explicit
    ] table[x expr=\thisrow{{Episode}}/ {}, y=CumulativeRegret, y error=StdDev] {{\mainfolder/Standard_UCBVI_Bernstein.dat}};
    \addlegendentry{{Standard Bernstein}}
"#, config.t));
    }

    format!(r#"\documentclass[border=10pt]{{standalone}}
\usepackage{{tikz}}
\usepackage{{pgfplots}}
\pgfplotsset{{compat=1.17}}
\begin{{document}}
\begin{{tikzpicture}}
    \def\mainfolder{{data/RegretCurves_H{}_S{}_A{}_T{}_{}}}
    \begin{{axis}}[
        title={{V-Shaping Performance vs. Offline Data Size ($K$)}},
        xlabel={{Fraction of $T$ (T={})}},
        ylabel={{Cumulative Regret}},
        width=14cm,
        height=10cm,
        grid=major,
        legend pos=north west,
        legend cell align={{left}},
        xtick={{0,0.2,0.4,0.6,0.8,1.0}},
        xmin=0, xmax=1,
        y tick label style={{/pgf/number format/sci, /pgf/number format/precision=1}},
        xticklabel style={{/pgf/number format/fixed}}
    ]
{}
{}
    \end{{axis}}
\end{{tikzpicture}}
\end{{document}}"#, config.h, config.s, config.a, config.t, layered_str, config.t, standard_plots, add_plots)
}

fn generate_plot_q_tex(config: &ExperimentConfig) -> String {
    let layered_str = if config.layered { "Layered" } else { "Standard" };
    let k_values = config.k_values.clone().unwrap_or_else(|| vec![20000, 40000, 80000]);
    let default_baseline_agents = vec!["standard_hoeffding".to_string()];
    let default_shaping_agents = vec!["v_shaping".to_string(), "q_shaping".to_string()];
    let baseline_agents = config.baseline_agents.as_ref().unwrap_or(&default_baseline_agents);
    let shaping_agents = config.shaping_agents.as_ref().unwrap_or(&default_shaping_agents);
    
    let mut add_plots = String::new();
    let colors = vec!["red", "orange", "blue", "green", "purple"];
    for (i, &k) in k_values.iter().enumerate() {
        let color = colors[i % colors.len()];
        // Q Shaping
        if shaping_agents.contains(&"q_shaping".to_string()) {
            add_plots.push_str(&format!(r#"    \addplot[{}, mark=square*, error bars/.cd, y dir=both, y explicit] 
        table[x expr=\thisrow{{Episode}}/ {}, y=CumulativeRegret, y error=StdDev] {{\mainfolder/K={}/Q_Shaping.dat}};
    \addlegendentry{{Q-Shaping ($K={}$k)}}
"#, color, config.t, k, k/1000));
        }
        
        // Count Init Hoeffding 
        if shaping_agents.contains(&"count_init_hoeffding".to_string()) {
            add_plots.push_str(&format!(r#"    \addplot[{}, mark=o, dashed, error bars/.cd, y dir=both, y explicit] 
        table[x expr=\thisrow{{Episode}}/ {}, y=CumulativeRegret, y error=StdDev] {{\mainfolder/K={}/Count_Init_UCBVI_Hoeffding.dat}};
    \addlegendentry{{Count-Init Hoeffding ($K={}$k)}}
"#, color, config.t, k, k/1000));
        }

        // Count Init Bernstein
        if shaping_agents.contains(&"count_init_bernstein".to_string()) {
            add_plots.push_str(&format!(r#"    \addplot[{}, mark=*, solid, error bars/.cd, y dir=both, y explicit] 
        table[x expr=\thisrow{{Episode}}/ {}, y=CumulativeRegret, y error=StdDev] {{\mainfolder/K={}/Count_Init_UCBVI_Bernstein.dat}};
    \addlegendentry{{Count-Init Bernstein ($K={}$k)}}
"#, color, config.t, k, k/1000));
        }
    }

    let mut standard_plots = String::new();
    if baseline_agents.contains(&"standard_hoeffding".to_string()) {
        standard_plots.push_str(&format!(r#"    % Standard Hoeffding
    \addplot[
        black,
        dashed,
        thick,
        no marks,
        error bars/.cd, y dir=both, y explicit
    ] table[x expr=\thisrow{{Episode}}/ {}, y=CumulativeRegret, y error=StdDev] {{\mainfolder/Standard_UCBVI_Hoeffding.dat}};
    \addlegendentry{{Standard Hoeffding}}
"#, config.t));
    }

    if baseline_agents.contains(&"standard_bernstein".to_string()) {
        standard_plots.push_str(&format!(r#"    % Standard Bernstein
    \addplot[
        black,
        solid,
        very thick,
        no marks,
        error bars/.cd, y dir=both, y explicit
    ] table[x expr=\thisrow{{Episode}}/ {}, y=CumulativeRegret, y error=StdDev] {{\mainfolder/Standard_UCBVI_Bernstein.dat}};
    \addlegendentry{{Standard Bernstein}}
"#, config.t));
    }

    format!(r#"\documentclass[border=10pt]{{standalone}}
\usepackage{{tikz}}
\usepackage{{pgfplots}}
\pgfplotsset{{compat=1.17}}
\begin{{document}}
\begin{{tikzpicture}}
    \def\mainfolder{{data/RegretCurves_H{}_S{}_A{}_T{}_{}}}
    \begin{{axis}}[
        title={{Q-Shaping Performance vs. Offline Data Size ($K$)}},
        xlabel={{Fraction of $T$ (T={})}},
        ylabel={{Cumulative Regret}},
        width=14cm,
        height=10cm,
        grid=major,
        legend pos=north west,
        legend cell align={{left}},
        xtick={{0,0.2,0.4,0.6,0.8,1.0}},
        xmin=0, xmax=1,
        y tick label style={{/pgf/number format/sci, /pgf/number format/precision=1}},
        xticklabel style={{/pgf/number format/fixed}}
    ]
{}
{}
    \end{{axis}}
\end{{tikzpicture}}
\end{{document}}"#, config.h, config.s, config.a, config.t, layered_str, config.t, standard_plots, add_plots)
}

fn generate_mdp_tex(config: &ExperimentConfig) -> String {
    let (folder, title, xlabel, axis_type, extra_options) = if config.experiment == "expanding_reward" {
        ("ExpandingReward", "Expanding Reward: Bonus Shaping Comparison", r"Reward Range Width $x$", "semilogxaxis", "")
    } else {
        ("SlidingWindow_width0p10", "Sliding Window: Bonus Shaping Comparison", r"Reward Window Start Location ($x$) in $R_{\text{term}} = (x, x+0.1)$", "axis", "")
    };

    let folder_path = format!("data/{}", folder);
    let mut add_plots = String::new();

    let default_shaping_algos = vec!["v_shaping".to_string(), "q_shaping".to_string(), "count_init_hoeffding".to_string()];
    let shaping_agents = config.shaping_agents.as_ref().unwrap_or(&default_shaping_algos);
    
    let possible_plots = vec![
        ("Bonus_Shaping_Only.dat", "Full-Bonus", "purple", "square*"),
        ("v_shaping.dat", "V-Shaping (Full-Bonus)", "purple", "square*"),
        ("Upper_Bonus_Shaping.dat", "Upper-Bonus", "green!70!black", "o"),
        ("q_shaping.dat", "Q-Shaping (Upper-Bonus)", "green!70!black", "o"),
        ("Count_Init_UCBVI.dat", "Count-Init", "blue", "triangle*"),
        ("count_init_hoeffding.dat", "Count-Init (Hoeffding)", "blue", "triangle*"),
        ("count_init_bernstein.dat", "Count-Init (Bernstein)", "blue", "diamond*"),
    ];

    for (file, label, color, mark) in possible_plots {
        let full_path = format!("{}/{}", folder_path, file);
        let agent_id = file.replace(".dat", "");
        if shaping_agents.contains(&agent_id) && Path::new(&full_path).exists() {
            add_plots.push_str(&format!(r#"    \addplot[{}, solid, thick, mark={}, error bars/.cd, y dir=both, y explicit]
        table[x=X_Value, y=Mean_Improvement, y error=Std_Improvement] {{\folderpath/{}}};
    \addlegendentry{{{}}}
"#, color, mark, file, label));
        }
    }

    format!(r#"\documentclass[multi=standalonefigure]{{standalone}}
\usepackage{{amsmath}}
\usepackage{{tikz}}
\usepackage{{pgfplots}}
\pgfplotsset{{compat=1.17}}
\begin{{document}}
\begin{{standalonefigure}}
\begin{{tikzpicture}}
    \def\folderpath{{data/{}}}
    \begin{{{}}}[
        {extra_options}title={{{}}},
        xlabel={{{}}},
        ylabel={{Relative Regret Improvement}},
        width=12cm, height=8cm, grid=major,
        ymin=-0.1, ymax=1.0,
        yticklabel={{\tick}},
        legend style={{at={{(0.5,0.98)}}, anchor=north, draw=none, fill=none}},
        legend columns=2,
        legend cell align={{left}}
    ]
{}
    \end{{{}}}
\end{{tikzpicture}}
\end{{standalonefigure}}
\end{{document}}"#, folder, axis_type, title, xlabel, add_plots, axis_type)
}

fn generate_r_vs_d_tex(_config: &ExperimentConfig) -> String {
    r#"\documentclass[tikz]{standalone}
\usepackage{amsmath}
\usepackage{pgfplots}
\pgfplotsset{compat=1.17}
\begin{document}
\begin{tikzpicture}
\begin{axis}[
    width=12cm,
    height=8cm,
    title={Convergence of Bounds at $h=2$},
    xlabel={Offline Dataset Size ($K$ trajectories)},
    ylabel={Width of Bounding Interval},
    legend pos=north east,       
    legend style={
        draw=none,               
        fill=none                
    },
    grid=major, 
]
\addplot[orange, mark=*, mark options={fill=orange}] 
    table[x=K, y=D_s6] {data/R_vs_D.dat};
\addlegendentry{$D_{s6}$}
\addplot[orange, mark=square*, mark options={fill=orange}] 
    table[x=K, y=D_s7] {data/R_vs_D.dat};
\addlegendentry{$D_{s7}$}
\addplot[orange, mark=triangle*, mark options={fill=orange}] 
    table[x=K, y=D_s8] {data/R_vs_D.dat};
\addlegendentry{$D_{s8}$}
\addplot[blue, thick] 
    table[x=K, y=R_h_layer] {data/R_vs_D.dat};
\addlegendentry{$R_h$ (In-Layer Range)}
\addplot[red, thick] 
    table[x=K, y=span_h_layer] {data/R_vs_D.dat};
\addlegendentry{Optimal Span}
\end{axis}
\end{tikzpicture}
\end{document}"#.to_string()
}
