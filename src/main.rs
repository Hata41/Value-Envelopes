use rand::{SeedableRng, Rng};
use clap::Parser;
use rayon::prelude::*;
use std::fs::{self, File};
use std::io::Write;
use value_envelopes::*;
use ndarray::prelude::*;

mod plotting;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value = "regret_curves")]
    experiment: String,

    #[arg(short, long, default_value_t = 4)]
    h: usize,

    #[arg(short, long, default_value_t = 12)]
    s: usize,

    #[arg(short, long, default_value_t = 3)]
    a: usize,

    #[arg(short, long, default_value_t = 1000000)]
    t: usize,

    #[arg(short, long, default_value_t = 10)]
    n_seeds: usize,
}

fn main() {
    let args = Args::parse();

    match args.experiment.as_str() {
        "regret_curves" => run_regret_curves(&args),
        "expanding_reward" => run_mdp_trials(&args, "ExpandingReward"),
        "sliding_window" => run_mdp_trials(&args, "SlidingWindow_width0p10"),
        "convergence" => run_convergence_experiment(&args),
        _ => println!("Unknown experiment: {}", args.experiment),
    }
}

fn run_regret_curves(args: &Args) {
    let h = args.h;
    let s = args.s;
    let a = args.a;
    let t = args.t;
    let n_seeds = args.n_seeds;
    let delta = 0.05;
    let master_seed = 42;

    let mdp = TabularMDP::random(h, s, a, true, Some((0.0, 0.0)), Some((0.0, 1.0)), master_seed);
    
    let folder_name = format!("RegretCurves_H{}_S{}_A{}_T{}_Layered", h, s, a, t);
    fs::create_dir_all(&folder_name).unwrap();

    let k_values = vec![h * 10000]; // matching Python's default [int(H_val * 1 * 1e4)]
    
    for &k in &k_values {
        fs::create_dir_all(format!("{}/K={}", folder_name, k)).unwrap();
    }

    let mut rng = rand::rngs::StdRng::seed_from_u64(master_seed + 1);
    let mut agent_seeds = Vec::new();
    for _ in 0..n_seeds {
        agent_seeds.push(rng.r#gen::<u64>());
    }

    println!("Running Standard UCBVI...");
    let standard_results: Vec<(Vec<f64>, Vec<f64>)> = agent_seeds.par_iter().map(|&seed| {
        run_standard_ucbvi(&mdp, t, delta, seed)
    }).collect();
    save_regret_data(&format!("{}/Standard_UCBVI.dat", folder_name), &standard_results, t);

    for &k in &k_values {
        println!("Running shaping agents for K={}...", k);
        let dataset = generate_dataset(&mdp, k, Some(master_seed + k as u64));
        let offline_bounds = compute_offline_bounds(&mdp, &dataset, delta, Some(master_seed + k as u64 + 1), true);

        let algos = vec!["V_Shaping", "Q_Shaping", "Count_Init_UCBVI"];
        for algo in algos {
            let results: Vec<(Vec<f64>, Vec<f64>)> = agent_seeds.par_iter().map(|&seed| {
                match algo {
                    "V_Shaping" => run_v_shaping(&mdp, &offline_bounds, t, delta, seed),
                    "Q_Shaping" => run_q_shaping(&mdp, &offline_bounds, t, delta, seed),
                    "Count_Init_UCBVI" => run_count_initialized_ucbvi(&mdp, &offline_bounds, t, delta, seed),
                    _ => unreachable!(),
                }
            }).collect();
            save_regret_data(&format!("{}/K={}/{}.dat", folder_name, k, algo), &results, t);
        }
    }

    println!("Generating plots...");
    if let Err(e) = plotting::plot_regret_curves(&folder_name, "V_Shaping", "rust_v_shaping.png") {
        eprintln!("Error plotting V_Shaping: {}", e);
    }
    if let Err(e) = plotting::plot_regret_curves(&folder_name, "Q_Shaping", "rust_q_shaping.png") {
        eprintln!("Error plotting Q_Shaping: {}", e);
    }
}

fn save_regret_data(path: &str, results: &[(Vec<f64>, Vec<f64>)], t: usize) {
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

    let num_points = 200.min(t);
    let mut file = File::create(path).unwrap();
    writeln!(file, "Episode CumulativeRegret StdDev").unwrap();

    for i in 0..num_points {
        let idx = (i * t) / num_points;
        if idx >= t { continue; }
        writeln!(file, "{} {:.6} {:.6}", idx + 1, mean[idx], std[idx]).unwrap();
    }
    println!("Saved {}", path);
}

fn run_mdp_trials(args: &Args, mode: &str) {
    let h = args.h;
    let s = args.s;
    let a = args.a;
    let t = args.t;
    let n_seeds = args.n_seeds;
    let delta = 0.05;
    let k = 60000;
    let num_points = 10;

    let folder_name = mode.to_string();
    fs::create_dir_all(&folder_name).unwrap();

    let x_values: Vec<f64> = if mode == "ExpandingReward" {
        (0..num_points).map(|i| 0.01 * (1.0 / 0.01 as f64).powf(i as f64 / (num_points - 1) as f64)).collect()
    } else {
        let width = 0.1;
        (0..num_points).map(|i| i as f64 * (1.0 - width) / (num_points - 1) as f64).collect()
    };

    let shaping_algos = vec!["Bonus_Shaping_Only", "Upper_Bonus_Shaping", "Count_Init_UCBVI"];
    let mut final_performance: std::collections::HashMap<String, Vec<(f64, f64)>> = std::collections::HashMap::new();
    for algo in &shaping_algos {
        final_performance.insert(algo.to_string(), Vec::new());
    }

    for &x in &x_values {
        let terminal_range = if mode == "ExpandingReward" {
            (1.0 - x, 1.0)
        } else {
            let width = 0.1;
            (x, x + width)
        };

        println!("Running for x={:.4}, R_term=({:.3}, {:.3})", x, terminal_range.0, terminal_range.1);
        
        let mdp_seed = (x * 100000.0) as u64;
        let mdp = TabularMDP::random(h, s, a, true, Some((0.0, 0.0)), Some(terminal_range), mdp_seed);
        
        let mut rng = rand::rngs::StdRng::seed_from_u64(mdp_seed + 1);
        let mut agent_seeds = Vec::new();
        for _ in 0..n_seeds {
            agent_seeds.push(rng.r#gen::<u64>());
        }

        let dataset = generate_dataset(&mdp, k, Some(mdp_seed + 2));
        let offline_bounds = compute_offline_bounds(&mdp, &dataset, delta, Some(mdp_seed + 3), true);

        let baseline_results: Vec<f64> = agent_seeds.par_iter().map(|&seed| {
            let (regrets, _) = run_standard_ucbvi(&mdp, t, delta, seed);
            regrets.iter().sum::<f64>()
        }).collect();

        for &algo in &shaping_algos {
            let algo_results: Vec<f64> = agent_seeds.par_iter().map(|&seed| {
                let (regrets, _) = match algo {
                    "Bonus_Shaping_Only" => run_bonus_shaping_only(&mdp, &offline_bounds, t, delta, seed),
                    "Upper_Bonus_Shaping" => run_upper_bonus_shaping(&mdp, &offline_bounds, t, delta, seed),
                    "Count_Init_UCBVI" => run_count_initialized_ucbvi(&mdp, &offline_bounds, t, delta, seed),
                    _ => unreachable!(),
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
        writeln!(file, "# X_Value Mean_Improvement Std_Improvement").unwrap();
        for (i, &x) in x_values.iter().enumerate() {
            let (mean, std) = stats[i];
            writeln!(file, "{:.8} {:.8} {:.8}", x, mean, std).unwrap();
        }
        println!("Saved {}", path);
    }

    println!("Generating plots...");
    if let Err(e) = plotting::plot_mdp_trials(&folder_name, mode, &format!("{}.png", mode), mode == "ExpandingReward") {
        eprintln!("Error plotting MDP trials ({}): {}", mode, e);
    }
}

fn run_convergence_experiment(args: &Args) {
    let h_total = args.h;
    let s_total = args.s;
    let a_total = args.a;
    let delta = 0.05;
    let master_seed = 42;
    let h_specific = 2; // Fixed as in python script

    let mdp = TabularMDP::random(h_total, s_total, a_total, true, None, None, master_seed);
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
    let num_points = 25;
    let mut k_values = Vec::new();
    for i in 0..num_points {
        let k_raw = k_min + i * (k_max - k_min) / (num_points - 1);
        let k = k_raw - (k_raw % h_total);
        if k > 0 && !k_values.contains(&k) {
            k_values.push(k);
        }
    }

    println!("Generating dataset (K_max={})...", k_max);
    let large_dataset = generate_dataset(&mdp, k_max, Some(master_seed + 1));

    let mut file = File::create("R_vs_D.dat").unwrap();
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
        let bounds = compute_offline_bounds(&mdp, subset, delta, Some(master_seed + k as u64), true);
        
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
    println!("\nSaved R_vs_D.dat");

    println!("Generating plots...");
    if let Err(e) = plotting::plot_convergence("R_vs_D.dat", "rust_R_vs_D.png") {
        eprintln!("Error plotting convergence: {}", e);
    }
}
