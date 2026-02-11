use rand::{SeedableRng, Rng};
use clap::Parser;
use rayon::prelude::*;
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;
use std::process::Command;
use value_envelopes::*;
use ndarray::prelude::*;

mod plotting;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value = "regret_curves")]
    experiment: String,

    #[arg(short = 'H', long, default_value_t = 4)]
    h: usize,

    #[arg(short, long, default_value_t = 12)]
    s: usize,

    #[arg(short, long, default_value_t = 3)]
    a: usize,

    #[arg(short, long, default_value_t = 10000)]
    t: usize,

    #[arg(short, long, default_value_t = 10)]
    n_seeds: usize,
}

fn main() {
    // Create output directories if they don't exist
    fs::create_dir_all("data").unwrap();
    fs::create_dir_all("png").unwrap();
    fs::create_dir_all("pdf").unwrap();
    fs::create_dir_all("tex files").unwrap();

    let args = Args::parse();

    match args.experiment.as_str() {
        "regret_curves" => run_regret_curves(&args),
        "expanding_reward" => run_mdp_trials(&args, "ExpandingReward"),
        "sliding_window" => run_mdp_trials(&args, "SlidingWindow_width0p10"),
        "convergence" => run_convergence_experiment(&args),
        "compile_only" => {
            if let Err(e) = compile_and_move_pdfs(&args) {
                eprintln!("Error compiling PDFs: {}", e);
            }
        }
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
    
    let folder_name = format!("data/RegretCurves_H{}_S{}_A{}_T{}_Layered", h, s, a, t);
    fs::create_dir_all(&folder_name).unwrap();

    let k_values = vec![20000, 40000, 80000]; // K values for plotting
    
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
    if let Err(e) = plotting::plot_regret_curves(&folder_name, "V_Shaping", "png/rust_v_shaping.png") {
        eprintln!("Error plotting V_Shaping: {}", e);
    }
    if let Err(e) = plotting::plot_regret_curves(&folder_name, "Q_Shaping", "png/rust_q_shaping.png") {
        eprintln!("Error plotting Q_Shaping: {}", e);
    }

    // Compile LaTeX files and move PDFs
    if let Err(e) = compile_and_move_pdfs(args) {
        eprintln!("Error compiling PDFs: {}", e);
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
    let k = 6000;
    let num_points = 5;

    let folder_name = format!("data/{}", mode);
    fs::create_dir_all(&folder_name).unwrap();

    let x_values: Vec<f64> = if mode == "ExpandingReward" {
        (0..num_points).map(|i| 0.01 * (1.0 / 0.01 as f64).powf(i as f64 / (num_points - 1) as f64)).collect()
    } else {
        let width = 0.1;
        (0..num_points).map(|i| i as f64 * (1.0 - width) / (num_points - 1) as f64).collect()
    };

    let shaping_algos = vec!["Bonus_Shaping_Onl    cargo run --release -- --experiment compile_onlyy", "Upper_Bonus_Shaping", "Count_Init_UCBVI"];
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
    if let Err(e) = plotting::plot_mdp_trials(&folder_name, mode, &format!("png/{}.png", mode), mode == "ExpandingReward") {
        eprintln!("Error plotting MDP trials ({}): {}", mode, e);
    }

    // Compile LaTeX files and move PDFs
    if let Err(e) = compile_and_move_pdfs(args) {
        eprintln!("Error compiling PDFs: {}", e);
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
    println!("\nSaved data/R_vs_D.dat");

    println!("Generating plots...");
    if let Err(e) = plotting::plot_convergence("data/R_vs_D.dat", "png/rust_R_vs_D.png") {
        eprintln!("Error plotting convergence: {}", e);
    }

    // Compile LaTeX files and move PDFs
    if let Err(e) = compile_and_move_pdfs(args) {
        eprintln!("Error compiling PDFs: {}", e);
    }
}

fn compile_and_move_pdfs(args: &Args) -> Result<(), Box<dyn std::error::Error>> {

    let tex_dir = Path::new("tex files");
    let pdf_dir = Path::new("pdf");

    let tex_files = match args.experiment.as_str() {
        "regret_curves" => vec!["plot_v_shaping_vs_K.tex", "plot_q_shaping_vs_K.tex"],
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
            "plot_v_shaping_vs_K.tex" => generate_plot_v_tex(args),
            "plot_q_shaping_vs_K.tex" => generate_plot_q_tex(args),
            "mdp_trials.tex" => generate_mdp_tex(args),
            "R_vs_D.tex" => generate_r_vs_d_tex(args),
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
            let target_pdf = pdf_dir.join(pdf_file.file_name().unwrap());
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

    Ok(())
}

fn generate_plot_v_tex(args: &Args) -> String {
    format!(r#"\documentclass[border=10pt]{{standalone}}
\usepackage{{tikz}}
\usepackage{{pgfplots}}
\pgfplotsset{{compat=1.17}}
\begin{{document}}
\begin{{tikzpicture}}
    \def\mainfolder{{data/RegretCurves_H{}_S{}_A{}_T{}_Layered}}
    \begin{{axis}}[
        title={{V-Shaping Performance vs. Offline Data Size (K)}},
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
    \addplot[
        black,
        dashed,
        thick,
        no marks
    ] table[x expr=\thisrow{{Episode}}/ {}, y=CumulativeRegret] {{\mainfolder/Standard_UCBVI.dat}};
    \addlegendentry{{Standard UCBVI}}
    \addplot[red, mark=square*] 
        table[x expr=\thisrow{{Episode}}/ {}, y=CumulativeRegret] {{\mainfolder/K=20000/V_Shaping.dat}};
    \addlegendentry{{V-Shaping (K=20k)}}
    \addplot[orange, mark=square*] 
        table[x expr=\thisrow{{Episode}}/ {}, y=CumulativeRegret] {{\mainfolder/K=40000/V_Shaping.dat}};
    \addlegendentry{{V-Shaping (K=40k)}}
    \addplot[blue, mark=square*] 
        table[x expr=\thisrow{{Episode}}/ {}, y=CumulativeRegret] {{\mainfolder/K=80000/V_Shaping.dat}};
    \addlegendentry{{V-Shaping (K=80k)}}
    \addplot[red, mark=o] 
        table[x expr=\thisrow{{Episode}}/ {}, y=CumulativeRegret] {{\mainfolder/K=20000/Count_Init_UCBVI.dat}};
    \addlegendentry{{Count-Init (K=20k)}}
    \addplot[orange, mark=o] 
        table[x expr=\thisrow{{Episode}}/ {}, y=CumulativeRegret] {{\mainfolder/K=40000/Count_Init_UCBVI.dat}};
    \addlegendentry{{Count-Init (K=40k)}}
    \addplot[blue, mark=o] 
        table[x expr=\thisrow{{Episode}}/ {}, y=CumulativeRegret] {{\mainfolder/K=80000/Count_Init_UCBVI.dat}};
    \addlegendentry{{Count-Init (K=80k)}}
    \end{{axis}}
\end{{tikzpicture}}
\end{{document}}"#, args.h, args.s, args.a, args.t, args.t, args.t, args.t, args.t, args.t, args.t, args.t, args.t)
}

fn generate_plot_q_tex(args: &Args) -> String {
    format!(r#"\documentclass[border=10pt]{{standalone}}
\usepackage{{tikz}}
\usepackage{{pgfplots}}
\pgfplotsset{{compat=1.17}}
\begin{{document}}
\begin{{tikzpicture}}
    \def\mainfolder{{data/RegretCurves_H{}_S{}_A{}_T{}_Layered}}
    \begin{{axis}}[
        title={{Q-Shaping Performance vs. Offline Data Size (K)}},
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
    \addplot[
        black,
        dashed,
        thick,
        no marks
    ] table[x expr=\thisrow{{Episode}}/ {}, y=CumulativeRegret] {{\mainfolder/Standard_UCBVI.dat}};
    \addlegendentry{{Standard UCBVI}}
    \addplot[red, mark=square*] 
        table[x expr=\thisrow{{Episode}}/ {}, y=CumulativeRegret] {{\mainfolder/K=20000/Q_Shaping.dat}};
    \addlegendentry{{Q-Shaping (K=20k)}}
    \addplot[orange, mark=square*] 
        table[x expr=\thisrow{{Episode}}/ {}, y=CumulativeRegret] {{\mainfolder/K=40000/Q_Shaping.dat}};
    \addlegendentry{{Q-Shaping (K=40k)}}
    \addplot[blue, mark=square*] 
        table[x expr=\thisrow{{Episode}}/ {}, y=CumulativeRegret] {{\mainfolder/K=80000/Q_Shaping.dat}};
    \addlegendentry{{Q-Shaping (K=80k)}}
    \addplot[red, mark=o] 
        table[x expr=\thisrow{{Episode}}/ {}, y=CumulativeRegret] {{\mainfolder/K=20000/Count_Init_UCBVI.dat}};
    \addlegendentry{{Count-Init (K=20k)}}
    \addplot[orange, mark=o] 
        table[x expr=\thisrow{{Episode}}/ {}, y=CumulativeRegret] {{\mainfolder/K=40000/Count_Init_UCBVI.dat}};
    \addlegendentry{{Count-Init (K=40k)}}
    \addplot[blue, mark=o] 
        table[x expr=\thisrow{{Episode}}/ {}, y=CumulativeRegret] {{\mainfolder/K=80000/Count_Init_UCBVI.dat}};
    \addlegendentry{{Count-Init (K=80k)}}
    \end{{axis}}
\end{{tikzpicture}}
\end{{document}}"#, args.h, args.s, args.a, args.t, args.t, args.t, args.t, args.t, args.t, args.t, args.t, args.t)
}

fn generate_mdp_tex(args: &Args) -> String {
    let (folder, title, xlabel, axis_type, extra_options) = if args.experiment == "expanding_reward" {
        ("ExpandingReward", "Expanding Reward: Bonus Shaping Comparison", r"Reward Range Width x", "semilogxaxis", "")
    } else {
        ("SlidingWindow_width0p10", "Sliding Window: Bonus Shaping Comparison", r"Reward Window Start Location ($x$) in $R_{\text{term}} = (x, x+0.1)$", "axis", "")
    };
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
    \addplot[purple, solid, thick, mark=square*]
        table {{\folderpath/Bonus_Shaping_Only.dat}};
    \addlegendentry{{Full-Bonus}}
    \addplot[green!70!black, solid, thick, mark=o]
        table {{\folderpath/Upper_Bonus_Shaping.dat}};
    \addlegendentry{{Upper-Bonus}}
    \addplot[blue, solid, thick, mark=triangle*]
        table {{\folderpath/Count_Init_UCBVI.dat}};
    \addlegendentry{{Count-Init}}
    \end{{{}}}
\end{{tikzpicture}}
\end{{standalonefigure}}
\end{{document}}"#, folder, axis_type, title, xlabel, axis_type)
}

fn generate_r_vs_d_tex(_args: &Args) -> String {
    r#"\documentclass[tikz]{standalone}
\usepackage{amsmath}
\usepackage{pgfplots}
\pgfplotsset{compat=1.17}
\begin{document}
\begin{tikzpicture}
\begin{axis}[
    width=12cm,
    height=8cm,
    title={Convergence of Bounds at h=2},
    xlabel={Offline Dataset Size (K trajectories)},
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
