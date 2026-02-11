use ndarray::prelude::*;
use rand::prelude::*;
use rand::distributions::WeightedIndex;
use std::io::{self, Write};
use std::time::Instant;
use crate::mdp::{TabularMDP, evaluate_policy, value_iteration};
use crate::offline::OfflineBounds;

pub fn run_standard_ucbvi(
    mdp: &TabularMDP,
    t: usize,
    delta: f64,
    seed: u64,
    show_progress: bool,
) -> (Vec<f64>, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let (v_opt, _) = value_iteration(mdp);
    let opt_global = mdp.rho.dot(&v_opt.slice(s![0, ..]));

    let mut n_sa = Array3::<i64>::zeros((mdp.h, mdp.s, mdp.a));
    let mut n_sas = Array4::<i64>::zeros((mdp.h, mdp.s, mdp.a, mdp.s));

    let l = ((4.0 * mdp.s as f64 * mdp.a as f64 * mdp.h as f64 * t as f64) / delta.max(1e-12)).ln();
    let c1 = 2.0;
    let c2 = 14.0 / 3.0;

    let r_next_base = Array1::from_shape_fn(mdp.h, |hh| {
        if hh + 1 < mdp.h { (mdp.h - (hh + 1)) as f64 } else { 0.0 }
    });

    let mut regrets: Vec<f64> = Vec::with_capacity(t);
    let mut rewards = Vec::with_capacity(t);

    let start_time = Instant::now();

    for episode in 0..t {
        if show_progress && episode % 100 == 0 {
            let elapsed = start_time.elapsed();
            let progress = episode as f64 / t as f64;
            let percentage = progress * 100.0;
            if progress > 0.0 {
                let total_estimated = elapsed.div_f64(progress);
                let remaining = total_estimated - elapsed;
                print!("\rAgent Progress: {}/{} ({:.1}%) ETA: {:.1}s", episode, t, percentage, remaining.as_secs_f64());
            } else {
                print!("\rAgent Progress: {}/{} ({:.1}%)", episode, t, percentage);
            }
            io::stdout().flush().unwrap();
        }
        let mut p_hat = Array4::<f64>::zeros((mdp.h, mdp.s, mdp.a, mdp.s));
        for hh in 0..mdp.h {
            for ss in 0..mdp.s {
                for aa in 0..mdp.a {
                    let n = n_sa[[hh, ss, aa]];
                    if n == 0 {
                        p_hat.slice_mut(s![hh, ss, aa, ..]).fill(1.0 / mdp.s as f64);
                    } else {
                        for ss_next in 0..mdp.s {
                            p_hat[[hh, ss, aa, ss_next]] = n_sas[[hh, ss, aa, ss_next]] as f64 / n as f64;
                        }
                    }
                }
            }
        }

        let mut q_hat = Array3::<f64>::zeros((mdp.h, mdp.s, mdp.a));
        let mut v_hat = Array2::<f64>::zeros((mdp.h + 1, mdp.s));

        for hh in (0..mdp.h).rev() {
            for ss in 0..mdp.s {
                for aa in 0..mdp.a {
                    let n = n_sa[[hh, ss, aa]];
                    let b;
                    if n <= 1 {
                        b = r_next_base[hh];
                    } else {
                        let sigma = 0.5 * r_next_base[hh];
                        let b_calc = c1 * sigma * (l / n as f64).sqrt() + c2 * r_next_base[hh] * (l / n as f64);
                        b = b_calc.min(r_next_base[hh]);
                    }
                    let mut expected_v = 0.0;
                    for ss_next in 0..mdp.s {
                        expected_v += p_hat[[hh, ss, aa, ss_next]] * v_hat[[hh + 1, ss_next]];
                    }
                    q_hat[[hh, ss, aa]] = mdp.r[[hh, ss, aa]] + expected_v + b;
                }
                let mut max_q = -f64::INFINITY;
                for aa in 0..mdp.a {
                    if q_hat[[hh, ss, aa]] > max_q { max_q = q_hat[[hh, ss, aa]]; }
                }
                v_hat[[hh, ss]] = max_q.min((mdp.h - hh) as f64);
            }
        }

        let mut policy = Array2::<usize>::zeros((mdp.h, mdp.s));
        for hh in 0..mdp.h {
            for ss in 0..mdp.s {
                let mut best_q = -f64::INFINITY;
                let mut best_a = 0;
                for aa in 0..mdp.a {
                    if q_hat[[hh, ss, aa]] > best_q {
                        best_q = q_hat[[hh, ss, aa]];
                        best_a = aa;
                    }
                }
                policy[[hh, ss]] = best_a;
            }
        }

        let v_pi = evaluate_policy(mdp, &policy);
        let expected_return = mdp.rho.dot(&v_pi.slice(s![0, ..]));
        regrets.push(opt_global - expected_return);

        let mut ss_current = WeightedIndex::new(mdp.rho.to_vec()).unwrap().sample(&mut rng);
        let mut ep_reward = 0.0;
        for hh in 0..mdp.h {
            let aa = policy[[hh, ss_current]];
            let ss_next = WeightedIndex::new(mdp.p.slice(s![hh, ss_current, aa, ..]).to_vec()).unwrap().sample(&mut rng);
            
            n_sa[[hh, ss_current, aa]] += 1;
            n_sas[[hh, ss_current, aa, ss_next]] += 1;
            ep_reward += mdp.r[[hh, ss_current, aa]];
            ss_current = ss_next;
        }
        rewards.push(ep_reward);
    }

    if show_progress {
        println!();
    }

    (regrets, rewards)
}

pub fn run_standard_ucbvi_bernstein(
    mdp: &TabularMDP,
    t: usize,
    delta: f64,
    seed: u64,
    show_progress: bool,
) -> (Vec<f64>, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let (v_opt, _) = value_iteration(mdp);
    let opt_global = mdp.rho.dot(&v_opt.slice(s![0, ..]));

    let mut n_sa = Array3::<i64>::zeros((mdp.h, mdp.s, mdp.a));
    let mut n_sas = Array4::<i64>::zeros((mdp.h, mdp.s, mdp.a, mdp.s));

    let l = ((4.0 * mdp.s as f64 * mdp.a as f64 * mdp.h as f64 * t as f64) / delta.max(1e-12)).ln();
    // Constants from Azar et al. (2017)
    let c1 = 8.0f64.sqrt(); // approx 2.828
    let c2 = 14.0 / 3.0;

    let r_next_base = Array1::from_shape_fn(mdp.h, |hh| {
        if hh + 1 < mdp.h { (mdp.h - (hh + 1)) as f64 } else { 0.0 }
    });

    let mut regrets: Vec<f64> = Vec::with_capacity(t);
    let mut rewards = Vec::with_capacity(t);

    let start_time = Instant::now();

    for episode in 0..t {
        if show_progress && episode % 100 == 0 {
            let elapsed = start_time.elapsed();
            let progress = episode as f64 / t as f64;
            let percentage = progress * 100.0;
            if progress > 0.0 {
                let total_estimated = elapsed.div_f64(progress);
                let remaining = total_estimated - elapsed;
                print!("\rAgent Progress: {}/{} ({:.1}%) ETA: {:.1}s", episode, t, percentage, remaining.as_secs_f64());
            } else {
                print!("\rAgent Progress: {}/{} ({:.1}%)", episode, t, percentage);
            }
            io::stdout().flush().unwrap();
        }
        let mut p_hat = Array4::<f64>::zeros((mdp.h, mdp.s, mdp.a, mdp.s));
        for hh in 0..mdp.h {
            for ss in 0..mdp.s {
                for aa in 0..mdp.a {
                    let n = n_sa[[hh, ss, aa]];
                    if n == 0 {
                        p_hat.slice_mut(s![hh, ss, aa, ..]).fill(1.0 / mdp.s as f64);
                    } else {
                        for ss_next in 0..mdp.s {
                            p_hat[[hh, ss, aa, ss_next]] = n_sas[[hh, ss, aa, ss_next]] as f64 / n as f64;
                        }
                    }
                }
            }
        }

        let mut q_hat = Array3::<f64>::zeros((mdp.h, mdp.s, mdp.a));
        let mut v_hat = Array2::<f64>::zeros((mdp.h + 1, mdp.s));

        for hh in (0..mdp.h).rev() {
            for ss in 0..mdp.s {
                for aa in 0..mdp.a {
                    let n = n_sa[[hh, ss, aa]];
                    let b;
                    if n <= 1 {
                        b = r_next_base[hh];
                    } else {
                        // Calculate empirical variance of V_{h+1}
                        // Var(X) = E[X^2] - (E[X])^2
                        let mut expected_v = 0.0;
                        let mut expected_v_sq = 0.0;
                        for ss_next in 0..mdp.s {
                            let val = v_hat[[hh + 1, ss_next]];
                            let prob = p_hat[[hh, ss, aa, ss_next]];
                            expected_v += prob * val;
                            expected_v_sq += prob * val * val;
                        }
                        let variance = (expected_v_sq - expected_v * expected_v).max(0.0);
                        
                        let b_calc = c1 * variance.sqrt() * (l / n as f64).sqrt() + c2 * r_next_base[hh] * (l / n as f64);
                        b = b_calc.min(r_next_base[hh]);
                    }
                    
                    let mut expected_v = 0.0;
                    for ss_next in 0..mdp.s {
                        expected_v += p_hat[[hh, ss, aa, ss_next]] * v_hat[[hh + 1, ss_next]];
                    }
                    
                    q_hat[[hh, ss, aa]] = mdp.r[[hh, ss, aa]] + expected_v + b;
                }
                let mut max_q = -f64::INFINITY;
                for aa in 0..mdp.a {
                    if q_hat[[hh, ss, aa]] > max_q { max_q = q_hat[[hh, ss, aa]]; }
                }
                v_hat[[hh, ss]] = max_q.min((mdp.h - hh) as f64);
            }
        }

        let mut policy = Array2::<usize>::zeros((mdp.h, mdp.s));
        for hh in 0..mdp.h {
            for ss in 0..mdp.s {
                let mut best_q = -f64::INFINITY;
                let mut best_a = 0;
                for aa in 0..mdp.a {
                    if q_hat[[hh, ss, aa]] > best_q {
                        best_q = q_hat[[hh, ss, aa]];
                        best_a = aa;
                    }
                }
                policy[[hh, ss]] = best_a;
            }
        }

        let v_pi = evaluate_policy(mdp, &policy);
        let expected_return = mdp.rho.dot(&v_pi.slice(s![0, ..]));
        regrets.push(opt_global - expected_return);

        let mut ss_current = WeightedIndex::new(mdp.rho.to_vec()).unwrap().sample(&mut rng);
        let mut ep_reward = 0.0;
        for hh in 0..mdp.h {
            let aa = policy[[hh, ss_current]];
            let ss_next = WeightedIndex::new(mdp.p.slice(s![hh, ss_current, aa, ..]).to_vec()).unwrap().sample(&mut rng);
            
            n_sa[[hh, ss_current, aa]] += 1;
            n_sas[[hh, ss_current, aa, ss_next]] += 1;
            ep_reward += mdp.r[[hh, ss_current, aa]];
            ss_current = ss_next;
        }
        rewards.push(ep_reward);
    }

    if show_progress {
        println!();
    }

    (regrets, rewards)
}

pub fn run_count_initialized_ucbvi(
    mdp: &TabularMDP,
    offline_bounds: &OfflineBounds,
    t: usize,
    delta: f64,
    seed: u64,
    use_bernstein: bool,
    show_progress: bool,
) -> (Vec<f64>, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let (v_opt, _) = value_iteration(mdp);
    let opt_global = mdp.rho.dot(&v_opt.slice(s![0, ..]));

    let mut n_sa = offline_bounds.n_sa.clone();
    let mut n_sas = offline_bounds.n_sas.clone();

    let l = ((4.0 * mdp.s as f64 * mdp.a as f64 * mdp.h as f64 * t as f64) / delta.max(1e-12)).ln();
    
    // Constants setup
    let (c1, c2) = if use_bernstein {
        (8.0f64.sqrt(), 14.0 / 3.0)
    } else {
        (2.0, 14.0 / 3.0)
    };

    let r_next_base = Array1::from_shape_fn(mdp.h, |hh| {
        if hh + 1 < mdp.h { (mdp.h - (hh + 1)) as f64 } else { 0.0 }
    });

    let mut regrets: Vec<f64> = Vec::with_capacity(t);
    let mut rewards = Vec::with_capacity(t);

    let start_time = Instant::now();

    for episode in 0..t {
        if show_progress && episode % 100 == 0 {
            let elapsed = start_time.elapsed();
            let progress = episode as f64 / t as f64;
            let percentage = progress * 100.0;
            if progress > 0.0 {
                let total_estimated = elapsed.div_f64(progress);
                let remaining = total_estimated - elapsed;
                print!("\rAgent Progress: {}/{} ({:.1}%) ETA: {:.1}s", episode, t, percentage, remaining.as_secs_f64());
            } else {
                print!("\rAgent Progress: {}/{} ({:.1}%)", episode, t, percentage);
            }
            io::stdout().flush().unwrap();
        }
        let mut p_hat = Array4::<f64>::zeros((mdp.h, mdp.s, mdp.a, mdp.s));
        for hh in 0..mdp.h {
            for ss in 0..mdp.s {
                for aa in 0..mdp.a {
                    let n = n_sa[[hh, ss, aa]];
                    if n == 0 {
                        p_hat.slice_mut(s![hh, ss, aa, ..]).fill(1.0 / mdp.s as f64);
                    } else {
                        for ss_next in 0..mdp.s {
                            p_hat[[hh, ss, aa, ss_next]] = n_sas[[hh, ss, aa, ss_next]] as f64 / n as f64;
                        }
                    }
                }
            }
        }

        let mut q_hat = Array3::<f64>::zeros((mdp.h, mdp.s, mdp.a));
        let mut v_hat = Array2::<f64>::zeros((mdp.h + 1, mdp.s));

        for hh in (0..mdp.h).rev() {
            for ss in 0..mdp.s {
                for aa in 0..mdp.a {
                    let n = n_sa[[hh, ss, aa]];
                    let b;
                    if n <= 1 {
                        b = r_next_base[hh];
                    } else {
                        let sigma = if use_bernstein {
                             // Empirical variance of V_{h+1}
                            let mut expected_v = 0.0;
                            let mut expected_v_sq = 0.0;
                            for ss_next in 0..mdp.s {
                                let val = v_hat[[hh + 1, ss_next]];
                                let prob = p_hat[[hh, ss, aa, ss_next]];
                                expected_v += prob * val;
                                expected_v_sq += prob * val * val;
                            }
                            (expected_v_sq - expected_v * expected_v).max(0.0).sqrt()
                        } else {
                            0.5 * r_next_base[hh]
                        };

                        let b_calc = c1 * sigma * (l / n as f64).sqrt() + c2 * r_next_base[hh] * (l / n as f64);
                        b = b_calc.min(r_next_base[hh]);
                    }
                    let mut expected_v = 0.0;
                    for ss_next in 0..mdp.s {
                        expected_v += p_hat[[hh, ss, aa, ss_next]] * v_hat[[hh + 1, ss_next]];
                    }
                    q_hat[[hh, ss, aa]] = mdp.r[[hh, ss, aa]] + expected_v + b;
                }
                let mut max_q = -f64::INFINITY;
                for aa in 0..mdp.a {
                    if q_hat[[hh, ss, aa]] > max_q { max_q = q_hat[[hh, ss, aa]]; }
                }
                v_hat[[hh, ss]] = max_q.min((mdp.h - hh) as f64);
            }
        }

        let mut policy = Array2::<usize>::zeros((mdp.h, mdp.s));
        for hh in 0..mdp.h {
            for ss in 0..mdp.s {
                let mut best_q = -f64::INFINITY;
                let mut best_a = 0;
                for aa in 0..mdp.a {
                    if q_hat[[hh, ss, aa]] > best_q {
                        best_q = q_hat[[hh, ss, aa]];
                        best_a = aa;
                    }
                }
                policy[[hh, ss]] = best_a;
            }
        }

        let v_pi = evaluate_policy(mdp, &policy);
        let expected_return = mdp.rho.dot(&v_pi.slice(s![0, ..]));

        regrets.push(opt_global - expected_return);

        let mut ss_current = WeightedIndex::new(mdp.rho.to_vec()).unwrap().sample(&mut rng);
        let mut ep_reward = 0.0;
        for hh in 0..mdp.h {
            let aa = policy[[hh, ss_current]];
            let ss_next = WeightedIndex::new(mdp.p.slice(s![hh, ss_current, aa, ..]).to_vec()).unwrap().sample(&mut rng);
            
            n_sa[[hh, ss_current, aa]] += 1;
            n_sas[[hh, ss_current, aa, ss_next]] += 1;
            ep_reward += mdp.r[[hh, ss_current, aa]];
            ss_current = ss_next;
        }
        rewards.push(ep_reward);
    }

    if show_progress {
        println!();
    }

    (regrets, rewards)
}
