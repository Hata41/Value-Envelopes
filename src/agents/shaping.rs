use ndarray::prelude::*;
use rand::prelude::*;
use rand::distributions::WeightedIndex;
use crate::mdp::{TabularMDP, evaluate_policy, value_iteration};
use crate::offline::OfflineBounds;

pub fn run_v_shaping(
    mdp: &TabularMDP,
    offline_bounds: &OfflineBounds,
    t: usize,
    delta: f64,
    seed: u64,
) -> (Vec<f64>, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let (v_opt, _) = value_iteration(mdp);
    let opt_global = mdp.rho.dot(&v_opt.slice(s![0, ..]));

    let mut n_sa = Array3::<i64>::zeros((mdp.h, mdp.s, mdp.a));
    let mut n_sas = Array4::<i64>::zeros((mdp.h, mdp.s, mdp.a, mdp.s));

    let l = ((8.0 * mdp.s as f64 * mdp.a as f64 * mdp.h as f64 * t as f64) / delta.max(1e-12)).ln();
    let c1 = 2.0;
    let c2 = 14.0 / 3.0;

    let mut regrets: Vec<f64> = Vec::with_capacity(t);
    let mut rewards = Vec::with_capacity(t);

    for _ in 0..t {
        let mut p_hat = Array4::<f64>::zeros((mdp.h, mdp.s, mdp.a, mdp.s));
        for hh_ in 0..mdp.h {
            for ss_ in 0..mdp.s {
                for aa_ in 0..mdp.a {
                    let n = n_sa[[hh_, ss_, aa_]];
                    if n == 0 {
                        p_hat.slice_mut(s![hh_, ss_, aa_, ..]).fill(1.0 / mdp.s as f64);
                    } else {
                        for ss_next in 0..mdp.s {
                            p_hat[[hh_, ss_, aa_, ss_next]] = n_sas[[hh_, ss_, aa_, ss_next]] as f64 / n as f64;
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
                        b = offline_bounds.r_next[hh];
                    } else {
                        let m_next_zeros;
                        let d_next_zeros;
                        let m_next = if hh + 1 < mdp.h { 
                            offline_bounds.m.slice(s![hh + 1, ..]) 
                        } else { 
                            m_next_zeros = Array1::zeros(mdp.s);
                            m_next_zeros.slice(s![..]) 
                        };
                        let d_next = if hh + 1 < mdp.h { 
                            offline_bounds.d.slice(s![hh + 1, ..]) 
                        } else { 
                            d_next_zeros = Array1::zeros(mdp.s);
                            d_next_zeros.slice(s![..]) 
                        };
                        
                        let mut mu_m = 0.0;
                        for ss_next in 0..mdp.s {
                            mu_m += p_hat[[hh, ss, aa, ss_next]] * m_next[ss_next];
                        }
                        let mut var_m = 0.0;
                        let mut e_d2 = 0.0;
                        for ss_next in 0..mdp.s {
                            let p = p_hat[[hh, ss, aa, ss_next]];
                            var_m += p * (m_next[ss_next] - mu_m).powi(2);
                            e_d2 += p * d_next[ss_next].powi(2);
                        }
                        
                        let sigma = var_m.max(0.0).sqrt() + 0.5 * e_d2.max(0.0).sqrt();
                        let b_calc = c1 * sigma * (l / n as f64).sqrt() + c2 * offline_bounds.r_next[hh] * (l / n as f64);
                        b = b_calc.min(offline_bounds.r_next[hh]);
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
                v_hat[[hh, ss]] = max_q.min(offline_bounds.u_values[[hh, ss]]);
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

    (regrets, rewards)
}

pub fn run_q_shaping(
    mdp: &TabularMDP,
    offline_bounds: &OfflineBounds,
    t: usize,
    delta: f64,
    seed: u64,
) -> (Vec<f64>, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let (v_opt, _) = value_iteration(mdp);
    let opt_global = mdp.rho.dot(&v_opt.slice(s![0, ..]));

    let mut n_sa = Array3::<i64>::zeros((mdp.h, mdp.s, mdp.a));
    let mut n_sas = Array4::<i64>::zeros((mdp.h, mdp.s, mdp.a, mdp.s));

    let l = ((4.0 * mdp.s as f64 * mdp.a as f64 * mdp.h as f64 * t as f64) / delta.max(1e-12)).ln();
    let c1 = 2.0;
    let c2 = 14.0 / 3.0;

    let mut regrets: Vec<f64> = Vec::with_capacity(t);
    let mut rewards = Vec::with_capacity(t);

    for _ in 0..t {
        let mut p_hat = Array4::<f64>::zeros((mdp.h, mdp.s, mdp.a, mdp.s));
        for hh_ in 0..mdp.h {
            for ss_ in 0..mdp.s {
                for aa_ in 0..mdp.a {
                    let n = n_sa[[hh_, ss_, aa_]];
                    if n == 0 {
                        p_hat.slice_mut(s![hh_, ss_, aa_, ..]).fill(1.0 / mdp.s as f64);
                    } else {
                        for ss_next in 0..mdp.s {
                            p_hat[[hh_, ss_, aa_, ss_next]] = n_sas[[hh_, ss_, aa_, ss_next]] as f64 / n as f64;
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
                        b = offline_bounds.r_next[hh];
                    } else {
                        let m_next_zeros;
                        let d_next_zeros;
                        let m_next = if hh + 1 < mdp.h { 
                            offline_bounds.m.slice(s![hh + 1, ..]) 
                        } else { 
                            m_next_zeros = Array1::zeros(mdp.s);
                            m_next_zeros.slice(s![..]) 
                        };
                        let d_next = if hh + 1 < mdp.h { 
                            offline_bounds.d.slice(s![hh + 1, ..]) 
                        } else { 
                            d_next_zeros = Array1::zeros(mdp.s);
                            d_next_zeros.slice(s![..]) 
                        };
                        
                        let mut mu_m = 0.0;
                        for ss_next in 0..mdp.s {
                            mu_m += p_hat[[hh, ss, aa, ss_next]] * m_next[ss_next];
                        }
                        let mut var_m = 0.0;
                        let mut e_d2 = 0.0;
                        for ss_next in 0..mdp.s {
                            let p = p_hat[[hh, ss, aa, ss_next]];
                            var_m += p * (m_next[ss_next] - mu_m).powi(2);
                            e_d2 += p * d_next[ss_next].powi(2);
                        }
                        
                        let sigma = var_m.max(0.0).sqrt() + 0.5 * e_d2.max(0.0).sqrt();
                        let b_calc = c1 * sigma * (l / n as f64).sqrt() + c2 * offline_bounds.r_next[hh] * (l / n as f64);
                        b = b_calc.min(offline_bounds.r_next[hh]);
                    }
                    let mut expected_v = 0.0;
                    for ss_next in 0..mdp.s {
                        expected_v += p_hat[[hh, ss, aa, ss_next]] * v_hat[[hh + 1, ss_next]];
                    }
                    let q_raw = mdp.r[[hh, ss, aa]] + expected_v + b;
                    q_hat[[hh, ss, aa]] = q_raw.min(offline_bounds.u_q[[hh, ss, aa]]);
                }
                let mut max_q = -f64::INFINITY;
                for aa in 0..mdp.a {
                    if q_hat[[hh, ss, aa]] > max_q { max_q = q_hat[[hh, ss, aa]]; }
                }
                v_hat[[hh, ss]] = max_q;
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

    (regrets, rewards)
}

pub fn run_bonus_shaping_only(
    mdp: &TabularMDP,
    offline_bounds: &OfflineBounds,
    t: usize,
    delta: f64,
    seed: u64,
) -> (Vec<f64>, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let (v_opt, _) = value_iteration(mdp);
    let opt_global = mdp.rho.dot(&v_opt.slice(s![0, ..]));
    let mut n_sa = Array3::<i64>::zeros((mdp.h, mdp.s, mdp.a));
    let mut n_sas = Array4::<i64>::zeros((mdp.h, mdp.s, mdp.a, mdp.s));
    let l = ((8.0 * mdp.s as f64 * mdp.a as f64 * mdp.h as f64 * t as f64) / delta.max(1e-12)).ln();
    let c1 = 2.0;
    let c2 = 14.0 / 3.0;
    let mut regrets: Vec<f64> = Vec::with_capacity(t);
    for _ in 0..t {
        let mut p_hat = Array4::<f64>::zeros((mdp.h, mdp.s, mdp.a, mdp.s));
        for hh_ in 0..mdp.h {
            for ss_ in 0..mdp.s {
                for aa_ in 0..mdp.a {
                    let n = n_sa[[hh_, ss_, aa_]];
                    if n == 0 {
                        p_hat.slice_mut(s![hh_, ss_, aa_, ..]).fill(1.0 / mdp.s as f64);
                    } else {
                        for ss_next in 0..mdp.s {
                            p_hat[[hh_, ss_, aa_, ss_next]] = n_sas[[hh_, ss_, aa_, ss_next]] as f64 / n as f64;
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
                        b = offline_bounds.r_next[hh];
                    } else {
                        let m_next_zeros;
                        let d_next_zeros;
                        let m_next = if hh + 1 < mdp.h { offline_bounds.m.slice(s![hh + 1, ..]) } else { m_next_zeros = Array1::zeros(mdp.s); m_next_zeros.slice(s![..]) };
                        let d_next = if hh + 1 < mdp.h { offline_bounds.d.slice(s![hh + 1, ..]) } else { d_next_zeros = Array1::zeros(mdp.s); d_next_zeros.slice(s![..]) };
                        let mut mu_m = 0.0;
                        for ss_next in 0..mdp.s { mu_m += p_hat[[hh, ss, aa, ss_next]] * m_next[ss_next]; }
                        let mut var_m = 0.0;
                        let mut e_d2 = 0.0;
                        for ss_next in 0..mdp.s {
                            let p = p_hat[[hh, ss, aa, ss_next]];
                            var_m += p * (m_next[ss_next] - mu_m).powi(2);
                            e_d2 += p * d_next[ss_next].powi(2);
                        }
                        let sigma = var_m.max(0.0).sqrt() + 0.5 * e_d2.max(0.0).sqrt();
                        let b_calc = c1 * sigma * (l / n as f64).sqrt() + c2 * offline_bounds.r_next[hh] * (l / n as f64);
                        b = b_calc.min(offline_bounds.r_next[hh]);
                    }
                    let mut expected_v = 0.0;
                    for ss_next in 0..mdp.s { expected_v += p_hat[[hh, ss, aa, ss_next]] * v_hat[[hh + 1, ss_next]]; }
                    q_hat[[hh, ss, aa]] = mdp.r[[hh, ss, aa]] + expected_v + b;
                }
                let mut max_q = -f64::INFINITY;
                for aa in 0..mdp.a { if q_hat[[hh, ss, aa]] > max_q { max_q = q_hat[[hh, ss, aa]]; } }
                v_hat[[hh, ss]] = max_q.min((mdp.h - hh) as f64);
            }
        }
        let mut policy = Array2::<usize>::zeros((mdp.h, mdp.s));
        for hh in 0..mdp.h {
            for ss in 0..mdp.s {
                let mut best_q = -f64::INFINITY;
                let mut best_a = 0;
                for aa in 0..mdp.a {
                    if q_hat[[hh, ss, aa]] > best_q { best_q = q_hat[[hh, ss, aa]]; best_a = aa; }
                }
                policy[[hh, ss]] = best_a;
            }
        }
        let v_pi = evaluate_policy(mdp, &policy);
        let expected_return = mdp.rho.dot(&v_pi.slice(s![0, ..]));
        regrets.push(opt_global - expected_return);
        let mut ss_current = WeightedIndex::new(mdp.rho.to_vec()).unwrap().sample(&mut rng);
        for hh in 0..mdp.h {
            let aa = policy[[hh, ss_current]];
            let ss_next = WeightedIndex::new(mdp.p.slice(s![hh, ss_current, aa, ..]).to_vec()).unwrap().sample(&mut rng);
            n_sa[[hh, ss_current, aa]] += 1;
            n_sas[[hh, ss_current, aa, ss_next]] += 1;
            ss_current = ss_next;
        }
    }
    (regrets, Vec::new())
}

pub fn run_upper_bonus_shaping(
    mdp: &TabularMDP,
    offline_bounds: &OfflineBounds,
    t: usize,
    delta: f64,
    seed: u64,
) -> (Vec<f64>, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let (v_opt, _) = value_iteration(mdp);
    let opt_global = mdp.rho.dot(&v_opt.slice(s![0, ..]));
    let mut n_sa = Array3::<i64>::zeros((mdp.h, mdp.s, mdp.a));
    let mut n_sas = Array4::<i64>::zeros((mdp.h, mdp.s, mdp.a, mdp.s));
    let l = ((8.0 * mdp.s as f64 * mdp.a as f64 * mdp.h as f64 * t as f64) / delta.max(1e-12)).ln();
    let c1 = 2.0;
    let c2 = 14.0 / 3.0;
    let m_upper = offline_bounds.u_values.mapv(|u| 0.5 * u);
    let d_upper = offline_bounds.u_values.clone();
    let mut r_next_upper = Array1::<f64>::zeros(mdp.h);
    for hh in 0..mdp.h {
        if hh + 1 < mdp.h {
            let u_next = offline_bounds.u_values.slice(s![hh + 1, ..]);
            let mut max_u = -f64::INFINITY;
            for &val in u_next { if val > max_u { max_u = val; } }
            r_next_upper[hh] = max_u;
        }
    }
    let mut regrets: Vec<f64> = Vec::with_capacity(t);
    for _ in 0..t {
        let mut p_hat = Array4::<f64>::zeros((mdp.h, mdp.s, mdp.a, mdp.s));
        for hh_ in 0..mdp.h {
            for ss_ in 0..mdp.s {
                for aa_ in 0..mdp.a {
                    let n = n_sa[[hh_, ss_, aa_]];
                    if n == 0 { p_hat.slice_mut(s![hh_, ss_, aa_, ..]).fill(1.0 / mdp.s as f64); }
                    else {
                        for ss_next in 0..mdp.s { p_hat[[hh_, ss_, aa_, ss_next]] = n_sas[[hh_, ss_, aa_, ss_next]] as f64 / n as f64; }
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
                    if n <= 1 { b = r_next_upper[hh]; }
                    else {
                        let m_next_zeros;
                        let d_next_zeros;
                        let m_next = if hh + 1 < mdp.h { m_upper.slice(s![hh + 1, ..]) } else { m_next_zeros = Array1::zeros(mdp.s); m_next_zeros.slice(s![..]) };
                        let d_next = if hh + 1 < mdp.h { d_upper.slice(s![hh + 1, ..]) } else { d_next_zeros = Array1::zeros(mdp.s); d_next_zeros.slice(s![..]) };
                        let mut mu_m = 0.0;
                        for ss_next in 0..mdp.s { mu_m += p_hat[[hh, ss, aa, ss_next]] * m_next[ss_next]; }
                        let mut var_m = 0.0;
                        let mut e_d2 = 0.0;
                        for ss_next in 0..mdp.s {
                            let p = p_hat[[hh, ss, aa, ss_next]];
                            var_m += p * (m_next[ss_next] - mu_m).powi(2);
                            e_d2 += p * d_next[ss_next].powi(2);
                        }
                        let sigma = var_m.max(0.0).sqrt() + 0.5 * e_d2.max(0.0).sqrt();
                        let b_calc = c1 * sigma * (l / n as f64).sqrt() + c2 * r_next_upper[hh] * (l / n as f64);
                        b = b_calc.min(r_next_upper[hh]);
                    }
                    let mut expected_v = 0.0;
                    for ss_next in 0..mdp.s { expected_v += p_hat[[hh, ss, aa, ss_next]] * v_hat[[hh + 1, ss_next]]; }
                    q_hat[[hh, ss, aa]] = mdp.r[[hh, ss, aa]] + expected_v + b;
                }
                v_hat[[hh, ss]] = q_hat.slice(s![hh, ss, ..]).iter().fold(-f64::INFINITY, |a, &b| a.max(b)).min((mdp.h - hh) as f64);
            }
        }
        let mut policy = Array2::<usize>::zeros((mdp.h, mdp.s));
        for hh in 0..mdp.h {
            for ss in 0..mdp.s {
                let mut best_q = -f64::INFINITY;
                let mut best_a = 0;
                for aa in 0..mdp.a {
                    if q_hat[[hh, ss, aa]] > best_q { best_q = q_hat[[hh, ss, aa]]; best_a = aa; }
                }
                policy[[hh, ss]] = best_a;
            }
        }
        let v_pi = evaluate_policy(mdp, &policy);
        regrets.push(opt_global - mdp.rho.dot(&v_pi.slice(s![0, ..])));
        let mut ss_current = WeightedIndex::new(mdp.rho.to_vec()).unwrap().sample(&mut rng);
        for hh in 0..mdp.h {
            let aa = policy[[hh, ss_current]];
            let ss_next = WeightedIndex::new(mdp.p.slice(s![hh, ss_current, aa, ..]).to_vec()).unwrap().sample(&mut rng);
            n_sa[[hh, ss_current, aa]] += 1;
            n_sas[[hh, ss_current, aa, ss_next]] += 1;
            ss_current = ss_next;
        }
    }
    (regrets, Vec::new())
}
