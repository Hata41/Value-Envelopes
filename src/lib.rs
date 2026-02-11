use ndarray::prelude::*;
use ndarray_rand::rand_distr::{Uniform, Distribution};
use ndarray_rand::RandomExt;
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

pub fn value_iteration(mdp: &TabularMDP) -> (Array2<f64>, Array2<usize>) {
    let h = mdp.h;
    let s = mdp.s;
    let a = mdp.a;
    let mut v = Array2::<f64>::zeros((h + 1, s));
    let mut pi = Array2::<usize>::zeros((h, s));

    for hh in (0..h).rev() {
        for ss in 0..s {
            let mut best_q = -f64::INFINITY;
            let mut best_a = 0;
            for aa in 0..a {
                let mut expected_v = 0.0;
                for ss_next in 0..s {
                    expected_v += mdp.p[[hh, ss, aa, ss_next]] * v[[hh + 1, ss_next]];
                }
                let q_val = mdp.r[[hh, ss, aa]] + expected_v;
                if q_val > best_q {
                    best_q = q_val;
                    best_a = aa;
                }
            }
            v[[hh, ss]] = best_q;
            pi[[hh, ss]] = best_a;
        }
    }
    (v, pi)
}

pub fn evaluate_policy(mdp: &TabularMDP, policy: &Array2<usize>) -> Array2<f64> {
    let h = mdp.h;
    let s = mdp.s;
    let mut v_pi = Array2::<f64>::zeros((h + 1, s));

    for hh in (0..h).rev() {
        for ss in 0..s {
            let aa = policy[[hh, ss]];
            let mut expected_v = 0.0;
            for ss_next in 0..s {
                expected_v += mdp.p[[hh, ss, aa, ss_next]] * v_pi[[hh + 1, ss_next]];
            }
            v_pi[[hh, ss]] = mdp.r[[hh, ss, aa]] + expected_v;
        }
    }
    v_pi
}

pub fn generate_dataset(
    mdp: &TabularMDP,
    k: usize,
    seed: Option<u64>,
) -> Vec<(Vec<usize>, Vec<usize>, Vec<f64>)> {
    let mut rng = if let Some(s) = seed { StdRng::seed_from_u64(s) } else { StdRng::from_entropy() };
    let mut trajectories = Vec::with_capacity(k);

    for _ in 0..k {
        let mut states = Vec::with_capacity(mdp.h + 1);
        let mut actions = Vec::with_capacity(mdp.h);
        let mut rewards = Vec::with_capacity(mdp.h);

        // Reset
        let mut ss = WeightedIndex::new(mdp.rho.to_vec()).unwrap().sample(&mut rng);
        states.push(ss);

        for hh in 0..mdp.h {
            let aa = rng.gen_range(0..mdp.a);
            let reward = mdp.r[[hh, ss, aa]];
            let ss_next = WeightedIndex::new(mdp.p.slice(s![hh, ss, aa, ..]).to_vec()).unwrap().sample(&mut rng);

            actions.push(aa);
            rewards.push(reward);
            ss = ss_next;
            states.push(ss);
        }
        trajectories.push((states, actions, rewards));
    }
    trajectories
}

#[derive(Clone, Serialize, Deserialize)]
pub struct TabularMDP {
    pub h: usize,
    pub s: usize,
    pub a: usize,
    pub p: Array4<f64>, // [H, S, A, S]
    pub r: Array3<f64>, // [H, S, A]
    pub rho: Array1<f64>, // [S]
}

impl TabularMDP {
    pub fn new(h: usize, s: usize, a: usize, p: Array4<f64>, r: Array3<f64>, rho: Option<Array1<f64>>) -> Self {
        let mut p_norm = p.mapv(|x| x.max(0.0));
        for hh in 0..h {
            for ss in 0..s {
                for aa in 0..a {
                    let mut row = p_norm.slice_mut(s![hh, ss, aa, ..]);
                    let sum: f64 = row.sum();
                    if sum <= 0.0 {
                        row.fill(1.0 / s as f64);
                    } else {
                        row.mapv_inplace(|x| x / sum);
                    }
                }
            }
        }

        let r_clip = r.mapv(|x| x.clamp(0.0, 1.0));
        
        let rho_norm = if let Some(r_in) = rho {
            let r_clip = r_in.mapv(|x| x.max(0.0));
            let total = r_clip.sum();
            if total <= 0.0 {
                Array1::from_elem(s, 1.0 / s as f64)
            } else {
                r_clip / total
            }
        } else {
            Array1::from_elem(s, 1.0 / s as f64)
        };

        Self { h, s, a, p: p_norm, r: r_clip, rho: rho_norm }
    }

    pub fn random(
        h: usize,
        s: usize,
        a: usize,
        layered: bool,
        intermediate_reward_range: Option<(f64, f64)>,
        terminal_reward_range: Option<(f64, f64)>,
        seed: u64,
    ) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut p = Array4::zeros((h, s, a, s));
        let mut rho = Array1::zeros(s);

        if layered {
            assert_eq!(s % h, 0, "For layered MDP, S must be divisible by H");
            let s_layer = s / h;
            for hh in 0..h {
                let next_layer_start = ((hh + 1) % h) * s_layer;
                for ss in (hh * s_layer)..((hh + 1) * s_layer) {
                    for aa in 0..a {
                        let rand_probs = Array1::random(s_layer, Uniform::new(0.0, 1.0));
                        let sum = rand_probs.sum();
                        let mut slice = p.slice_mut(s![hh, ss, aa, next_layer_start..next_layer_start + s_layer]);
                        slice.assign(&(rand_probs / sum));
                    }
                }
            }
            rho.slice_mut(s![0..s_layer]).fill(1.0 / s_layer as f64);
        } else {
            p = Array4::random((h, s, a, s), Uniform::new(0.0, 1.0));
            // rho remains uniform by default in constructor if passed None
        }

        let mut r = if let Some((low, high)) = intermediate_reward_range {
            let span = high - low;
            Array3::random((h, s, a), Uniform::new(0.0, 1.0)) * span + low
        } else {
            Array3::random((h, s, a), Uniform::new(0.0, 1.0))
        };

        if let Some((low, high)) = terminal_reward_range {
            let span = high - low;
            let mut v_h_star = Array1::random(s, Uniform::new(0.0, 1.0)) * span + low;
            
            let mut indices: Vec<usize> = (0..s).collect();
            indices.shuffle(&mut rng);
            v_h_star[indices[0]] = low;
            v_h_star[indices[1]] = high;

            for ss in 0..s {
                for aa in 0..a {
                    r[[h - 1, ss, aa]] = v_h_star[ss];
                }
            }
        }

        Self::new(h, s, a, p, r, Some(rho))
    }
}

pub struct OfflineBounds {
    pub u_values: Array2<f64>, // [H+1, S]
    pub w_values: Array2<f64>, // [H+1, S]
    pub u_q: Array3<f64>,      // [H, S, A]
    pub m: Array2<f64>,        // [H, S]
    pub d: Array2<f64>,        // [H, S]
    pub r_next: Array1<f64>,   // [H]
    pub d_next_max: Array1<f64>, // [H]
    pub n_sa: Array3<i64>,     // [H, S, A]
    pub n_sas: Array4<i64>,    // [H, S, A, S]
}

pub fn compute_offline_bounds(
    mdp: &TabularMDP,
    trajectories: &[(Vec<usize>, Vec<usize>, Vec<f64>)],
    delta: f64,
    seed: Option<u64>,
    use_h_split: bool,
) -> OfflineBounds {
    let h = mdp.h;
    let s = mdp.s;
    let a = mdp.a;
    let k = trajectories.len();
    assert_eq!(k % h, 0, "Number of trajectories must be a multiple of H");

    let mut n_sa = Array3::<i64>::zeros((h, s, a));
    let mut n_sas = Array4::<i64>::zeros((h, s, a, s));

    if use_h_split {
        let mut rng = if let Some(s) = seed { StdRng::seed_from_u64(s) } else { StdRng::from_entropy() };
        let mut indices: Vec<usize> = (0..k).collect();
        indices.shuffle(&mut rng);

        for (pos, &idx) in indices.iter().enumerate() {
            let hh = pos % h;
            let (states, actions, _) = &trajectories[idx];
            let ss = states[hh];
            let aa = actions[hh];
            let s_next = states[hh + 1];
            n_sa[[hh, ss, aa]] += 1;
            n_sas[[hh, ss, aa, s_next]] += 1;
        }
    } else {
        for (states, actions, _) in trajectories {
            for hh in 0..h {
                let ss = states[hh];
                let aa = actions[hh];
                let s_next = states[hh + 1];
                n_sa[[hh, ss, aa]] += 1;
                n_sas[[hh, ss, aa, s_next]] += 1;
            }
        }
    }

    let mut p_hat = Array4::<f64>::zeros((h, s, a, s));
    for hh in 0..h {
        for ss in 0..s {
            for aa in 0..a {
                let n = n_sa[[hh, ss, aa]];
                if n == 0 {
                    p_hat.slice_mut(s![hh, ss, aa, ..]).fill(1.0 / s as f64);
                } else {
                    for ss_next in 0..s {
                        p_hat[[hh, ss, aa, ss_next]] = n_sas[[hh, ss, aa, ss_next]] as f64 / n as f64;
                    }
                }
            }
        }
    }

    let l1 = ((8.0 * s as f64 * a as f64 * h as f64) / delta.max(1e-12)).ln();
    let c1 = 2.0;
    let c2 = 14.0 / 3.0;

    let mut u_values = Array2::<f64>::zeros((h + 1, s));
    let mut w_values = Array2::<f64>::zeros((h + 1, s));
    let mut u_q = Array3::<f64>::zeros((h, s, a));

    for hh in (0..h).rev() {
        let _q_lower_h = Array2::<f64>::zeros((s, a));
        
        // Parallelizing over states for speed, similar to numba.prange
        let results: Vec<(Vec<f64>, Vec<f64>)> = (0..s).into_par_iter().map(|ss| {
            let mut u_q_row = vec![0.0; a];
            let mut q_lower_row = vec![0.0; a];
            for aa in 0..a {
                let n = n_sa[[hh, ss, aa]];
                let b;
                let mut mu_u = 0.0;
                let mut mu_w = 0.0;

                if n <= 1 {
                    b = (h - hh) as f64;
                } else {
                    for ss_next in 0..s {
                        let p = p_hat[[hh, ss, aa, ss_next]];
                        mu_u += p * u_values[[hh + 1, ss_next]];
                        mu_w += p * w_values[[hh + 1, ss_next]];
                    }

                    let mut var_u = 0.0;
                    let mut var_w = 0.0;
                    for ss_next in 0..s {
                        let p = p_hat[[hh, ss, aa, ss_next]];
                        var_u += p * (u_values[[hh + 1, ss_next]] - mu_u).powi(2);
                        var_w += p * (w_values[[hh + 1, ss_next]] - mu_w).powi(2);
                    }

                    let var_max = var_u.max(var_w);
                    let b_calc = c1 * (var_max * l1 / n as f64).sqrt() + c2 * (h - hh) as f64 * l1 / n as f64;
                    b = b_calc.min((h - hh) as f64);
                }

                u_q_row[aa] = (mdp.r[[hh, ss, aa]] + mu_u + b).min((h - hh) as f64);
                q_lower_row[aa] = (mdp.r[[hh, ss, aa]] + mu_w - b).max(0.0);
            }
            (u_q_row, q_lower_row)
        }).collect();

        for ss in 0..s {
            let (u_q_row, q_lower_row) = &results[ss];
            let mut max_u = -f64::INFINITY;
            let mut max_w = -f64::INFINITY;
            for aa in 0..a {
                u_q[[hh, ss, aa]] = u_q_row[aa];
                if u_q_row[aa] > max_u { max_u = u_q_row[aa]; }
                if q_lower_row[aa] > max_w { max_w = q_lower_row[aa]; }
            }
            u_values[[hh, ss]] = if max_u == -f64::INFINITY { 0.0 } else { max_u };
            w_values[[hh, ss]] = if max_w == -f64::INFINITY { 0.0 } else { max_w };
        }
    }

    let mut m = Array2::<f64>::zeros((h, s));
    let mut d = Array2::<f64>::zeros((h, s));
    for hh in 0..h {
        for ss in 0..s {
            m[[hh, ss]] = 0.5 * (u_values[[hh, ss]] + w_values[[hh, ss]]);
            d[[hh, ss]] = u_values[[hh, ss]] - w_values[[hh, ss]];
        }
    }

    let mut r_next = Array1::<f64>::zeros(h);
    let mut d_next_max = Array1::<f64>::zeros(h);
    for hh in 0..h {
        if hh + 1 < h {
            let u_next = u_values.slice(s![hh + 1, ..]);
            let w_next = w_values.slice(s![hh + 1, ..]);
            let mut max_u = -f64::INFINITY;
            let mut min_w = f64::INFINITY;
            for &val in u_next { if val > max_u { max_u = val; } }
            for &val in w_next { if val < min_w { min_w = val; } }
            r_next[hh] = max_u - min_w;

            let d_next = d.slice(s![hh + 1, ..]);
            let mut max_d = -f64::INFINITY;
            for &val in d_next { if val > max_d { max_d = val; } }
            d_next_max[hh] = max_d;
        } else {
            r_next[hh] = 0.0;
            d_next_max[hh] = 0.0;
        }
    }

    OfflineBounds {
        u_values,
        w_values,
        u_q,
        m,
        d,
        r_next,
        d_next_max,
        n_sa,
        n_sas,
    }
}

pub fn run_standard_ucbvi(
    mdp: &TabularMDP,
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

    let r_next_base = Array1::from_shape_fn(mdp.h, |hh| {
        if hh + 1 < mdp.h { (mdp.h - (hh + 1)) as f64 } else { 0.0 }
    });

    let mut regrets: Vec<f64> = Vec::with_capacity(t);
    let mut rewards = Vec::with_capacity(t);

    for _ in 0..t {
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

    (regrets, rewards)
}

pub fn run_count_initialized_ucbvi(
    mdp: &TabularMDP,
    offline_bounds: &OfflineBounds,
    t: usize,
    delta: f64,
    seed: u64,
) -> (Vec<f64>, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let (v_opt, _) = value_iteration(mdp);
    let opt_global = mdp.rho.dot(&v_opt.slice(s![0, ..]));

    let mut n_sa = offline_bounds.n_sa.clone();
    let mut n_sas = offline_bounds.n_sas.clone();

    let l = ((4.0 * mdp.s as f64 * mdp.a as f64 * mdp.h as f64 * t as f64) / delta.max(1e-12)).ln();
    let c1 = 2.0;
    let c2 = 14.0 / 3.0;

    let r_next_base = Array1::from_shape_fn(mdp.h, |hh| {
        if hh + 1 < mdp.h { (mdp.h - (hh + 1)) as f64 } else { 0.0 }
    });

    let mut regrets: Vec<f64> = Vec::with_capacity(t);
    let mut rewards = Vec::with_capacity(t);

    for _ in 0..t {
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

    (regrets, rewards)
}

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
