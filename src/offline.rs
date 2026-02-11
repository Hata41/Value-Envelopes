use ndarray::prelude::*;
use rayon::prelude::*;
use rand::prelude::*;
use crate::mdp::TabularMDP;

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
