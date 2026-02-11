use ndarray::prelude::*;
use ndarray_rand::rand_distr::{Uniform, Distribution};
use ndarray_rand::RandomExt;
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::io::{self, Write};
use std::time::Instant;

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
    show_progress: bool,
) -> Vec<(Vec<usize>, Vec<usize>, Vec<f64>)> {
    let mut rng = if let Some(s) = seed { StdRng::seed_from_u64(s) } else { StdRng::from_entropy() };
    let mut trajectories = Vec::with_capacity(k);

    let start_time = Instant::now();

    for i in 0..k {
        if show_progress && i % 1000 == 0 {
            let elapsed = start_time.elapsed();
            let progress = i as f64 / k as f64;
            let percentage = progress * 100.0;
            if progress > 0.0 {
                let total_estimated = elapsed.div_f64(progress);
                let remaining = total_estimated - elapsed;
                print!("\rDataset Generation: {}/{} ({:.1}%) ETA: {:.1}s", i, k, percentage, remaining.as_secs_f64());
            } else {
                print!("\rDataset Generation: {}/{} ({:.1}%)", i, k, percentage);
            }
            io::stdout().flush().unwrap();
        }

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

    if show_progress {
        println!("\rDataset Generation: {}/{} (100.0%)", k, k);
    }

    trajectories
}
