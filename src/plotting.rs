use plotters::prelude::*;
use plotters::style::full_palette::{PURPLE, ORANGE};
use std::fs::File;
use std::io::{BufRead, BufReader};

pub fn plot_regret_curves(
    folder: &str,
    shaping_algo: &str,
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(output_path, (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;

    let standard_file = format!("{}/Standard_UCBVI.dat", folder);
    let standard_data = read_dat_file(&standard_file)?;
    
    if standard_data.is_empty() { return Ok(()); }

    let max_regret = standard_data.iter().map(|d| d.1).fold(0.0, f64::max) * 1.5;
    let max_episode = standard_data.iter().map(|d| d.0).fold(0.0, f64::max);

    let mut chart = ChartBuilder::on(&root)
        .caption(format!("{} vs Baseline", shaping_algo), ("sans-serif", 30).into_font())
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0.0..max_episode, 0.0..max_regret)?;

    chart.configure_mesh()
        .x_desc("Episode Number")
        .y_desc("Cumulative Regret")
        .draw()?;

    chart.draw_series(LineSeries::new(
        standard_data.iter().map(|d| (d.0, d.1)),
        ShapeStyle::from(&BLACK).stroke_width(2),
    ))?
    .label("Standard UCBVI")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLACK));

    let paths = std::fs::read_dir(folder)?;
    let mut k_folders = Vec::new();
    for path in paths {
        let entry = path?.path();
        if entry.is_dir() {
            let name = entry.file_name().unwrap().to_str().unwrap();
            if name.starts_with("K=") {
                k_folders.push(entry);
            }
        }
    }

    let colors = vec![&RED, &BLUE, &GREEN, &CYAN, &MAGENTA, &YELLOW];
    
    for (i, k_folder) in k_folders.iter().enumerate() {
        let color = colors[i % colors.len()];
        let k_str = k_folder.file_name().unwrap().to_str().unwrap();

        let shaping_file = format!("{}/{}.dat", k_folder.display(), shaping_algo);
        if let Ok(data) = read_dat_file(&shaping_file) {
            chart.draw_series(LineSeries::new(
                data.iter().map(|d| (d.0, d.1)),
                ShapeStyle::from(color).stroke_width(2),
            ))?
            .label(format!("{} ({})", shaping_algo, k_str))
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color));
        }

        let count_init_file = format!("{}/Count_Init_UCBVI.dat", k_folder.display());
        if let Ok(data) = read_dat_file(&count_init_file) {
            chart.draw_series(LineSeries::new(
                data.iter().map(|d| (d.0, d.1)),
                ShapeStyle::from(color).stroke_width(1),
            ))?
            .label(format!("Count-Init ({})", k_str))
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color));
        }
    }

    chart.configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;
    Ok(())
}

pub fn plot_mdp_trials(
    folder: &str,
    title: &str,
    output_path: &str,
    is_log_x: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(output_path, (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;

    let algos = vec![
        ("Bonus_Shaping_Only.dat", "Full-Bonus", &PURPLE),
        ("Upper_Bonus_Shaping.dat", "Upper-Bonus", &GREEN),
        ("Count_Init_UCBVI.dat", "Count-Init", &BLUE),
    ];

    let mut all_data = Vec::new();
    for (file_name, label, color) in algos {
        let path = format!("{}/{}", folder, file_name);
        if let Ok(data) = read_dat_file(&path) {
            all_data.push((data, label, color));
        }
    }

    if all_data.is_empty() { return Ok(()); }

    let min_x = all_data.iter().flat_map(|d| d.0.iter().map(|p| p.0)).fold(f64::INFINITY, f64::min);
    let max_x = all_data.iter().flat_map(|d| d.0.iter().map(|p| p.0)).fold(f64::NEG_INFINITY, f64::max);
    
    let mut chart_builder = ChartBuilder::on(&root);
    chart_builder.caption(title, ("sans-serif", 30).into_font())
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60);

    if is_log_x {
        let mut chart = chart_builder.build_cartesian_2d((min_x..max_x).log_scale(), 0.0..1.0)?;
        chart.configure_mesh().draw()?;
        for (data, label, color) in all_data {
            chart.draw_series(LineSeries::new(
                data.iter().map(|d| (d.0, d.1)),
                ShapeStyle::from(color).stroke_width(2),
            ))?
            .label(label)
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color));
        }
        chart.configure_series_labels().draw()?;
    } else {
        let mut chart = chart_builder.build_cartesian_2d(min_x..max_x, 0.0..1.0)?;
        chart.configure_mesh().draw()?;
        for (data, label, color) in all_data {
            chart.draw_series(LineSeries::new(
                data.iter().map(|d| (d.0, d.1)),
                ShapeStyle::from(color).stroke_width(2),
            ))?
            .label(label)
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color));
        }
        chart.configure_series_labels().draw()?;
    }

    root.present()?;
    Ok(())
}

pub fn plot_convergence(
    file_path: &str,
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(output_path, (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;

    let file = File::open(file_path)?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();
    
    let header = lines.next().ok_or("Empty file")??;
    let columns: Vec<&str> = header.split_whitespace().collect();
    
    let mut data = Vec::new();
    for line in lines {
        let line = line?;
        let vals: Vec<f64> = line.split_whitespace().map(|s| s.parse().unwrap()).collect();
        data.push(vals);
    }

    let k_col = 0;
    let r_col = columns.iter().position(|&c| c == "R_h_layer").ok_or("Missing R_h_layer")?;
    let span_col = columns.iter().position(|&c| c == "span_h_layer").ok_or("Missing span_h_layer")?;
    let d_cols: Vec<usize> = columns.iter().enumerate()
        .filter(|&(_, c)| c.starts_with("D_s"))
        .map(|(i, _)| i)
        .collect();

    let max_k = data.iter().map(|v| v[k_col]).fold(0.0, f64::max);
    let max_y = data.iter().map(|v| v[r_col]).fold(0.0, f64::max) * 1.5;

    let mut chart = ChartBuilder::on(&root)
        .caption("Convergence of Bounds", ("sans-serif", 30).into_font())
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0.0..max_k, 0.0..max_y)?;

    chart.configure_mesh().draw()?;

    for &col in &d_cols {
        chart.draw_series(LineSeries::new(
            data.iter().map(|v| (v[k_col], v[col])),
            ShapeStyle::from(&ORANGE).stroke_width(1),
        ))?;
    }

    chart.draw_series(LineSeries::new(
        data.iter().map(|v| (v[k_col], v[r_col])),
        ShapeStyle::from(&BLUE).stroke_width(2),
    ))?
    .label("R_h (In-Layer Range)")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    chart.draw_series(LineSeries::new(
        data.iter().map(|v| (v[k_col], v[span_col])),
        ShapeStyle::from(&RED).stroke_width(2),
    ))?
    .label("Optimal Span")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    chart.configure_series_labels().draw()?;

    root.present()?;
    Ok(())
}

fn read_dat_file(path: &str) -> Result<Vec<(f64, f64, f64)>, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut data = Vec::new();
    for line in reader.lines().skip(1) {
        let line = line?;
        if line.starts_with('#') || line.trim().is_empty() { continue; }
        let vals: Vec<f64> = line.split_whitespace().map(|s| s.parse().unwrap()).collect();
        if vals.len() >= 2 {
            let v1 = vals[0];
            let v2 = vals[1];
            let v3 = if vals.len() >= 3 { vals[2] } else { 0.0 };
            data.push((v1, v2, v3));
        }
    }
    Ok(data)
}
