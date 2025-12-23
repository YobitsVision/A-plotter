use eframe::egui;
use egui::{Align, Layout, RichText, ScrollArea, TextEdit, Vec2};
use plotter::{Plotter, PlotterTask};
use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::thread;

// Your existing mods/crates...
// (Keep all the mod declarations and imports from your previous main.rs)

#[derive(serde::Deserialize, serde::Serialize, Default)]
struct PlotterGui {
    numeric_id: String,
    start_nonce: String,
    total_nonces: String,
    plot_size_nonces: String,
    temp_dir: String,
    drives: String, // comma-separated
    cpu_threads: String,
    mem: String,

    #[serde(skip)]
    log: String,
    #[serde(skip)]
    current_progress: f64,
    #[serde(skip)]
    current_status: String,
    #[serde(skip)]
    is_plotting: bool,
}

impl eframe::App for PlotterGui {
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, eframe::APP_KEY, self);
    }

    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Signum Plotter GUI");

            ui.horizontal(|ui| {
                ui.label("Account ID:");
                ui.text_edit_singleline(&mut self.numeric_id);
            });

            ui.horizontal(|ui| {
                ui.label("Start Nonce:");
                ui.text_edit_singleline(&mut self.start_nonce);
            });

            ui.horizontal(|ui| {
                ui.label("Total Nonces:");
                ui.text_edit_singleline(&mut self.total_nonces);
            });

            ui.horizontal(|ui| {
                ui.label("Plot Size (nonces):");
                ui.text_edit_singleline(&mut self.plot_size_nonces);
                ui.label("(default ~4_194_304 ≈ 1 TiB)");
            });

            ui.horizontal(|ui| {
                ui.label("SSD Temp Dir:");
                ui.text_edit_singleline(&mut self.temp_dir);
            });

            ui.horizontal(|ui| {
                ui.label("Final Drives (comma-separated):");
                ui.text_edit_singleline(&mut self.drives);
            });

            ui.horizontal(|ui| {
                ui.label("CPU Threads:");
                ui.text_edit_singleline(&mut self.cpu_threads);
            });

            ui.horizontal(|ui| {
                ui.label("Memory Limit:");
                ui.text_edit_singleline(&mut self.mem);
            });

            ui.add_space(10.0);

            if self.is_plotting {
                ui.horizontal(|ui| {
                    ui.label("Status:");
                    ui.label(RichText::new(&self.current_status).strong());
                });
                ui.add(egui::ProgressBar::new(self.current_progress as f32).text("Plotting..."));
                if ui.button("Cancel").clicked() {
                    // Simple cancel: just set flag (you can enhance with channel)
                    self.is_plotting = false;
                }
            } else if ui.button("Start Plotting").clicked() {
                self.start_plotting(ctx);
            }

            ui.add_space(10.0);
            ui.separator();
            ui.label("Log:");
            ScrollArea::vertical().max_height(200.0).show(ui, |ui| {
                ui.add(TextEdit::multiline(&mut self.log).desired_rows(10).interactive(false));
            });
        });
    }
}

impl PlotterGui {
    fn start_plotting(&mut self, ctx: &egui::Context) {
        let numeric_id: u64 = match self.numeric_id.trim().parse() {
            Ok(v) => v,
            Err(_) => { self.log += "Invalid Account ID\n"; return; }
        };
        let mut current_nonce: u64 = self.start_nonce.trim().parse().unwrap_or(0);
        let total_nonces: u64 = match self.total_nonces.trim().parse() {
            Ok(v) => v,
            Err(_) => { self.log += "Invalid Total Nonces\n"; return; }
        };
        let plot_size_nonces: u64 = self.plot_size_nonces.trim().parse().unwrap_or(4_194_304);
        let temp_dir = if self.temp_dir.trim().is_empty() { None } else { Some(PathBuf::from(&self.temp_dir)) };
        let final_dirs: Vec<PathBuf> = self.drives.split(',').map(|s| PathBuf::from(s.trim())).filter(|p| !p.as_os_str().is_empty()).collect();
        if final_dirs.is_empty() { self.log += "No final drives specified\n"; return; }

        let cpu_threads: u8 = self.cpu_threads.trim().parse().unwrap_or(0);
        let mem = if self.mem.trim().is_empty() { "0B".to_string() } else { self.mem.clone() };

        self.is_plotting = true;
        self.current_progress = 0.0;
        self.current_status = "Starting...".to_string();
        self.log += "Plotting started\n";

        let app_state = Arc::new(Mutex::new(self));
        let ctx = ctx.clone();
        let p = Plotter::new();

        thread::spawn(move || {
            let mut remaining = total_nonces;
            let mut drive_index = 0;

            while remaining > 0 && app_state.lock().unwrap().is_plotting {
                let this_plot_nonces = remaining.min(plot_size_nonces);
                let final_dir = &final_dirs[drive_index];
                let scoops = 4096;
                let filename = format!("{}_{}_{}_{}", numeric_id, current_nonce, this_plot_nonces, scoops);

                let write_dir = temp_dir.as_ref().unwrap_or(final_dir);
                let output_path_str = write_dir.to_str().unwrap().to_string();

                {
                    let mut state = app_state.lock().unwrap();
                    state.current_status = format!("Plotting {} ({}/{})", filename, total_nonces - remaining + this_plot_nonces, total_nonces);
                    state.log += &format!("Starting {} nonces → {}\n", this_plot_nonces, filename);
                }
                ctx.request_repaint();

                // Run the actual plotting (you can add progress callback if plotter supports it)
                p.run(PlotterTask {
                    numeric_id,
                    start_nonce: current_nonce,
                    nonces: this_plot_nonces,
                    output_path: output_path_str.clone(),
                    mem: mem.clone(),
                    cpu_threads,
                    gpus: None,
                    direct_io: true,
                    async_io: true,
                    quiet: true,
                    benchmark: false,
                    zcb: false,
                });

                // Move file if using temp
                if temp_dir.is_some() {
                    let temp_file = write_dir.join(&filename);
                    let final_file = final_dir.join(&filename);
                    fs::create_dir_all(final_dir).ok();
                    if let Err(e) = fs::rename(&temp_file, &final_file) {
                        if e.kind() == std::io::ErrorKind::CrossesDevices {
                            fs::copy(&temp_file, &final_file).ok();
                            fs::remove_file(&temp_file).ok();
                        }
                    }
                }

                {
                    let mut state = app_state.lock().unwrap();
                    state.log += &format!("Completed {}\n", filename);
                    state.current_progress = ((total_nonces - remaining) as f64 / total_nonces as f64) * 100.0;
                }
                ctx.request_repaint();

                remaining -= this_plot_nonces;
                current_nonce += this_plot_nonces;
                drive_index = (drive_index + 1) % final_dirs.len();
            }

            let mut state = app_state.lock().unwrap();
            state.is_plotting = false;
            state.current_status = "Finished!".to_string();
            state.log += "All plotting complete!\n";
            ctx.request_repaint();
        });
    }
}

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size(Vec2::new(800.0, 600.0)),
        ..Default::default()
    };

    eframe::run_native(
        "Signum Plotter GUI",
        options,
        Box::new(|cc| {
            let mut app: PlotterGui = eframe::get_value(cc.storage.unwrap(), eframe::APP_KEY).unwrap_or_default();
            app.plot_size_nonces = "4194304".to_string();
            Box::new(app)
        }),
    )
}
