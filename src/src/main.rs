use eframe::egui;
use egui::{ComboBox, RichText, ScrollArea, TextEdit, Vec2};
use std::fs;
use std::io::{self, Read, Write}; // <-- Added Read and Write
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::thread;

// Module declarations
mod buffer;
mod cpu_hasher;
mod poc_hashing;
mod gpu_hasher;
mod ocl;
mod scheduler;
mod shabal256;
mod utils;
mod writer;

mod plotter;
use plotter::{Plotter, PlotterTask};
use ocl_core::{
    get_platform_ids, get_device_ids, get_device_info, DeviceInfo, DeviceType,
};
use ocl::platform_info;

#[derive(serde::Deserialize, serde::Serialize, Default, Clone)]
struct PlotterGui {
    numeric_id: String,
    start_nonce: String,
    total_nonces: String,
    plot_size_nonces: String,
    temp_dir: String,
    drives: String,
    cpu_threads: String,
    mem: String,
    selected_gpu: String,
    show_gpu_info: bool,

    #[serde(skip)]
    gpu_options: Vec<String>,
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

            ui.horizontal(|ui| {
                ui.label("GPU Selection:");
                ComboBox::from_id_source("gpu_combo")
                    .width(450.0)
                    .selected_text(&self.selected_gpu)
                    .show_ui(ui, |ui| {
                        for option in &self.gpu_options {
                            ui.selectable_value(&mut self.selected_gpu, option.clone(), option);
                        }
                    });
            });

            ui.horizontal(|ui| {
                ui.checkbox(&mut self.show_gpu_info, "Show detailed GPU info in console on start");
            });

            ui.add_space(10.0);

            if self.is_plotting {
                ui.horizontal(|ui| {
                    ui.label("Status:");
                    ui.label(RichText::new(&self.current_status).strong());
                });
                ui.add(egui::ProgressBar::new(self.current_progress as f32 / 100.0).text("Plotting..."));
                if ui.button("Cancel").clicked() {
                    self.is_plotting = false;
                }
            } else if ui.button("Start Plotting").clicked() {
                self.start_plotting(ctx);
            }

            ui.add_space(10.0);
            ui.separator();
            ui.label("Log:");
            ScrollArea::vertical()
                .max_height(200.0)
                .show(ui, |ui| {
                    ui.add(
                        TextEdit::multiline(&mut self.log.as_str())
                            .desired_rows(10)
                            .interactive(false),
                    );
                });
        });
    }
}

impl PlotterGui {
    fn start_plotting(&mut self, ctx: &egui::Context) {
        let numeric_id: u64 = match self.numeric_id.trim().parse() {
            Ok(v) => v,
            Err(_) => {
                self.log += "Invalid Account ID\n";
                return;
            }
        };
        let mut current_nonce: u64 = self.start_nonce.trim().parse().unwrap_or(0);
        let total_nonces: u64 = match self.total_nonces.trim().parse() {
            Ok(v) => v,
            Err(_) => {
                self.log += "Invalid Total Nonces\n";
                return;
            }
        };
        let plot_size_nonces: u64 = self.plot_size_nonces.trim().parse().unwrap_or(4_194_304);
        let temp_dir = if self.temp_dir.trim().is_empty() {
            None
        } else {
            Some(PathBuf::from(self.temp_dir.trim()))
        };
        let final_dirs: Vec<PathBuf> = self
            .drives
            .split(',')
            .map(|s| PathBuf::from(s.trim()))
            .filter(|p| !p.as_os_str().is_empty())
            .collect();
        if final_dirs.is_empty() {
            self.log += "No final drives specified\n";
            return;
        }

        let cpu_threads: u8 = self.cpu_threads.trim().parse().unwrap_or(0);
        let mem = if self.mem.trim().is_empty() {
            "0B".to_string()
        } else {
            self.mem.trim().to_string()
        };

        let gpus: Option<Vec<String>> = if self.selected_gpu == "CPU Only" {
            self.log += "Using CPU only\n\n";
            None
        } else if self.selected_gpu == "All GPUs" {
            self.log += "Using all available GPUs\n\n";
            Some(vec![])
        } else if self.selected_gpu.starts_with("Device ") {
            let id_str = self.selected_gpu
                .split(": ")
                .next()
                .unwrap()
                .split_whitespace()
                .nth(1)
                .unwrap();
            self.log += &format!("Using GPU: {}\n\n", self.selected_gpu);
            Some(vec![id_str.to_string()])
        } else {
            self.log += "No valid GPU selected — using CPU only\n\n";
            None
        };

        self.is_plotting = true;
        self.current_progress = 0.0;
        self.current_status = "Starting...".to_string();
        self.log += "Plotting started\n";

        if self.show_gpu_info && gpus.is_some() {
            self.log += "--- Detailed OpenCL Info ---\n";
            self.log += "(Full list printed to console)\n";
            self.log += "---------------------------\n\n";
            platform_info();
        }

        let app_state = Arc::new(Mutex::new(self.clone()));
        let ctx = ctx.clone();

        thread::spawn(move || {
            let mut remaining = total_nonces;
            let mut drive_index = 0;

            while remaining > 0 && app_state.lock().unwrap().is_plotting {
                let this_plot_nonces = remaining.min(plot_size_nonces);
                let final_dir = &final_dirs[drive_index];
                let scoops = 4096;
                let filename = format!(
                    "{}_{}_{}_{}",
                    numeric_id, current_nonce, this_plot_nonces, scoops
                );

                let write_dir = temp_dir.as_ref().unwrap_or(final_dir);
                let output_path_str = write_dir.to_str().unwrap().to_string();

                {
                    let mut state = app_state.lock().unwrap();
                    state.current_status = format!(
                        "Plotting {} ({}/{})",
                        filename,
                        total_nonces - remaining + this_plot_nonces,
                        total_nonces
                    );
                    state.log += &format!("Starting {} nonces → {}\n", this_plot_nonces, filename);
                }
                ctx.request_repaint();

                let plotter = Plotter::new();
                plotter.run(PlotterTask {
                    numeric_id,
                    start_nonce: current_nonce,
                    nonces: this_plot_nonces,
                    output_path: output_path_str.clone(),
                    mem: mem.clone(),
                    cpu_threads,
                    gpus: gpus.clone(),
                    direct_io: true,
                    async_io: true,
                    quiet: true,
                    benchmark: false,
                    zcb: false,
                });

                // === Ultra-robust cross-device file move/copy with progress and diagnostics ===
                if temp_dir.is_some() {
                    let temp_file = write_dir.join(&filename);
                    let final_file = final_dir.join(&filename);
                    let _ = fs::create_dir_all(final_dir);

                    let mut state = app_state.lock().unwrap();

                    state.log += &format!("=== FINALIZING PLOT {} ===\n", filename);
                    state.log += &format!("Temp path: {}\n", temp_file.display());
                    state.log += &format!("Final path: {}\n", final_file.display());

                    if !temp_file.exists() {
                        state.log += "ERROR: Temp file does not exist! Skipping move.\n";
                        ctx.request_repaint();
                    } else if temp_file.metadata().map(|m| m.len()).unwrap_or(0) == 0 {
                        state.log += "ERROR: Temp file is empty (0 bytes)!\n";
                        ctx.request_repaint();
                    } else {
                        state.log += &format!("Temp file size: {} GiB — ready for move/copy\n", temp_file.metadata().unwrap().len() / 1024 / 1024 / 1024);
                        ctx.request_repaint();

                        match fs::rename(&temp_file, &final_file) {
                            Ok(_) => {
                                state.log += "✓ Successfully moved (same drive — instant)\n";
                            }
                            Err(e) if e.kind() == io::ErrorKind::CrossesDevices => {
                                state.log += "Cross-device move detected — starting copy...\n";
                                ctx.request_repaint();

                                match Self::robust_copy(&temp_file, &final_file, &app_state, &ctx) {
                                    Ok(copied) => {
                                        state.log += &format!("✓ Successfully copied ({} GiB)\n", copied / 1024 / 1024 / 1024);
                                        if let Err(del_err) = fs::remove_file(&temp_file) {
                                            state.log += &format!("Warning: Failed to delete temp file: {}\n", del_err);
                                        } else {
                                            state.log += "✓ Temp file cleaned up\n";
                                        }
                                    }
                                    Err(copy_err) => {
                                        state.log += &format!("✗ COPY FAILED: {}\n", copy_err);
                                        state.log += "Possible causes:\n";
                                        state.log += "- Antivirus/Windows Defender blocking large file copy\n";
                                        state.log += "- Insufficient permissions on final drive\n";
                                        state.log += "- Final drive is full or read-only\n";
                                        state.log += "- File is locked by another process\n";
                                        state.log += "Plot remains in temp directory for safety.\n";
                                    }
                                }
                            }
                            Err(e) => {
                                state.log += &format!("✗ Move failed: {}\n", e);
                            }
                        }
                    }

                    state.log += "=== END FINALIZE ===\n\n";
                    ctx.request_repaint();
                }

                {
                    let mut state = app_state.lock().unwrap();
                    state.log += &format!("Completed {}\n", filename);
                    state.current_progress =
                        ((total_nonces - remaining + this_plot_nonces) as f64 / total_nonces as f64) * 100.0;
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

    // Helper function for robust copy with progress feedback
    fn robust_copy(
        src: &PathBuf,
        dst: &PathBuf,
        app_state: &Arc<Mutex<Self>>,
        ctx: &egui::Context,
    ) -> io::Result<u64> {
        let metadata = fs::metadata(src)?;
        let total_size = metadata.len();
        let mut copied: u64 = 0;

        let mut reader = fs::File::open(src)?;
        let mut writer = fs::File::create(dst)?;

        let mut buffer = vec![0u8; 8 * 1024 * 1024]; // 8 MiB buffer for speed

        loop {
            let bytes_read = reader.read(&mut buffer)?;
            if bytes_read == 0 {
                break;
            }

            writer.write_all(&buffer[..bytes_read])?;
            copied += bytes_read as u64;

            // Update log every 500 MiB or at end
            if copied % (500 * 1024 * 1024) < bytes_read as u64 || copied == total_size {
                let mut state = app_state.lock().unwrap();
                let percent = if total_size > 0 {
                    (copied as f64 / total_size as f64) * 100.0
                } else {
                    100.0
                };
                state.log += &format!(
                    "Copy progress: {:.1}% ({} / {} GiB)\n",
                    percent,
                    copied / 1024 / 1024 / 1024,
                    total_size / 1024 / 1024 / 1024
                );
                ctx.request_repaint();
            }
        }

        writer.flush()?;
        Ok(copied)
    }
}

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size(Vec2::new(800.0, 600.0)),
        ..Default::default()
    };

    eframe::run_native(
        "Signum Plotter GUI",
        options,
        Box::new(|cc| {
            let mut app: PlotterGui = cc
                .storage
                .and_then(|s| eframe::get_value(s, eframe::APP_KEY))
                .unwrap_or_default();

            if app.plot_size_nonces.is_empty() {
                app.plot_size_nonces = "4194304".to_string();
            }

            if app.gpu_options.is_empty() {
                app.gpu_options.push("CPU Only".to_string());
                app.gpu_options.push("All GPUs".to_string());

                let platforms = match get_platform_ids() {
                    Ok(p) => p,
                    Err(_) => vec![],
                };

                let mut global_id = 0;
                for platform in platforms {
                    let devices = match get_device_ids(platform, Some(DeviceType::GPU), None) {
                        Ok(d) => d,
                        Err(_) => continue,
                    };

                    for device in devices {
                        let name = get_device_info(device, DeviceInfo::Name)
                            .ok()
                            .map(|v| v.to_string())
                            .unwrap_or_else(|| "Unknown Device".to_string());

                        let vendor = get_device_info(device, DeviceInfo::Vendor)
                            .ok()
                            .map(|v| v.to_string())
                            .unwrap_or_else(|| "Unknown Vendor".to_string());

                        let label = format!("Device {}: {} ({})", global_id, name.trim(), vendor.trim());
                        app.gpu_options.push(label);
                        global_id += 1;
                    }
                }

                if app.gpu_options.len() == 2 {
                    app.gpu_options[1] = "All GPUs (none detected)".to_string();
                }
            }

            if app.selected_gpu.is_empty() {
                if app.gpu_options.len() > 2 {
                    app.selected_gpu = "All GPUs".to_string();
                } else {
                    app.selected_gpu = "CPU Only".to_string();
                }
            }

            Ok(Box::new(app))
        }),
    )
}