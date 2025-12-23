#[macro_use]
extern crate clap;
#[macro_use]
extern crate cfg_if;

mod cpu_hasher;
#[cfg(feature = "opencl")]
mod gpu_hasher;
#[cfg(feature = "opencl")]
mod ocl;
mod plotter;
mod poc_hashing;
mod scheduler;
mod shabal256;
mod utils;
mod writer;
mod buffer;

use crate::plotter::{Plotter, PlotterTask};
use crate::utils::set_low_prio;
use clap::AppSettings::{ArgRequiredElseHelp, DeriveDisplayOrder, VersionlessSubcommands};
#[cfg(feature = "opencl")]
use clap::ArgGroup;
use clap::{App, Arg};
use std::cmp::min;
use std::fs;
use std::path::PathBuf;

fn main() {
    let mut arg = App::new("signum-plotter")
        .version(crate_version!())
        .author(crate_authors!())
        .about(crate_description!())
        .setting(ArgRequiredElseHelp)
        .setting(DeriveDisplayOrder)
        .setting(VersionlessSubcommands)
        .arg(
            Arg::with_name("disable direct i/o")
                .short("d")
                .long("ddio")
                .help("Disables direct i/o")
                .global(true),
        )
        .arg(
            Arg::with_name("disable async i/o")
                .short("a")
                .long("daio")
                .help("Disables async writing (single RAM buffer mode)")
                .global(true),
        )
        .arg(
            Arg::with_name("low priority")
                .short("l")
                .long("prio")
                .help("Runs with low priority")
                .global(true),
        )
        .arg(
            Arg::with_name("non-verbosity")
                .short("q")
                .long("quiet")
                .help("Runs in non-verbose mode")
                .global(true),
        )
        .arg(
            Arg::with_name("benchmark")
                .short("b")
                .long("bench")
                .help("Runs in xPU benchmark mode")
                .global(true),
        )
        .arg(
            Arg::with_name("numeric id")
                .short("i")
                .long("id")
                .value_name("numeric_ID")
                .help("Your numeric Account ID")
                .takes_value(true)
                .required_unless("ocl-devices"),
        )
        .arg(
            Arg::with_name("start nonce")
                .short("s")
                .long("sn")
                .value_name("start_nonce")
                .help("Starting nonce")
                .takes_value(true)
                .default_value("0"),
        )
        .arg(
            Arg::with_name("nonces")
                .short("n")
                .long("nonces")
                .value_name("total_nonces")
                .help("Total number of nonces to plot (can span multiple files/drives)")
                .takes_value(true)
                .required(true),
        )
        .arg(
            Arg::with_name("temp")
                .short("t")
                .long("temp")
                .value_name("temp_dir")
                .help("SSD directory for fast temporary plotting (highly recommended)")
                .takes_value(true)
                .required(false),
        )
        .arg(
            Arg::with_name("drives")
                .short("D")
                .long("drives")
                .value_name("path1,path2,...")
                .help("Comma-separated list of final HDD plot directories (queue will cycle through them)")
                .takes_value(true)
                .required_unless("path"),
        )
        .arg(
            Arg::with_name("path")
                .short("p")
                .long("path")
                .value_name("single_path")
                .help("Single final directory (legacy mode, no queue)")
                .takes_value(true)
                .required_unless("drives"),
        )
        .arg(
            Arg::with_name("plotsize")
                .long("plotsize")
                .value_name("nonces")
                .help("Nonces per individual plot file (default: 4194304 ≈ 1 TiB)")
                .takes_value(true)
                .default_value("4194304"),
        )
        .arg(
            Arg::with_name("memory")
                .short("m")
                .long("mem")
                .value_name("memory")
                .help("Maximum memory usage (optional)")
                .takes_value(true),
        )
        .args(&[
            Arg::with_name("cpu")
                .short("c")
                .long("cpu")
                .value_name("threads")
                .help("CPU threads to use")
                .takes_value(true),
            #[cfg(feature = "opencl")]
            Arg::with_name("gpu")
                .short("g")
                .long("gpu")
                .value_name("platform_id:device_id:cores")
                .multiple(true)
                .takes_value(true),
        ])
        .groups(&[#[cfg(feature = "opencl")]
            ArgGroup::with_name("processing")
                .args(&["cpu", "gpu"])
                .multiple(true)]);

    #[cfg(feature = "opencl")]
    let arg = arg
        .arg(
            Arg::with_name("ocl-devices")
                .short("o")
                .long("opencl")
                .help("Display OpenCL platforms and devices")
                .global(true),
        )
        .arg(
            Arg::with_name("zero-copy")
                .short("z")
                .long("zcb")
                .help("Enables zero copy buffers for integrated GPUs")
                .global(true),
        );

    let matches = arg.get_matches();

    if matches.is_present("low priority") {
        set_low_prio();
    }

    if matches.is_present("ocl-devices") {
        #[cfg(feature = "opencl")]
        ocl::platform_info();
        return;
    }

    let numeric_id = value_t!(matches, "numeric id", u64).unwrap_or_else(|e| e.exit());
    let mut current_nonce = value_t!(matches, "start nonce", u64).unwrap_or_else(|e| e.exit());
    let total_nonces = value_t!(matches, "nonces", u64).unwrap_or_else(|e| e.exit());
    let plot_size_nonces = value_t!(matches, "plotsize", u64).unwrap_or_else(|e| e.exit());

    let temp_dir: Option<PathBuf> = matches.value_of("temp").map(PathBuf::from);

    // Determine final drives
    let final_dirs: Vec<PathBuf> = if let Some(drives_str) = matches.value_of("drives") {
        drives_str.split(',').map(|s| PathBuf::from(s.trim())).collect()
    } else {
        // Legacy single path
        vec![PathBuf::from(
            matches.value_of("path").unwrap_or("."),
        )]
    };

    if final_dirs.is_empty() {
        eprintln!("Error: No final plot directories specified.");
        return;
    }

    let mem = value_t!(matches, "memory", String).unwrap_or_else(|_| "0B".to_owned());
    let cpu_threads = value_t!(matches, "cpu", u8).unwrap_or(0u8);

    let gpus = if matches.occurrences_of("gpu") > 0 {
        Some(values_t!(matches, "gpu", String).unwrap())
    } else {
        None
    };

    let cores = sys_info::cpu_num().unwrap() as u8;
    let cpu_threads = if cpu_threads == 0 {
        cores
    } else {
        min(2 * cores, cpu_threads)
    };

    #[cfg(feature = "opencl")]
    let cpu_threads = if matches.occurrences_of("gpu") > 0 && matches.occurrences_of("cpu") == 0 {
        0u8
    } else {
        cpu_threads
    };

    let p = Plotter::new();

    let mut remaining = total_nonces;
    let mut drive_index = 0;

    println!("Starting queue: {} total nonces → {} plot files across {} drive(s)", total_nonces, (total_nonces + plot_size_nonces - 1) / plot_size_nonces, final_dirs.len());

    while remaining > 0 {
        let this_plot_nonces = remaining.min(plot_size_nonces);
        let final_dir = &final_dirs[drive_index];
        let scoops = 4096;
        let filename = format!("{}_{}_{}_{}", numeric_id, current_nonce, this_plot_nonces, scoops);

        // Effective write path (SSD temp if provided)
        let write_dir = temp_dir.as_ref().unwrap_or(final_dir);
        let write_path = write_dir.join(&filename);
        let output_path_str = write_dir
            .clone()
            .into_os_string()
            .into_string()
            .unwrap();

        println!("\nPlotting {} nonces → {} (to {})", this_plot_nonces, filename, write_path.display());

        // Ensure write directory exists
        if let Err(e) = fs::create_dir_all(write_dir) {
            eprintln!("Failed to create write directory: {}", e);
            break;
        }

        // Run plotting
        p.run(PlotterTask {
            numeric_id,
            start_nonce: current_nonce,
            nonces: this_plot_nonces,
            output_path: output_path_str,
            mem: mem.clone(),
            cpu_threads,
            gpus: gpus.clone(),
            direct_io: !matches.is_present("disable direct i/o"),
            async_io: !matches.is_present("disable async i/o"),
            quiet: matches.is_present("non-verbosity"),
            benchmark: matches.is_present("benchmark"),
            zcb: matches.is_present("zero-copy"),
        });

        // Move to final drive if temp was used
        if temp_dir.is_some() {
            let final_file = final_dir.join(&filename);

            if let Err(e) = fs::create_dir_all(final_dir) {
                eprintln!("Failed to create final dir: {}", e);
                break;
            }

            println!("Moving to final location: {}", final_file.display());

            if let Err(e) = fs::rename(&write_path, &final_file) {
                if e.kind() == std::io::ErrorKind::CrossesDevices {
                    println!("Cross-device: copying instead...");
                    if fs::copy(&write_path, &final_file).is_err() || fs::remove_file(&write_path).is_err() {
                        eprintln!("Copy/delete failed — manual cleanup needed");
                    }
                } else {
                    eprintln!("Move failed: {}", e);
                }
            }
        }

        println!("Completed: {} → {}", filename, final_dir.display());

        remaining -= this_plot_nonces;
        current_nonce += this_plot_nonces;
        drive_index = (drive_index + 1) % final_dirs.len();
    }

    println!("\nAll plotting complete! Total nonces plotted: {}", total_nonces);
}
