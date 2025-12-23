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
                .help("your numeric Account ID")
                .takes_value(true)
                .required_unless("ocl-devices"),
        )
        .arg(
            Arg::with_name("start nonce")
                .short("s")
                .long("sn")
                .value_name("start_nonce")
                .help("where you want to start plotting")
                .takes_value(true)
                .required_unless("ocl-devices"),
        )
        .arg(
            Arg::with_name("nonces")
                .short("n")
                .long("n")
                .value_name("nonces")
                .help("how many nonces you want to plot")
                .takes_value(true)
                .required_unless("ocl-devices"),
        )
        .arg(
            Arg::with_name("path")
                .short("p")
                .long("path")
                .value_name("path")
                .help("final directory for plot file (HDD storage)")
                .takes_value(true)
                .required(false),
        )
        .arg(
            Arg::with_name("temp")
                .short("t")
                .long("temp")
                .value_name("temp_dir")
                .help("temporary SSD directory for fast plotting (file will be moved to --path after completion)")
                .takes_value(true)
                .required(false),
        )
        .arg(
            Arg::with_name("memory")
                .short("m")
                .long("mem")
                .value_name("memory")
                .help("maximum memory usage (optional)")
                .takes_value(true)
                .required(false),
        )
        .args(&[
            Arg::with_name("cpu")
                .short("c")
                .long("cpu")
                .value_name("threads")
                .help("maximum cpu threads you want to use (optional)")
                .required(false)
                .takes_value(true),
            #[cfg(feature = "opencl")]
            Arg::with_name("gpu")
                .short("g")
                .long("gpu")
                .value_name("platform_id:device_id:cores")
                .help("GPU(s) you want to use for plotting (optional)")
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
                .help("Enables zero copy buffers for shared mem (integrated) gpus")
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
    let start_nonce = value_t!(matches, "start nonce", u64).unwrap_or_else(|e| e.exit());
    let nonces = value_t!(matches, "nonces", u64).unwrap_or_else(|e| e.exit());

    // Determine final and temporary paths
    let final_dir_str = value_t!(matches, "path", String).unwrap_or_else(|_| {
        std::env::current_dir()
            .unwrap()
            .into_os_string()
            .into_string()
            .unwrap()
    });
    let final_dir = PathBuf::from(&final_dir_str);

    let temp_dir: Option<PathBuf> = matches.value_of("temp").map(PathBuf::from);

    // Effective output path passed to plotter
    let effective_output_path = if let Some(ref temp) = temp_dir {
        temp.clone()
    } else {
        final_dir.clone()
    };

    let output_path_str = effective_output_path
        .into_os_string()
        .into_string()
        .unwrap();

    let mem = value_t!(matches, "memory", String).unwrap_or_else(|_| "0B".to_owned());
    let cpu_threads = value_t!(matches, "cpu", u8).unwrap_or(0u8);

    let gpus = if matches.occurrences_of("gpu") > 0 {
        let gpu = values_t!(matches, "gpu", String);
        Some(gpu.unwrap())
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

    // Inform user
    if temp_dir.is_some() {
        println!("Plotting to SSD temp: {}", output_path_str);
        println!("Will move to final dir after completion: {}", final_dir_str);
    }

    // Run plotting
    let p = Plotter::new();
    p.run(PlotterTask {
        numeric_id,
        start_nonce,
        nonces,
        output_path: output_path_str,
        mem,
        cpu_threads,
        gpus,
        direct_io: !matches.is_present("disable direct i/o"),
        async_io: !matches.is_present("disable async i/o"),
        quiet: matches.is_present("non-verbosity"),
        benchmark: matches.is_present("benchmark"),
        zcb: matches.is_present("zero-copy"),
    });

    // After plotting: move file if temp was used
    if let Some(temp_dir_path) = temp_dir {
        // Construct expected filename: {id}_{start}_{nonces}_{scoops}
        // Signum PoC2 uses 4096 scoops
        let scoops = 4096;
        let filename = format!("{}_{}_{}_{}", numeric_id, start_nonce, nonces, scoops);

        let temp_file = temp_dir_path.join(&filename);
        let final_file = final_dir.join(&filename);

        // Ensure final directory exists
        if let Err(e) = fs::create_dir_all(&final_dir) {
            eprintln!("Failed to create final directory: {}", e);
            return;
        }

        println!("Plotting complete. Moving file to final destination...");

        // Try rename first (fast, atomic if same filesystem)
        if let Err(e) = fs::rename(&temp_file, &final_file) {
            if e.kind() == std::io::ErrorKind::CrossesDevices || e.kind() == std::io::ErrorKind::PermissionDenied {
                // Fallback: copy then delete
                println!("Cross-device move detected. Copying instead...");
                if let Err(e) = fs::copy(&temp_file, &final_file) {
                    eprintln!("Copy failed: {}", e);
                    return;
                }
                if let Err(e) = fs::remove_file(&temp_file) {
                    eprintln!("Failed to delete temp file (manual cleanup needed): {}", e);
                }
            } else {
                eprintln!("Move failed: {}", e);
                return;
            }
        }

        println!("Plot successfully moved to: {}", final_file.display());
    } else {
        println!("Plotting complete.");
    }
}
