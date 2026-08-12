//! `simdr probe` — ask a device what has to be known before a module can be built.
//!
//! # Why this exists, and why it stays small
//!
//! `decisions/DR-0002` is the reason. A module is specialised to a subgroup width, the width is
//! fixed by the hardware, and `Lanes::new` takes it as an argument so that no caller can forget
//! it exists. Which left an awkward gap: the only way to find out what to pass was to write a Rust
//! program that opens a device.
//!
//! So the design forces a question and there was no way to ask it. That is what this answers.
//!
//! `simdr list` is the second subcommand and arrived for the same reason: this machine turned out
//! to have two devices at two different subgroup widths, and there was no way to see the second
//! one without writing a program either.
//!
//! It stops there. A `simdr validate` would be three lines around `spirv-val` and is better as a
//! line in the README; a `simdr emit` would need a kernel description language, which is a second
//! and worse API beside the one that already exists — the value of the library is that kernels are
//! Rust with types. Benchmarks are `cargo run --example`.
//!
//! # What it reports, and why each line is there
//!
//! - **Subgroup width**, because nothing can be built without it.
//! - **Which subgroup features**, because a kernel declaring a capability the device lacks fails
//!   at *pipeline creation* rather than at validation — later and less clearly than it should.
//! - **Narrow element types**, because they are six separate permissions and one of them —
//!   `shaderSubgroupExtendedTypes` — leaves no trace in the module at all.
//! - **Memory types**, because a host-visible type without `HOST_CACHED` is write-combined and
//!   reading a mapping back out of one runs at a fraction of the bus. That cost 8× on transfers
//!   here before anybody looked, and looking is now one command.

// A tool that reports on a device should not be the thing that crashes. Every failure below is an
// exit code and a sentence on stderr.
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use runner::Gpu;
use std::process::ExitCode;

/// What the arguments asked for.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Command {
    /// Describe the device.
    Probe,
    /// Name every device that could run compute work.
    ///
    /// Worth its own subcommand because a machine with two GPUs usually has two *subgroup widths*,
    /// and until there was a way to see the second one, only the first had ever been built for.
    List,
    /// Explain the tool.
    Help,
    /// Something that is not a subcommand, kept so the message can name it.
    Unknown(String),
}

/// Decide what to do, from the arguments after the program name.
///
/// Separated from `main` so it can be tested: it is the only thing here that decides anything, and
/// everything else prints. No arguments at all means `probe`, because there is one useful thing to
/// do and making a reader type it adds nothing.
fn parse(arguments: &[String]) -> Command {
    match arguments.first().map(String::as_str) {
        None | Some("probe") => Command::Probe,
        Some("list") => Command::List,
        Some("--help" | "-h" | "help") => Command::Help,
        Some(other) => Command::Unknown(other.to_owned()),
    }
}

fn main() -> ExitCode {
    let arguments: Vec<String> = std::env::args().skip(1).collect();

    match parse(&arguments) {
        Command::Probe => probe(),
        Command::List => list(),
        Command::Help => {
            usage();
            ExitCode::SUCCESS
        }
        Command::Unknown(name) => {
            eprintln!("simdr: no subcommand `{name}`\n");
            usage();
            ExitCode::FAILURE
        }
    }
}

/// What this tool does, in the form a reader can act on.
fn usage() {
    println!("simdr — what a device offers, before a module is built for it");
    println!();
    println!("USAGE:");
    println!("  simdr probe     report the subgroup width, the features, and the memory types");
    println!("  simdr list      name every device that could run compute work");
    println!();
    println!("The subgroup width is the one number a kernel cannot be built without: see");
    println!("decisions/DR-0002. Pass it to `Lanes::new` or `Shape::new`.");
}

/// Name every device that could run compute work.
fn list() -> ExitCode {
    let names = match Gpu::names() {
        Ok(names) => names,
        Err(error) => {
            eprintln!("could not enumerate devices: {error}");
            return ExitCode::FAILURE;
        }
    };

    if names.is_empty() {
        eprintln!("no Vulkan device with a compute queue");
        return ExitCode::FAILURE;
    }

    for name in &names {
        println!("{name}");
    }
    println!();
    println!("`simdr probe` describes whichever of these SIMDR_DEVICE names, or the discrete one.");
    println!("SIMDR_DEVICE matches a substring, case-insensitively: SIMDR_DEVICE=radeon");
    ExitCode::SUCCESS
}

/// Open the first compute device and describe it.
fn probe() -> ExitCode {
    let gpu = match Gpu::open() {
        Ok(Some(gpu)) => gpu,
        Ok(None) => {
            eprintln!("no Vulkan device with a compute queue");
            return ExitCode::FAILURE;
        }
        Err(error) => {
            eprintln!("could not open a device: {error}");
            return ExitCode::FAILURE;
        }
    };

    let limits = gpu.limits();
    println!("{}", limits.name);
    println!();
    println!("  subgroup width      {}", limits.subgroup_size);
    println!(
        "  timestamp period    {}",
        if limits.timestamp_period_ns > 0.0 {
            format!("{:.1} ns per tick", limits.timestamp_period_ns)
        } else {
            "not offered — timings fall back to the host clock".to_owned()
        }
    );

    println!("\n  subgroup features");
    for (name, present, needed_for) in [
        (
            "arithmetic",
            limits.subgroup_arithmetic,
            "reduce_sum, reduce_max, reduce_min, prefix_sum",
        ),
        (
            "clustered",
            limits.subgroup_clustered,
            "vectors narrower than the subgroup",
        ),
        (
            "shuffle",
            limits.subgroup_shuffle,
            "butterfly, broadcast, shift_up, shift_down",
        ),
        ("ballot", limits.subgroup_ballot, "any, all, ballot"),
    ] {
        println!(
            "    {:<12} {:<5}  {needed_for}",
            name,
            if present { "yes" } else { "NO" }
        );
    }

    narrow(limits);
    memory(&gpu);
    advice(limits.subgroup_size);
    ExitCode::SUCCESS
}

/// What this device offers for elements narrower than a lane.
///
/// Four flags rather than one line, because they are four permissions and a device may hold any
/// subset — `decisions/DR-0004`. The last of them is the one worth printing loudest: nothing in a
/// module says it is needed, so a kernel that reduces over `i8` looks fine everywhere and runs
/// only here.
fn narrow(limits: &runner::Limits) {
    let narrow = limits.narrow;
    println!("\n  narrow element types");
    for (name, present, needed_for) in [
        ("shaderInt8", narrow.int8, "arithmetic in i8 and u8"),
        ("shaderInt16", narrow.int16, "arithmetic in i16 and u16"),
        ("shaderFloat16", narrow.float16, "arithmetic in f16"),
        (
            "8-bit storage",
            narrow.storage8,
            "a buffer of i8 or u8 — a quarter of the bytes",
        ),
        (
            "16-bit storage",
            narrow.storage16,
            "a buffer of i16, u16 or f16",
        ),
        (
            "extended types",
            narrow.subgroup_extended_types,
            "reductions and shuffles over narrow types — no capability says so",
        ),
    ] {
        println!(
            "    {:<16} {:<5}  {needed_for}",
            name,
            if present { "yes" } else { "NO" }
        );
    }

    let usable: Vec<&str> = [
        ("i8, u8", narrow.byte_kernel()),
        ("i16, u16", narrow.short_kernel()),
        ("f16", narrow.half_kernel()),
    ]
    .into_iter()
    .filter_map(|(name, ok)| ok.then_some(name))
    .collect();

    if usable.is_empty() {
        println!("    → no narrow element type can both compute and be held in a buffer here");
    } else {
        println!("    → usable end to end: {}", usable.join(", "));
    }
}

/// The memory types, and whether the staging path can get a cached one.
fn memory(gpu: &Gpu) {
    let types = gpu.memory_types();
    println!("\n  memory types");
    println!(
        "    {:>5}  {:<13} {:<13} {:<14} host-cached",
        "index", "device-local", "host-visible", "host-coherent"
    );
    for kind in &types {
        println!(
            "    {:>5}  {:<13} {:<13} {:<14} {}",
            kind.index,
            yes_no(kind.device_local),
            yes_no(kind.host_visible),
            yes_no(kind.host_coherent),
            yes_no(kind.host_cached)
        );
    }

    let cached = types
        .iter()
        .any(|kind| kind.host_visible && kind.host_coherent && kind.host_cached);
    println!();
    println!(
        "    staging: {}",
        if cached {
            "a cached host-visible type exists, and the runner asks for it"
        } else {
            "no cached host-visible type — reading results back will be slow, \
             and that is the device rather than the code"
        }
    );
}

/// What to do with the number, so the output is not just facts.
fn advice(width: u32) {
    println!("\n  what to pass");
    println!("    Shape::new({width}, 64, 2)      a two-buffer kernel, 64 invocations");
    println!(
        "    Lanes::new(&mut module, {width})  if you are building against the module directly"
    );
    println!();
    println!(
        "    A Simd<T, {width}> maps one element per lane here. Narrower divides into clusters,"
    );
    println!("    wider strip-mines. Anything that is neither is refused by name.");
}

/// `yes` or `no`, aligned.
const fn yes_no(value: bool) -> &'static str {
    if value { "yes" } else { "no" }
}

#[cfg(test)]
mod tests {
    use super::{Command, parse, yes_no};

    fn arguments(words: &[&str]) -> Vec<String> {
        words.iter().map(|word| (*word).to_owned()).collect()
    }

    #[test]
    fn no_arguments_probes() {
        // There is one useful thing to do, so making a reader type it adds nothing.
        assert_eq!(parse(&[]), Command::Probe);
    }

    #[test]
    fn the_subcommand_is_taken_by_name() {
        assert_eq!(parse(&arguments(&["probe"])), Command::Probe);
        assert_eq!(parse(&arguments(&["list"])), Command::List);
        assert_eq!(parse(&arguments(&["help"])), Command::Help);
        assert_eq!(parse(&arguments(&["--help"])), Command::Help);
        assert_eq!(parse(&arguments(&["-h"])), Command::Help);
    }

    #[test]
    fn an_unknown_subcommand_is_carried_out_by_name() {
        // Named rather than swallowed: "no subcommand `prboe`" is a message somebody can act on
        // and "usage:" alone is not.
        assert_eq!(
            parse(&arguments(&["prboe"])),
            Command::Unknown("prboe".to_owned())
        );
    }

    #[test]
    fn later_arguments_do_not_change_the_subcommand() {
        assert_eq!(parse(&arguments(&["probe", "--verbose"])), Command::Probe);
    }

    #[test]
    fn the_two_answers_are_different() {
        assert_eq!(yes_no(true), "yes");
        assert_eq!(yes_no(false), "no");
    }
}
