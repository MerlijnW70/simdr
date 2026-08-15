//! Which device a run opens, and what happens when it asks for one that is not here.
//!
//! `SIMDR_DEVICE` is how the whole suite gets pointed at the other GPU in a machine, which is the
//! only way the second subgroup width ever runs — `decisions/DR-0002` argues that a module is built
//! for one width, and until that variable existed only one had ever been exercised.
//!
//! **So the variable is load-bearing, and it had no test at all.** A name that matched nothing
//! returned `Ok(None)`, exactly as a machine with no GPU does, and the suite skipped every test and
//! exited zero: `SIMDR_DEVICE=llvmpipe` here, where the two devices are called something else, was
//! 157 skips reporting `no Vulkan device` beside two Vulkan devices. Nothing was wrong with the
//! code under test; the run simply never happened, and said so in a channel `libtest` swallows.
//!
//! These tests cover the three answers `Gpu::open_matching` can give, on whatever this machine has.

use runner::{Error, Gpu};

/// Every device that could run compute work here, or `None` when this machine has none.
///
/// The skip is the honest one: a bare machine cannot say anything about device selection. It is
/// also the *only* skip in this file, which is the point of the file.
fn present(label: &str) -> Option<Vec<String>> {
    match Gpu::names() {
        Ok(names) if names.is_empty() => {
            eprintln!("SKIPPED {label}: no Vulkan device to choose between");
            None
        }
        Ok(names) => Some(names),
        Err(error) => {
            eprintln!("SKIPPED {label}: could not enumerate devices — {error}");
            None
        }
    }
}

#[test]
fn a_name_no_device_answers_to_is_an_error_and_not_an_empty_machine() {
    let Some(names) = present("selection-absent") else {
        return;
    };

    // Long enough that no vendor string will ever contain it, and readable in the failure if one
    // somehow does.
    let wanted = "there-is-no-device-called-this";
    let error = Gpu::open_matching(Some(wanted))
        .err()
        .expect("a name nothing answers to must not open a device");

    match error {
        Error::NoSuchDevice {
            wanted: asked,
            present,
        } => {
            assert_eq!(asked, wanted, "the error repeats what was asked for");
            // What is here, so the message tells the reader what to have typed instead. The same
            // function produces this list and the strings the filter matches against, so the two
            // cannot describe a device differently.
            assert_eq!(present, names, "the error names every device that is here");
        }
        other => panic!(
            "expected NoSuchDevice beside {} devices, got {other}",
            names.len()
        ),
    }
}

#[test]
fn a_name_a_device_answers_to_opens_that_device_whatever_its_case() {
    let Some(names) = present("selection-present") else {
        return;
    };

    // Upper-cased on purpose: the match is documented as case-insensitive, and `simdr list` prints
    // names nobody types back correctly — "AMD Radeon(TM) Graphics" twice in a row, in particular.
    let wanted = names[0].to_uppercase();
    let gpu = Gpu::open_matching(Some(&wanted))
        .expect("a name this machine answers to is not an error")
        .expect("nor an empty machine");

    assert_eq!(
        gpu.limits().name.to_lowercase(),
        names[0].to_lowercase(),
        "the device that opened is the one that was named"
    );
}

#[test]
fn no_name_opens_whatever_is_here() {
    let Some(names) = present("selection-default") else {
        return;
    };

    let gpu = Gpu::open_matching(None)
        .expect("choosing for itself is not an error")
        .expect("and this machine has a device");

    assert!(
        names.contains(&gpu.limits().name),
        "the device chosen with no pattern is one of {names:?}, not {:?}",
        gpu.limits().name
    );
}
