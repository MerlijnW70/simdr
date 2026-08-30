use runner::{Error, Gpu};

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
