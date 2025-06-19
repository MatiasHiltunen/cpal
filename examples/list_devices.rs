use cpal::traits::{DeviceTrait, HostTrait};
use cpal::platform::DeviceInner;

fn print_supported_configs(device_index: usize, device: &cpal::Device) {
    // Input configs
    if let Ok(conf) = device.default_input_config() {
        println!("    Default input stream config:\n      {:?}", conf);
    }
    match device.supported_input_configs() {
        Ok(configs) => {
            let list: Vec<_> = configs.collect();
            if !list.is_empty() {
                println!("    All supported input stream configs:");
                for (config_index, config) in list.into_iter().enumerate() {
                    println!(
                        "      {}.{}. {:?}",
                        device_index + 1,
                        config_index + 1,
                        config
                    );
                }
            }
        }
        Err(e) => println!("    Error getting supported input configs: {:?}", e),
    }

    // Output configs
    if let Ok(conf) = device.default_output_config() {
        println!("    Default output stream config:\n      {:?}", conf);
    }
    match device.supported_output_configs() {
        Ok(configs) => {
            let list: Vec<_> = configs.collect();
            if !list.is_empty() {
                println!("    All supported output stream configs:");
                for (config_index, config) in list.into_iter().enumerate() {
                    println!(
                        "      {}.{}. {:?}",
                        device_index + 1,
                        config_index + 1,
                        config
                    );
                }
            }
        }
        Err(e) => println!("    Error getting supported output configs: {:?}", e),
    }
}

// -----------------------------------------------------------------------------
// macOS implementation (rich metadata)
// -----------------------------------------------------------------------------

#[cfg(target_os = "macos")]
fn main() {
    let host = cpal::default_host();

    println!("Host: {:?}", host.id());
    println!("-----------------------\n");

    // Show the system-default devices first (if any).
    use cpal::traits::HostTrait as _;
    if let Some(def_out) = host.default_output_device() {
        if let cpal::platform::DeviceInner::CoreAudio(dev) = def_out.as_inner() {
            if let Ok(meta) = dev.metadata() {
                println!("*** SYSTEM DEFAULT OUTPUT ***");
                println!("{}", meta.name);
                println!("    UID          : {:?}", meta.uid);
                println!("    Transport    : {:?}", meta.transport_type);
                println!("    Is Alive     : {}", meta.is_alive);
                println!("    Default Dev? : {}", meta.is_default);
                println!("\n");
                print_supported_configs(0, &def_out);
            }
        }
    }
    if let Some(def_in) = host.default_input_device() {
        if let cpal::platform::DeviceInner::CoreAudio(dev) = def_in.as_inner() {
            if let Ok(meta) = dev.metadata() {
                println!("*** SYSTEM DEFAULT INPUT  ***");
                println!("{}", meta.name);
                println!("    UID          : {:?}", meta.uid);
                println!("    Transport    : {:?}", meta.transport_type);
                println!("    Is Alive     : {}", meta.is_alive);
                println!("    Default Dev? : {}", meta.is_default);
                println!("\n");
                print_supported_configs(0, &def_in);
            }
        }
    }

    match host.devices() {
        Ok(devices) => {
            for (idx, device) in devices.enumerate() {
                if let DeviceInner::CoreAudio(inner) = device.as_inner() {
                    match inner.metadata() {
                        Ok(meta) => {
                            println!("{}", meta.name);
                            println!("    UID          : {:?}", meta.uid);
                            println!("    Transport    : {:?}", meta.transport_type);
                            println!("    Is Alive     : {}", meta.is_alive);
                            println!("    Default Dev? : {}", meta.is_default);
                            println!("\n");
                            print_supported_configs(idx, &device);
                        }
                        Err(err) => {
                            let name = inner.name().unwrap_or_else(|_| "<Unknown>".into());
                            println!("{} (metadata error: {})\n", name, err.description);
                        }
                    }
                } else {
                    // Shouldn't happen: on macOS default host is CoreAudio.
                    let name = device.name().unwrap_or_else(|_| "<Unknown>".into());
                    println!("{}", name);
                    print_supported_configs(0, &device);
                }
            }
        }
        Err(e) => eprintln!("Failed to enumerate devices: {e}"),
    }
}

// -----------------------------------------------------------------------------
// Fallback implementation for non-macOS targets (builds but prints a message)
// -----------------------------------------------------------------------------

#[cfg(not(target_os = "macos"))]
fn main() {
    println!("The `list_devices` example is intended for macOS as it relies on Core Audio-specific metadata APIs.");
} 