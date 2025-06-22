//! A simple VoIP example that sends audio recorded from the microphone to a remote peer, and
//! plays audio received from the peer.
//!
//! This is a peer-to-peer application. You will need to run this example on two devices.
//!
//! On the first device, run:
//! ```sh
//! cargo run --example voip -- 0.0.0.0:5001 127.0.0.1:5002
//! ```
//! Change `127.0.0.1` to the IP address of the second device.
//! You can find your IP address by running `ipconfig` on Windows or `ifconfig` or `ip addr` on Linux/macOS.
//!
//! On the second device, run:
//! ```sh
//! cargo run --example voip -- 0.0.0.0:5002 127.0.0.1:5001
//! ```
//! Make sure to change `127.0.0.1` to the IP address of the first device.
//!
//! The example will use the default input and output audio devices.

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::collections::VecDeque;
use std::net::UdpSocket;
use std::sync::{Arc, Mutex};
use std::thread;

fn main() -> Result<(), anyhow::Error> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: {} <BIND_ADDR:PORT> <REMOTE_ADDR:PORT>", args[0]);
        return Ok(());
    }
    let bind_addr = &args[1];
    let remote_addr = &args[2];

    // Create a UDP socket.
    let socket = UdpSocket::bind(bind_addr)?;
    socket.connect(remote_addr)?;
    let send_socket = socket.try_clone()?;

    // Set up CPAL host and devices.
    let host = cpal::default_host();
    let input_device = host
        .default_input_device()
        .expect("no input device available");
    let output_device = host
        .default_output_device()
        .expect("no output device available");

    println!("Using input device: {}", input_device.name()?);
    println!("Using output device: {}", output_device.name()?);

    // Find a supported stream configuration.
    let config = input_device
        .supported_input_configs()?
        .filter(|c| c.sample_format() == cpal::SampleFormat::F32)
        .find_map(|c| {
            output_device
                .supported_output_configs()
                .ok()?
                .find(|d| d.channels() == c.channels() && d.sample_format() == c.sample_format() && d.min_sample_rate() <= c.max_sample_rate() && d.max_sample_rate() >= c.min_sample_rate())
                .map(|d| {
                    let s_rate = c.max_sample_rate().clamp(d.min_sample_rate(), d.max_sample_rate());
                    cpal::StreamConfig {
                        channels: c.channels(),
                        sample_rate: s_rate,
                        buffer_size: cpal::BufferSize::Default,
                    }
                })
        })
        .ok_or_else(|| anyhow::anyhow!("No supported stream configuration found"))?;

    println!("Using config: {:?}", config);

    // Shared buffer for received audio data.
    let received_audio_buffer = Arc::new(Mutex::new(VecDeque::<f32>::new()));

    // --- Output Stream ---
    // Clones the buffer for the playback thread.
    let buffer_for_playback = received_audio_buffer.clone();
    let output_data_fn = move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
        let mut buffer = buffer_for_playback.lock().unwrap();
        let len = std::cmp::min(data.len(), buffer.len());

        // Write samples from buffer to the output stream.
        for (i, sample) in data.iter_mut().enumerate().take(len) {
            *sample = buffer.get(i).copied().unwrap_or(0.0);
        }
        // Remove the played samples from the buffer.
        buffer.drain(..len);

        // Fill the rest of the buffer with silence.
        for sample in data.iter_mut().skip(len) {
            *sample = 0.0;
        }
    };

    // --- Input Stream ---
    // The input stream callback, which sends audio data over the network.
    let input_data_fn = move |data: &[f32], _: &cpal::InputCallbackInfo| {
        // Convert f32 samples to bytes.
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const u8,
                data.len() * std::mem::size_of::<f32>(),
            )
        };

        // Send the audio data over the UDP socket.
        if let Err(e) = send_socket.send(bytes) {
            eprintln!("Error sending data: {}", e);
        }
    };

    let err_fn = |err| eprintln!("an error occurred on stream: {}", err);

    // Build and play the streams.
    let input_stream =
        input_device.build_input_stream(&config, input_data_fn, err_fn, None)?;
    let output_stream =
        output_device.build_output_stream(&config, output_data_fn, err_fn, None)?;

    input_stream.play()?;
    output_stream.play()?;

    // --- UDP Receiver Thread ---
    // This thread receives audio data from the peer and puts it into the shared buffer.
    let buffer_for_receiving = received_audio_buffer;
    thread::spawn(move || {
        // A buffer to store received UDP packets.
        let mut recv_buf = [0u8; 8192];
        loop {
            match socket.recv(&mut recv_buf) {
                Ok(num_bytes) => {
                    // Ensure the received data is a multiple of f32 size.
                    if num_bytes % std::mem::size_of::<f32>() == 0 {
                        let samples: &[f32] = unsafe {
                            std::slice::from_raw_parts(
                                recv_buf.as_ptr() as *const f32,
                                num_bytes / std::mem::size_of::<f32>(),
                            )
                        };
                        let mut buffer = buffer_for_receiving.lock().unwrap();
                        buffer.extend(samples);
                    } else {
                        eprintln!("Received UDP packet with size not a multiple of f32 size.");
                    }
                }
                Err(e) => {
                    eprintln!("Error receiving data: {}", e);
                    // Connection was likely closed.
                    break;
                }
            }
        }
    });

    println!("\nVoIP example started.");
    println!("Speaking into the microphone will send audio to {}.", remote_addr);
    println!("Audio from the remote peer will be played on your speakers.");
    println!("Press Ctrl+C to exit.");

    // Block the main thread indefinitely.
    thread::park();

    Ok(())
} 