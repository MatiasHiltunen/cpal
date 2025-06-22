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
use cpal::{Sample, SampleFormat, SizedSample, FromSample};
use std::collections::VecDeque;
use std::net::UdpSocket;
use std::sync::{Arc, Mutex};
use std::thread;
use std::cmp;

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

    // Helper to find a common stream configuration between the chosen input/output devices.
    fn find_common_config(
        input_device: &cpal::Device,
        output_device: &cpal::Device,
    ) -> Result<(cpal::StreamConfig, SampleFormat), anyhow::Error> {
        // Start with the default input config as our first candidate.
        let in_def = input_device.default_input_config()?;

        // Attempt to find an output configuration that matches sample-format + channels and
        // supports the same sample-rate.
        for out_range in output_device.supported_output_configs()? {
            if out_range.sample_format() == in_def.sample_format()
                && out_range.channels() == in_def.channels()
                && out_range.min_sample_rate() <= in_def.sample_rate()
                && out_range.max_sample_rate() >= in_def.sample_rate()
            {
                let cfg = cpal::StreamConfig {
                    channels: in_def.channels(),
                    sample_rate: in_def.sample_rate(),
                    buffer_size: cpal::BufferSize::Default,
                };
                return Ok((cfg, in_def.sample_format()));
            }
        }

        // As a fallback, iterate through *all* supported input configs until we find a compatible
        // output config.
        for in_range in input_device.supported_input_configs()? {
            for out_range in output_device.supported_output_configs()? {
                if in_range.sample_format() == out_range.sample_format()
                    && in_range.channels() == out_range.channels()
                    && in_range.min_sample_rate() <= out_range.max_sample_rate()
                {
                    // Select a sample-rate supported by both ranges.
                    let min_rate = cmp::max(in_range.min_sample_rate(), out_range.min_sample_rate());
                    let max_rate = cmp::min(in_range.max_sample_rate(), out_range.max_sample_rate());
                    if min_rate <= max_rate {
                        let cfg = cpal::StreamConfig {
                            channels: in_range.channels(),
                            sample_rate: max_rate, // pick highest common rate.
                            buffer_size: cpal::BufferSize::Default,
                        };
                        return Ok((cfg, in_range.sample_format()));
                    }
                }
            }
        }

        Err(anyhow::anyhow!("No supported stream configuration found"))
    }

    let (config, sample_format) = match find_common_config(&input_device, &output_device) {
        Ok(v) => {
            println!("Using common config: {:?}", v.0);
            v
        }
        Err(_e) => {
            // Fallback: use default input/output configs and handle conversion.
            let in_sup = input_device.default_input_config()?;
            let out_sup = output_device.default_output_config()?;
            println!(
                "Using fallback configs -> input: {:?}, output: {:?}",
                in_sup, out_sup
            );

            return run_voip_convert(
                input_device, output_device, in_sup.into(), out_sup.into(), send_socket, socket);
        }
    };

    // If we got here, we have matching sample format and channels.
    match sample_format {
        SampleFormat::F32 => run_voip::<f32>(
            input_device,
            output_device,
            config,
            send_socket,
            socket,
        ),
        SampleFormat::I16 => run_voip::<i16>(
            input_device,
            output_device,
            config,
            send_socket,
            socket,
        ),
        SampleFormat::U16 => run_voip::<u16>(
            input_device,
            output_device,
            config,
            send_socket,
            socket,
        ),
        sample_format => Err(anyhow::anyhow!(
            "Unsupported sample format '{sample_format}'"
        )),
    }
}

// Generic implementation for each supported sample type.
fn run_voip<T>(
    input_device: cpal::Device,
    output_device: cpal::Device,
    config: cpal::StreamConfig,
    send_socket: std::net::UdpSocket,
    socket: std::net::UdpSocket,
) -> Result<(), anyhow::Error>
where
    T: Sample + SizedSample + Send + 'static,
{
    // Shared buffer for received audio data.
    let received_audio_buffer = Arc::new(Mutex::new(VecDeque::<T>::new()));

    // --- Output Stream ---
    // Clones the buffer for the playback thread.
    let buffer_for_playback = received_audio_buffer.clone();
    let output_data_fn = move |data: &mut [T], _: &cpal::OutputCallbackInfo| {
        let mut buffer = buffer_for_playback.lock().unwrap();
        let len = std::cmp::min(data.len(), buffer.len());

        // Write samples from buffer to the output stream.
        for sample in data.iter_mut().take(len) {
            *sample = buffer.pop_front().unwrap_or(Sample::EQUILIBRIUM);
        }

        // Fill the rest of the buffer with silence (equilibrium).
        for sample in data.iter_mut().skip(len) {
            *sample = Sample::EQUILIBRIUM;
        }
    };

    // --- Input Stream ---
    // The input stream callback, which sends audio data over the network.
    let input_data_fn = move |data: &[T], _: &cpal::InputCallbackInfo| {
        // Convert samples to bytes.
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const u8,
                data.len() * std::mem::size_of::<T>(),
            )
        };

        // Send the audio data over the UDP socket.
        if let Err(e) = send_socket.send(bytes) {
            eprintln!("Error sending data: {}", e);
        }
    };

    let err_fn = |err| eprintln!("an error occurred on stream: {}", err);

    // Build and play the streams.
    let input_stream = input_device.build_input_stream(&config, input_data_fn, err_fn, None)?;
    let output_stream = output_device.build_output_stream(&config, output_data_fn, err_fn, None)?;

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
                    // Ensure the received data is a multiple of T size.
                    if num_bytes % std::mem::size_of::<T>() == 0 {
                        let samples: &[T] = unsafe {
                            std::slice::from_raw_parts(
                                recv_buf.as_ptr() as *const T,
                                num_bytes / std::mem::size_of::<T>(),
                            )
                        };
                        let mut buffer = buffer_for_receiving.lock().unwrap();
                        buffer.extend(samples);
                    } else {
                        eprintln!("Received UDP packet with incorrect size (not multiple of sample size).");
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
    println!("Speaking into the microphone will send audio to the remote peer.");
    println!("Audio from the remote peer will be played on your speakers.");
    println!("Press Ctrl+C to exit.");

    // Block the main thread indefinitely.
    thread::park();

    Ok(())
}

// Fallback path that converts between differing input/output formats and channels using f32 as
// the network representation.
#[allow(clippy::too_many_arguments)]
fn run_voip_convert(
    input_device: cpal::Device,
    output_device: cpal::Device,
    in_config: cpal::StreamConfig,
    out_config: cpal::StreamConfig,
    send_socket: std::net::UdpSocket,
    socket: std::net::UdpSocket,
) -> Result<(), anyhow::Error> {
    // Shared mono buffer of f32 samples received from the network.
    let received_audio_buffer = Arc::new(Mutex::new(VecDeque::<f32>::new()));

    // --------------------------------------------------
    // Build INPUT stream (device -> network, possibly format conversion)
    // --------------------------------------------------
    let channels_in = in_config.channels as usize;

    let err_fn = |err| eprintln!("an error occurred on stream: {}", err);

    // Helper: build input stream for each possible sample format.
    let input_stream = match input_device.default_input_config()?.sample_format() {
        SampleFormat::F32 => {
            input_device.build_input_stream(&in_config, move |data: &[f32], _| {
                send_input_as_f32(data, channels_in, &send_socket);
            }, err_fn, None)?
        }
        SampleFormat::I16 => {
            input_device.build_input_stream(&in_config, move |data: &[i16], _| {
                send_input_as_f32(data, channels_in, &send_socket);
            }, err_fn, None)?
        }
        SampleFormat::U16 => {
            input_device.build_input_stream(&in_config, move |data: &[u16], _| {
                send_input_as_f32(data, channels_in, &send_socket);
            }, err_fn, None)?
        }
        sf => return Err(anyhow::anyhow!("Unsupported input sample format '{sf}'")),
    };

    // --------------------------------------------------
    // Build OUTPUT stream (network -> device, possibly format conversion)
    // --------------------------------------------------
    let channels_out = out_config.channels as usize;
    let buffer_for_playback = received_audio_buffer.clone();

    let output_stream = match output_device.default_output_config()?.sample_format() {
        SampleFormat::F32 => {
            output_device.build_output_stream(&out_config, move |data: &mut [f32], _| {
                play_from_buffer::<f32>(data, channels_out, &buffer_for_playback);
            }, err_fn, None)?
        }
        SampleFormat::I16 => {
            output_device.build_output_stream(&out_config, move |data: &mut [i16], _| {
                play_from_buffer::<i16>(data, channels_out, &buffer_for_playback);
            }, err_fn, None)?
        }
        SampleFormat::U16 => {
            output_device.build_output_stream(&out_config, move |data: &mut [u16], _| {
                play_from_buffer::<u16>(data, channels_out, &buffer_for_playback);
            }, err_fn, None)?
        }
        sf => return Err(anyhow::anyhow!("Unsupported output sample format '{sf}'")),
    };

    input_stream.play()?;
    output_stream.play()?;

    // --------------------------------------------------
    // UDP receiver thread -> push f32 samples into buffer
    // --------------------------------------------------
    let buffer_for_receiving = received_audio_buffer;
    thread::spawn(move || {
        let mut recv_buf = [0u8; 8192];
        loop {
            match socket.recv(&mut recv_buf) {
                Ok(num_bytes) => {
                    if num_bytes % std::mem::size_of::<f32>() == 0 {
                        let samples: &[f32] = unsafe {
                            std::slice::from_raw_parts(
                                recv_buf.as_ptr() as *const f32,
                                num_bytes / std::mem::size_of::<f32>(),
                            )
                        };
                        if let Ok(mut buf) = buffer_for_receiving.lock() {
                            buf.extend(samples);
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Error receiving data: {}", e);
                    break;
                }
            }
        }
    });

    println!("\nVoIP example started (fallback path).");
    println!("Press Ctrl+C to exit.");

    thread::park();

    Ok(())
}

// ---------- helper functions for fallback ----------

fn send_input_as_f32<In>(data: &[In], channels: usize, socket: &UdpSocket)
where
    f32: FromSample<In>,
    In: Sample,
{
    // Convert interleaved input to mono f32 by taking first channel.
    let frames = data.len() / channels;
    let mut out_vec = Vec::<f32>::with_capacity(frames);
    for frame_idx in 0..frames {
        let sample_in: In = data[frame_idx * channels];
        let s: f32 = f32::from_sample(sample_in);
        out_vec.push(s);
    }

    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(out_vec.as_ptr() as *const u8, out_vec.len() * std::mem::size_of::<f32>())
    };
    let _ = socket.send(bytes);
}

fn play_from_buffer<Out>(data: &mut [Out], channels: usize, buffer: &Arc<Mutex<VecDeque<f32>>>)
where
    Out: Sample + FromSample<f32>,
{
    let mut buf = buffer.lock().unwrap();
    let frames = data.len() / channels;
    for frame_idx in 0..frames {
        let sample_f32 = buf.pop_front().unwrap_or(0.0);
        let out_sample: Out = Out::from_sample(sample_f32);
        for ch in 0..channels {
            data[frame_idx * channels + ch] = out_sample;
        }
    }
} 