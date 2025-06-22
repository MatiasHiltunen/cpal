//! A simple peer-to-peer VoIP demo that compresses audio with Opus over UDP.
//!
//! Build and run on two machines (or two terminals on localhost):
//!
//! ```bash
//! cargo run --example voip_opus -- 0.0.0.0:5001 192.168.0.42:5002
//! ```
//! ...and on the peer:
//! ```bash
//! cargo run --example voip_opus -- 0.0.0.0:5002 192.168.0.41:5001
//! ```
//!
//! Requirements:
//!   • `cmake` ‑ needed by the `audiopus` crate to build a bundled libopus.
//!
//! The demo picks a common stream configuration supported by both the default
//! input and output devices and uses Opus to compress 20 ms audio frames.
//! Each UDP packet contains a 16-bit sequence number followed by the Opus
//! payload.
//!
//! Bandwidth: ≈ 40 kbit/s mono @ 48 kHz, application = Audio.

use anyhow::{anyhow, Context, Result};
use audiopus::{coder::{Decoder as OpusDecoder, Encoder as OpusEncoder}, Application, SampleRate, Bitrate};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::collections::VecDeque;
use std::net::UdpSocket;
use std::sync::{Arc, Mutex};
use std::thread;

/// Duration of one Opus frame in milliseconds.
const FRAME_MS: u32 = 20; // permitted values: 2.5, 5, 10, 20, 40, 60
const MAX_PACKET_SIZE: usize = 1500; // bytes (safe upper-bound for UDP MTU)

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: {} <BIND_ADDR:PORT> <REMOTE_ADDR:PORT>", args[0]);
        return Ok(());
    }
    let bind_addr = &args[1];
    let remote_addr = &args[2];

    // --- networking ---
    let socket = UdpSocket::bind(bind_addr).with_context(|| format!("Bind {bind_addr}"))?;
    socket
        .connect(remote_addr)
        .with_context(|| format!("Connect {remote_addr}"))?;
    socket.set_nonblocking(true)?;
    let send_socket = socket.try_clone()?;

    // --- audio devices ---
    let host = cpal::default_host();
    let in_dev = host.default_input_device().context("No input device")?;
    let out_dev = host.default_output_device().context("No output device")?;

    println!("Input device : {}", in_dev.name()?);
    println!("Output device: {}", out_dev.name()?);

    // Pick a common configuration (f32 sample format).
    let (config, channels) = pick_common_config(&in_dev, &out_dev)?;
    let sample_rate = config.sample_rate.0;
    let opus_channels = if channels == 1 {
        audiopus::Channels::Mono
    } else {
        audiopus::Channels::Stereo
    };

    // Map CPAL sample rate to Opus enum
    let opus_sample_rate = match sample_rate {
        8000 => SampleRate::Hz8000,
        12000 => SampleRate::Hz12000,
        16000 => SampleRate::Hz16000,
        24000 => SampleRate::Hz24000,
        48000 => SampleRate::Hz48000,
        _ => return Err(anyhow!("Unsupported sample rate {sample_rate}")),
    };

    println!("Using config  : {:?}", config);

    // --- Opus coder/decoder ---
    let mut opus_enc = OpusEncoder::new(opus_sample_rate, opus_channels, Application::Audio)?;
    opus_enc.set_bitrate(Bitrate::BitsPerSecond(40_000))?;
    let mut opus_dec = OpusDecoder::new(opus_sample_rate, opus_channels)?;

    let frame_samples = (sample_rate as u32 / 1000 * FRAME_MS) as usize * channels as usize;

    // Shared buffer for decoded PCM to be played.
    let pcm_buffer: Arc<Mutex<VecDeque<f32>>> = Arc::new(Mutex::new(VecDeque::with_capacity(sample_rate as usize)));

    // ---------- CPAL OUTPUT STREAM ----------
    let buf_play = pcm_buffer.clone();
    let out_stream = out_dev.build_output_stream(
        &config,
        move |out: &mut [f32], _| {
            let mut buf = buf_play.lock().unwrap();
            for sample in out {
                *sample = buf.pop_front().unwrap_or(0.0);
            }
        },
        err_callback,
        None,
    )?;

    // ---------- CPAL INPUT STREAM ----------
    // We need to accumulate EXACTLY `frame_samples` before encoding.
    let mut frame_accum: Vec<f32> = Vec::with_capacity(frame_samples * 2);
    let input_send_socket = send_socket.try_clone().expect("Failed to clone UDP socket");
    let mut seq_no: u16 = 0;
    let in_stream = in_dev.build_input_stream(
        &config,
        move |input: &[f32], _| {
            frame_accum.extend_from_slice(input);
            while frame_accum.len() >= frame_samples {
                let frame: Vec<f32> = frame_accum.drain(..frame_samples).collect();
                // Prepare packet buffer (seq + opus data).
                let mut packet = [0u8; MAX_PACKET_SIZE];
                packet[..2].copy_from_slice(&seq_no.to_be_bytes());
                seq_no = seq_no.wrapping_add(1);
                match opus_enc.encode_float(&frame, &mut packet[2..]) {
                    Ok(encoded) => {
                        let _ = input_send_socket.send(&packet[..2 + encoded]);
                    }
                    Err(e) => eprintln!("Opus encode error: {e}"),
                }
            }
        },
        err_callback,
        None,
    )?;

    // ---------- RECEIVER THREAD (UDP → DECODE → BUFFER) ----------
    let recv_buf = pcm_buffer.clone();
    let ch = channels as usize;
    let sr = sample_rate;
    thread::spawn(move || {
        let mut pkt = [0u8; MAX_PACKET_SIZE];
        loop {
            match socket.recv(&mut pkt) {
                Ok(n) if n > 2 => {
                    let payload = &pkt[2..n];
                    let max_out_samples = (sr as usize / 1000 * 60) * ch; // up to 60 ms
                    let mut pcm = vec![0f32; max_out_samples];
                    match opus_dec.decode_float(Some(payload), &mut pcm, false) {
                        Ok(decoded) => {
                            let mut buf = recv_buf.lock().unwrap();
                            buf.extend(pcm[..decoded * ch].iter().copied());
                        }
                        Err(e) => eprintln!("Opus decode error: {e}"),
                    }
                }
                Ok(_) => { /* too small, drop */ }
                Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    // no data – yield
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
                Err(e) => {
                    eprintln!("UDP recv error: {e}");
                    break;
                }
            }
        }
    });

    // Start streams
    in_stream.play()?;
    out_stream.play()?;

    println!("Running – press Ctrl-C to quit.");
    thread::park();
    Ok(())
}

fn err_callback(err: cpal::StreamError) {
    eprintln!("Stream error: {err}");
}

/// Find a common f32 stream config between input and output devices.
fn pick_common_config(in_dev: &cpal::Device, out_dev: &cpal::Device) -> Result<(cpal::StreamConfig, cpal::ChannelCount)> {
    for in_cfg in in_dev.supported_input_configs()? {
        if in_cfg.sample_format() != cpal::SampleFormat::F32 {
            continue;
        }
        for out_cfg in out_dev.supported_output_configs()? {
            if out_cfg.sample_format() != cpal::SampleFormat::F32 {
                continue;
            }
            let ch = in_cfg.channels().min(out_cfg.channels());
            let min_rate = in_cfg.min_sample_rate().0.max(out_cfg.min_sample_rate().0);
            let max_rate = in_cfg.max_sample_rate().0.min(out_cfg.max_sample_rate().0);
            // Choose 48 k if inside the common range, otherwise any common rate.
            let rate = if (48_000 >= min_rate) && (48_000 <= max_rate) {
                48_000
            } else {
                min_rate
            };
            if min_rate <= max_rate {
                return Ok((cpal::StreamConfig {
                    channels: ch,
                    sample_rate: cpal::SampleRate(rate),
                    buffer_size: cpal::BufferSize::Default,
                }, ch));
            }
        }
    }
    Err(anyhow!("No common f32 config between devices"))
} 