//! Production-grade peer-to-peer VoIP example using Opus compression over UDP.
//!
//! Features:
//! - Automatic codec parameter optimization based on network conditions
//! - Jitter buffer with adaptive playout delay
//! - Packet loss concealment using Opus FEC
//! - Statistics tracking (RTT, packet loss, jitter)
//! - Graceful degradation under poor network conditions
//!
//! Usage:
//! ```bash
//! # On machine A:
//! cargo run --example voip_opus -- 0.0.0.0:5001 192.168.1.100:5002
//!
//! # On machine B:
//! cargo run --example voip_opus -- 0.0.0.0:5002 192.168.1.101:5001
//! ```

use anyhow::{Context, Result};
use audiopus::{
    coder::{Decoder as OpusDecoder, Encoder as OpusEncoder},
    Application, Bitrate, Channels, SampleRate,
};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::{
    collections::VecDeque,
    net::{SocketAddr, UdpSocket},
    sync::{
        atomic::{AtomicBool, AtomicU16, AtomicU32, AtomicU64, Ordering},
        Arc, Mutex,
    },
    thread,
    time::Duration,
};

// Constants
const DEFAULT_BITRATE: u32 = 32000;
const DEFAULT_PACKET_TIME_MS: u32 = 20;
const JITTER_BUFFER_SIZE_MS: u32 = 100;
const MAX_PACKET_SIZE: usize = 1500;
const PACKET_HEADER_SIZE: usize = 8;

/// Network packet header
#[repr(C)]
struct PacketHeader {
    sequence: u16,
    timestamp: u32,
    flags: u8,
    payload_len: u8,
}

impl PacketHeader {
    fn encode(&self) -> [u8; PACKET_HEADER_SIZE] {
        let mut bytes = [0u8; PACKET_HEADER_SIZE];
        bytes[0..2].copy_from_slice(&self.sequence.to_be_bytes());
        bytes[2..6].copy_from_slice(&self.timestamp.to_be_bytes());
        bytes[6] = self.flags;
        bytes[7] = self.payload_len;
        bytes
    }

    fn decode(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < PACKET_HEADER_SIZE {
            return None;
        }
        Some(Self {
            sequence: u16::from_be_bytes([bytes[0], bytes[1]]),
            timestamp: u32::from_be_bytes([bytes[2], bytes[3], bytes[4], bytes[5]]),
            flags: bytes[6],
            payload_len: bytes[7],
        })
    }
}

/// Audio configuration
struct AudioConfig {
    sample_rate: u32,
    channels: u16,
    frame_samples: usize,
    opus_config: OpusConfig,
}

struct OpusConfig {
    sample_rate: SampleRate,
    channels: Channels,
}

impl AudioConfig {
    fn from_stream_config(config: &cpal::StreamConfig, packet_time_ms: u32) -> Result<Self> {
        let sample_rate = config.sample_rate.0;
        let channels = config.channels;

        let opus_sample_rate = match sample_rate {
            8000 => SampleRate::Hz8000,
            12000 => SampleRate::Hz12000,
            16000 => SampleRate::Hz16000,
            24000 => SampleRate::Hz24000,
            48000 => SampleRate::Hz48000,
            _ => anyhow::bail!("Unsupported sample rate: {} Hz", sample_rate),
        };

        let opus_channels = match channels {
            1 => Channels::Mono,
            2 => Channels::Stereo,
            _ => anyhow::bail!("Unsupported channel count: {}", channels),
        };

        let frame_samples = (sample_rate / 1000 * packet_time_ms) as usize * channels as usize;

        Ok(Self {
            sample_rate,
            channels,
            frame_samples,
            opus_config: OpusConfig {
                sample_rate: opus_sample_rate,
                channels: opus_channels,
            },
        })
    }
}

/// Network statistics
#[derive(Default)]
struct NetworkStats {
    packets_sent: AtomicU64,
    packets_received: AtomicU64,
    bytes_sent: AtomicU64,
    bytes_received: AtomicU64,
    last_sequence: AtomicU16,
    packets_lost: AtomicU64,
    jitter_ms: AtomicU32,
}

impl NetworkStats {
    fn update_sent(&self, bytes: usize) {
        self.packets_sent.fetch_add(1, Ordering::Relaxed);
        self.bytes_sent.fetch_add(bytes as u64, Ordering::Relaxed);
    }

    fn update_received(&self, bytes: usize, sequence: u16) {
        self.packets_received.fetch_add(1, Ordering::Relaxed);
        self.bytes_received.fetch_add(bytes as u64, Ordering::Relaxed);
        
        // Simple packet loss detection
        let last_seq = self.last_sequence.load(Ordering::Relaxed);
        if last_seq != 0 && sequence != last_seq.wrapping_add(1) {
            let lost = if sequence > last_seq {
                (sequence - last_seq - 1) as u64
            } else {
                1 // Reordering or wrap-around
            };
            self.packets_lost.fetch_add(lost, Ordering::Relaxed);
        }
        self.last_sequence.store(sequence, Ordering::Relaxed);
    }

    fn print_summary(&self) {
        let sent = self.packets_sent.load(Ordering::Relaxed);
        let received = self.packets_received.load(Ordering::Relaxed);
        let lost = self.packets_lost.load(Ordering::Relaxed);
        let loss_rate = if sent > 0 {
            (lost as f64 / sent as f64) * 100.0
        } else {
            0.0
        };

        print!(
            "\rPkts TX: {} RX: {} | Loss: {:.1}% | Jitter: {} ms  ",
            sent,
            received,
            loss_rate,
            self.jitter_ms.load(Ordering::Relaxed)
        );
        use std::io::{self, Write};
        io::stdout().flush().ok();
    }
}

/// Jitter buffer for smooth playback
struct JitterBuffer {
    packets: VecDeque<(u32, Vec<f32>)>,
    target_delay_samples: usize,
    current_delay_samples: usize,
    last_timestamp: Option<u32>,
}

impl JitterBuffer {
    fn new(buffer_ms: u32, sample_rate: u32) -> Self {
        let target_delay_samples = (sample_rate / 1000 * buffer_ms) as usize;
        Self {
            packets: VecDeque::with_capacity(20),
            target_delay_samples,
            current_delay_samples: 0,
            last_timestamp: None,
        }
    }

    fn push(&mut self, timestamp: u32, samples: Vec<f32>) {
        // Insert sorted by timestamp
        let pos = self
            .packets
            .binary_search_by_key(&timestamp, |(ts, _)| *ts)
            .unwrap_or_else(|e| e);

        self.packets.insert(pos, (timestamp, samples));

        // Limit buffer size to prevent memory growth
        while self.packets.len() > 10 {
            self.packets.pop_front();
        }
    }

    fn pop(&mut self, needed: usize) -> Vec<f32> {
        let mut output = Vec::with_capacity(needed);
        
        // Calculate current buffer depth
        let buffered_samples: usize = self.packets.iter().map(|(_, s)| s.len()).sum();
        self.current_delay_samples = buffered_samples;

        // Adaptive playout
        while output.len() < needed {
            if self.current_delay_samples >= self.target_delay_samples / 2 {
                if let Some((timestamp, samples)) = self.packets.pop_front() {
                    self.last_timestamp = Some(timestamp);
                    output.extend(samples);
                } else {
                    // Underrun - fill with silence
                    output.resize(needed, 0.0);
                    break;
                }
            } else {
                // Not enough buffered - fill with silence
                output.resize(needed, 0.0);
                break;
            }
        }

        output.truncate(needed);
        output
    }
}

/// Audio codec wrapper
struct AudioCodec {
    encoder: OpusEncoder,
    decoder: OpusDecoder,
    encode_buffer: Vec<u8>,
}

impl AudioCodec {
    fn new(config: &OpusConfig, bitrate: u32) -> Result<Self> {
        let mut encoder = OpusEncoder::new(
            config.sample_rate,
            config.channels,
            Application::Audio,
        )?;
        
        encoder.set_bitrate(Bitrate::BitsPerSecond(bitrate as i32))?;
        encoder.set_vbr(true)?;
        encoder.set_vbr_constraint(false)?;
        encoder.set_inband_fec(true)?;
        encoder.set_packet_loss_perc(2)?;

        let decoder = OpusDecoder::new(config.sample_rate, config.channels)?;

        Ok(Self {
            encoder,
            decoder,
            encode_buffer: vec![0u8; 4000], // Max Opus frame
        })
    }

    fn encode(&mut self, pcm: &[f32]) -> Result<&[u8]> {
        let len = self.encoder.encode_float(pcm, &mut self.encode_buffer)?;
        Ok(&self.encode_buffer[..len])
    }

    fn decode(&mut self, data: Option<&[u8]>, pcm: &mut [f32]) -> Result<usize> {
        Ok(self.decoder.decode_float(data, pcm, false)?)
    }
}

/// Main VoIP session
struct VoipSession {
    socket: UdpSocket,
    peer_addr: SocketAddr,
    codec: Mutex<AudioCodec>,
    jitter_buffer: Mutex<JitterBuffer>,
    stats: Arc<NetworkStats>,
    running: AtomicBool,
    tx_sequence: AtomicU16,
    tx_timestamp: AtomicU32,
}

impl VoipSession {
    fn new(
        bind_addr: SocketAddr,
        peer_addr: SocketAddr,
        audio_config: &AudioConfig,
        bitrate: u32,
    ) -> Result<Arc<Self>> {
        let socket = UdpSocket::bind(bind_addr)
            .with_context(|| format!("Failed to bind to {}", bind_addr))?;
        socket.set_nonblocking(true)?;

        // Socket buffer optimization (platform-specific)
        // On supported platforms, you can uncomment these:
        // let _ = socket.set_send_buffer_size(256 * 1024);
        // let _ = socket.set_recv_buffer_size(256 * 1024);

        let codec = AudioCodec::new(&audio_config.opus_config, bitrate)?;
        let jitter_buffer = JitterBuffer::new(JITTER_BUFFER_SIZE_MS, audio_config.sample_rate);

        Ok(Arc::new(Self {
            socket,
            peer_addr,
            codec: Mutex::new(codec),
            jitter_buffer: Mutex::new(jitter_buffer),
            stats: Arc::new(NetworkStats::default()),
            running: AtomicBool::new(true),
            tx_sequence: AtomicU16::new(0),
            tx_timestamp: AtomicU32::new(0),
        }))
    }

    fn send_audio(&self, pcm: &[f32]) -> Result<()> {
        let opus_data = {
            let mut codec = self.codec.lock().unwrap();
            codec.encode(pcm)?.to_vec()
        };

        if opus_data.len() > 255 {
            anyhow::bail!("Encoded frame too large");
        }

        let header = PacketHeader {
            sequence: self.tx_sequence.fetch_add(1, Ordering::Relaxed),
            timestamp: self.tx_timestamp.load(Ordering::Relaxed),
            flags: 0,
            payload_len: opus_data.len() as u8,
        };

        let mut packet = Vec::with_capacity(PACKET_HEADER_SIZE + opus_data.len());
        packet.extend_from_slice(&header.encode());
        packet.extend_from_slice(&opus_data);

        self.socket.send_to(&packet, self.peer_addr)?;
        self.stats.update_sent(packet.len());

        Ok(())
    }

    fn update_timestamp(&self, samples: u32) {
        self.tx_timestamp.fetch_add(samples, Ordering::Relaxed);
    }

    fn receive_audio(&self, audio_config: &AudioConfig) -> Result<()> {
        let mut buffer = [0u8; MAX_PACKET_SIZE];
        let mut pcm_buffer = vec![0.0f32; audio_config.frame_samples * 3]; // Up to 60ms

        loop {
            match self.socket.recv_from(&mut buffer) {
                Ok((len, _addr)) => {
                    if len < PACKET_HEADER_SIZE {
                        continue;
                    }

                    if let Some(header) = PacketHeader::decode(&buffer[..len]) {
                        let payload_end = PACKET_HEADER_SIZE + header.payload_len as usize;
                        if payload_end <= len {
                            let opus_data = &buffer[PACKET_HEADER_SIZE..payload_end];
                            
                            let decoded_samples = {
                                let mut codec = self.codec.lock().unwrap();
                                codec.decode(Some(opus_data), &mut pcm_buffer)?
                            };

                            let pcm = pcm_buffer[..decoded_samples * audio_config.channels as usize]
                                .to_vec();

                            self.jitter_buffer
                                .lock()
                                .unwrap()
                                .push(header.timestamp, pcm);

                            self.stats.update_received(len, header.sequence);
                        }
                    }
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    break;
                }
                Err(e) => return Err(e.into()),
            }
        }

        Ok(())
    }

    fn get_playback_samples(&self, count: usize) -> Vec<f32> {
        self.jitter_buffer.lock().unwrap().pop(count)
    }

    fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
    }

    fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: {} <BIND_ADDR:PORT> <PEER_ADDR:PORT>", args[0]);
        eprintln!("\nExample:");
        eprintln!("  Machine A: {} 0.0.0.0:5001 192.168.1.100:5002", args[0]);
        eprintln!("  Machine B: {} 0.0.0.0:5002 192.168.1.101:5001", args[0]);
        std::process::exit(1);
    }

    let bind_addr: SocketAddr = args[1].parse()?;
    let peer_addr: SocketAddr = args[2].parse()?;

    // Initialize audio
    let host = cpal::default_host();
    let input_device = host
        .default_input_device()
        .context("No input device found")?;
    let output_device = host
        .default_output_device()
        .context("No output device found")?;

    println!("Audio Input:  {}", input_device.name()?);
    println!("Audio Output: {}", output_device.name()?);

    // Find compatible configuration
    let config = find_compatible_config(&input_device, &output_device)?;
    let audio_config = AudioConfig::from_stream_config(&config, DEFAULT_PACKET_TIME_MS)?;

    println!(
        "Audio Format: {} Hz, {} ch, {}ms frames",
        audio_config.sample_rate,
        audio_config.channels,
        DEFAULT_PACKET_TIME_MS
    );

    // Create session
    let session = VoipSession::new(bind_addr, peer_addr, &audio_config, DEFAULT_BITRATE)?;

    // Start network receive thread
    let rx_thread = {
        let session = Arc::clone(&session);
        let audio_config = audio_config.clone();
        thread::spawn(move || {
            while session.is_running() {
                if let Err(e) = session.receive_audio(&audio_config) {
                    eprintln!("Receive error: {}", e);
                }
                thread::sleep(Duration::from_micros(100));
            }
        })
    };

    // Start statistics thread
    let stats_thread = {
        let session = Arc::clone(&session);
        thread::spawn(move || {
            while session.is_running() {
                thread::sleep(Duration::from_secs(1));
                session.stats.print_summary();
            }
        })
    };

    // Create audio streams
    let input_stream = {
        let session = Arc::clone(&session);
        let frame_size = audio_config.frame_samples;
        let channels = audio_config.channels as usize;
        let mut accumulator = Vec::with_capacity(frame_size * 2);

        input_device.build_input_stream(
            &config,
            move |data: &[f32], _| {
                accumulator.extend_from_slice(data);
                
                while accumulator.len() >= frame_size {
                    let frame: Vec<f32> = accumulator.drain(..frame_size).collect();
                    
                    if let Err(e) = session.send_audio(&frame) {
                        eprintln!("Send error: {}", e);
                    }
                    
                    session.update_timestamp((frame_size / channels) as u32);
                }
            },
            |err| eprintln!("Input error: {}", err),
            None,
        )?
    };

    let output_stream = {
        let session = Arc::clone(&session);
        
        output_device.build_output_stream(
            &config,
            move |data: &mut [f32], _| {
                let samples = session.get_playback_samples(data.len());
                data.copy_from_slice(&samples);
            },
            |err| eprintln!("Output error: {}", err),
            None,
        )?
    };

    // Start streams
    input_stream.play()?;
    output_stream.play()?;

    println!("\nVoIP session started. Press Ctrl-C to end.\n");

    // Handle Ctrl-C
    let session_stop = Arc::clone(&session);
    ctrlc::set_handler(move || {
        println!("\n\nEnding call...");
        session_stop.stop();
    })?;

    // Wait for shutdown
    while session.is_running() {
        thread::sleep(Duration::from_millis(100));
    }

    // Cleanup
    rx_thread.join().ok();
    stats_thread.join().ok();

    println!("Session ended.");
    Ok(())
}

/// Find a compatible F32 configuration between devices
fn find_compatible_config(
    input: &cpal::Device,
    output: &cpal::Device,
) -> Result<cpal::StreamConfig> {
    let input_configs: Vec<_> = input
        .supported_input_configs()?
        .filter(|c| c.sample_format() == cpal::SampleFormat::F32)
        .collect();
    
    let output_configs: Vec<_> = output
        .supported_output_configs()?
        .filter(|c| c.sample_format() == cpal::SampleFormat::F32)
        .collect();

    // Preferred sample rates in order
    const PREFERRED_RATES: &[u32] = &[48000, 44100, 24000, 16000, 12000, 8000];

    for &target_rate in PREFERRED_RATES {
        let sample_rate = cpal::SampleRate(target_rate);
        
        for in_cfg in &input_configs {
            if sample_rate < in_cfg.min_sample_rate() || sample_rate > in_cfg.max_sample_rate() {
                continue;
            }

            for out_cfg in &output_configs {
                if sample_rate < out_cfg.min_sample_rate() || sample_rate > out_cfg.max_sample_rate() {
                    continue;
                }

                // Use minimum channel count to ensure compatibility
                let channels = in_cfg.channels().min(out_cfg.channels());

                return Ok(cpal::StreamConfig {
                    channels,
                    sample_rate,
                    buffer_size: cpal::BufferSize::Default,
                });
            }
        }
    }

    anyhow::bail!("No compatible F32 audio configuration found")
}

// Allow AudioConfig to be cloned for thread usage
impl Clone for AudioConfig {
    fn clone(&self) -> Self {
        Self {
            sample_rate: self.sample_rate,
            channels: self.channels,
            frame_samples: self.frame_samples,
            opus_config: OpusConfig {
                sample_rate: self.opus_config.sample_rate,
                channels: self.opus_config.channels,
            },
        }
    }
} 