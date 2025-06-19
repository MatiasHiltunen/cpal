use std::io;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, StreamConfig};

use crossterm::{event, execute, terminal};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Layout};
use ratatui::style::Style;
use ratatui::widgets::{BarChart, Block};
use ratatui::Terminal;
use rustfft::{FftPlanner, num_complex::Complex};

const FFT_SIZE: usize = 1024;
const REFRESH_MS: u64 = 100;

fn main() -> anyhow::Result<()> {
    // Terminal setup
    terminal::enable_raw_mode()?;
    execute!(io::stdout(), terminal::EnterAlternateScreen, event::EnableMouseCapture)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(io::stdout()))?;

    // CPAL default input
    let host = cpal::default_host();
    let device = host.default_input_device().expect("No input device");
    println!("Using input device: {}", device.name()?.to_string());
    let supported = device.default_input_config()?;
    let mut cfg: StreamConfig = supported.clone().into();

    let (tx, rx) = mpsc::channel::<f32>();

    // Build stream
    let sample_format = supported.sample_format();
    let err_fn = |err| eprintln!("Stream error: {err}");
    let stream = build_stream(&device, &cfg, sample_format, tx, err_fn)?;
    stream.play()?;

    // FFT planner
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(FFT_SIZE);
    let mut input_buf = vec![0f32; FFT_SIZE];
    let mut pos = 0usize;

    let bin_hz = supported.sample_rate().0 as f32 / FFT_SIZE as f32;
    let mut magnitudes: Vec<f32> = vec![0.0; FFT_SIZE / 2];

    let start = Instant::now();
    loop {
        // Read samples
        while let Ok(sample) = rx.try_recv() {
            input_buf[pos] = sample;
            pos += 1;
            if pos >= FFT_SIZE {
                // Perform FFT
                let mut complex: Vec<Complex<f32>> = input_buf.iter().map(|&v| Complex{ re: v, im: 0.0 }).collect();
                fft.process(&mut complex);
                for (i, c) in complex.iter().take(FFT_SIZE/2).enumerate() {
                    magnitudes[i] = (c.norm() / FFT_SIZE as f32).log10().max(-5.0)+5.0; // scale 0..5
                }
                pos = 0;
            }
        }

        // Draw
        terminal.draw(|f| {
            let size = f.size();
            let chunks = Layout::default().constraints([Constraint::Percentage(100)].as_ref()).split(size);

            // Prepare barchart data (reduce bins)
            const BARS: usize = 32;
            let step = (magnitudes.len() / BARS).max(1);

            // Build vector with owned Strings first to keep them alive.
            let mut items: Vec<(String, u64)> = Vec::with_capacity(BARS);
            for i in 0..BARS {
                let slice = &magnitudes[i*step..((i+1)*step).min(magnitudes.len())];
                let avg = slice.iter().copied().sum::<f32>() / slice.len() as f32;
                let label_hz = (i*step) as f32 * bin_hz;
                items.push((format!("{:.0}", label_hz), (avg*20.0) as u64));
            }

            // Convert to (&str,u64)
            let data: Vec<(&str, u64)> = items.iter().map(|(s,v)| (s.as_str(), *v)).collect();

            let barchart = BarChart::default()
                .block(Block::default().title("Frequency Spectrum (q to quit)"))
                .bar_width(3)
                .data(&data)
                .style(Style::default());
            f.render_widget(barchart, chunks[0]);
        })?;

        if event::poll(Duration::from_millis(REFRESH_MS))? {
            if let event::Event::Key(k) = event::read()? {
                if matches!(k.code, event::KeyCode::Char('q') | event::KeyCode::Esc) {
                    break;
                }
            }
        }
    }

    execute!(io::stdout(), terminal::LeaveAlternateScreen, event::DisableMouseCapture)?;
    terminal::disable_raw_mode()?;
    Ok(())
}

fn build_stream(
    device: &cpal::Device,
    config: &StreamConfig,
    sample_format: SampleFormat,
    sender: mpsc::Sender<f32>,
    err: impl FnMut(cpal::StreamError) + Send + 'static,
) -> Result<cpal::Stream, anyhow::Error> {
    match sample_format {
        SampleFormat::F32 => Ok(device.build_input_stream(config, move |d: &[f32], _| { for &v in d { let _ = sender.send(v); } }, err, None)?),
        SampleFormat::I16 => Ok(device.build_input_stream(config, move |d: &[i16], _| { for &v in d { let _ = sender.send(v as f32 / i16::MAX as f32); } }, err, None)?),
        SampleFormat::U16 => Ok(device.build_input_stream(config, move |d: &[u16], _| { for &v in d { let sample = v as f32 - 32768.0; let _ = sender.send(sample / 32768.0); } }, err, None)?),
        _ => anyhow::bail!("Unsupported format"),
    }
} 