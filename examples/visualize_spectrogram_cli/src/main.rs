// With WASAPI on Windows, RAW (unprocessed) mode can be requested by setting the environment variable
// `CPAL_WASAPI_REQUEST_FORCE_RAW=1`. When enabled, we ask the driver to bypass pre-processing such
// as AGC or noise suppression. So far no other hosts have been tested with this.
// Spectrogram demonstrates quite nicely the difference between true raw and the AGC or noise suppression.
//
// Windows cmd:
//  set CPAL_WASAPI_REQUEST_FORCE_RAW=1
//  cargo run --release --manifest-path examples/visualize_spectrogram_cli/Cargo.toml
//
// PowerShell:
//  $env:CPAL_WASAPI_REQUEST_FORCE_RAW=1; cargo run --release --manifest-path examples/visualize_spectrogram_cli/Cargo.toml


use std::io::{self, Write};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use anyhow::Context;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, StreamConfig};

use crossterm::{cursor, event, execute, queue, style, terminal};
use rustfft::{num_complex::Complex, FftPlanner};

const FFT_SIZE: usize = 1024;
const HISTORY: usize = 180; // terminal rows to keep
const REFRESH_MS: u64 = 10; // redraw & input poll interval
const ROW_INTERVAL_MS: u64 = 48; // push a new spectrogram row
const HIGH_FREQ_BOOST: f32 = 1.0; // up to ×(1+HIGH_FREQ_BOOST) gain at highest bin

fn main() -> anyhow::Result<()> {
    // Put terminal in raw + alternate screen mode
    terminal::enable_raw_mode()?;
    execute!(
        io::stdout(),
        terminal::EnterAlternateScreen,
        cursor::Hide,
        event::EnableMouseCapture
    )?;
    let mut stdout = io::stdout();

    // Audio setup
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .context("No input device available")?;
    let supported = device.default_input_config()?;
    let cfg: StreamConfig = supported.clone().into();

    let (tx, rx) = mpsc::channel::<f32>();
    let sample_format = supported.sample_format();

    let stream = build_stream(&device, &cfg, sample_format, tx.clone())?;
    stream.play()?;

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(FFT_SIZE);
    let mut buf = vec![0f32; FFT_SIZE];
    let mut pos = 0usize;
    let mut history: Vec<Vec<f32>> = Vec::with_capacity(HISTORY);

    // Determine initial bin count from terminal width
    let mut bins = {
        let (w, _) = terminal::size()?;
        usize::max(w as usize, 1)
    };

    // Accumulate the maximum magnitude observed for each bin over a ROW_INTERVAL_MS window
    let mut interval_row = vec![0f32; bins];
    let mut last_row_time = Instant::now();

    let mut needs_full_redraw = true;

    loop {
        // Detect terminal width changes and adjust bins accordingly
        let new_bins = {
            let (w, _) = terminal::size()?;
            usize::max(w as usize, 1)
        };
        if new_bins != bins {
            bins = new_bins;
            interval_row = vec![0f32; bins];
            history.clear();
            needs_full_redraw = true;
        }

        // Collect samples
        while let Ok(s) = rx.try_recv() {
            buf[pos] = s;
            pos += 1;
            if pos >= FFT_SIZE {
                // perform fft
                let mut complex: Vec<Complex<f32>> =
                    buf.iter().map(|&v| Complex { re: v, im: 0.0 }).collect();
                fft.process(&mut complex);
                let magnitudes: Vec<f32> =
                    complex.iter().take(FFT_SIZE / 2).map(|c| c.norm()).collect();

                // compress bins
                let mut row = vec![0f32; bins];
                let step = usize::max(magnitudes.len() / bins, 1);
                for i in 0..bins {
                    let start = i * step;
                    if start >= magnitudes.len() {
                        break;
                    }
                    let end = ((i + 1) * step).min(magnitudes.len());
                    let slice = &magnitudes[start..end];
                    let avg = slice.iter().sum::<f32>() / slice.len() as f32;
                    // Apply frequency-dependent gain: low bins ≈ 1×, highest bin ≈ 1+HIGH_FREQ_BOOST×
                    let weight = 1.0 + HIGH_FREQ_BOOST * (i as f32) / (bins.max(1) as f32 - 1.0);
                    row[i] = avg * weight;
                }
                // normalize row 0..1 logarithmic
                for v in &mut row {
                    *v = ((*v).log10().max(-5.0) + 5.0) / 5.0; // 0..1
                }

                // update per-second maxima
                for i in 0..bins {
                    interval_row[i] = interval_row[i].max(row[i]);
                }

                // every ROW_INTERVAL_MS, commit a row built from the maxima (or immediately if empty)
                if last_row_time.elapsed() >= Duration::from_millis(ROW_INTERVAL_MS) || history.is_empty() {
                    if history.len() == HISTORY {
                        history.remove(0);
                    }
                    history.push(interval_row.clone());
                    interval_row.fill(0.0);
                    last_row_time = Instant::now();
                    // Incremental draw only if a full redraw isn't scheduled
                    if !needs_full_redraw {
                        draw_incremental(&mut stdout, &history, bins)?;
                    }
                }
                pos = 0;
            }
        }

        // Draw only if something changed (width or new row)
        if needs_full_redraw {
            draw_full(&mut stdout, &history, bins)?;
            needs_full_redraw = false;
        }

        // Handle input
        if event::poll(Duration::from_millis(REFRESH_MS))? {
            if let event::Event::Key(k) = event::read()? {
                if matches!(k.code, event::KeyCode::Char('q') | event::KeyCode::Esc) {
                    break;
                }
            }
        }
    }

    // Restore terminal
    execute!(
        stdout,
        terminal::LeaveAlternateScreen,
        event::DisableMouseCapture,
        style::ResetColor,
        cursor::Show
    )?;
    terminal::disable_raw_mode()?;

    Ok(())
}

fn draw_full(stdout: &mut io::Stdout, history: &Vec<Vec<f32>>, _bins: usize) -> anyhow::Result<()> {
    // Move to origin and clear below to avoid flicker from full screen clear
    queue!(
        stdout,
        cursor::MoveTo(0, 0),
        terminal::Clear(terminal::ClearType::FromCursorDown)
    )?;

    // Title
    queue!(stdout, style::Print("Spectrogram (q to quit)\n"))?;

    for row in history.iter().rev() {
        for &v in row {
            let color = color_for(v);
            queue!(
                stdout,
                style::SetForegroundColor(color),
                style::Print("█")
            )?;
        }
        queue!(stdout, style::ResetColor, style::Print("\n"))?;
    }
    stdout.flush()?;
    Ok(())
}

fn draw_incremental(stdout: &mut io::Stdout, history: &Vec<Vec<f32>>, bins: usize) -> anyhow::Result<()> {
    if history.is_empty() {
        return Ok(());
    }
    // Scroll spectrogram area up by one (starting below title)
    queue!(stdout, cursor::MoveTo(0, 1), terminal::ScrollUp(1u16))?;

    // Re-draw title because scrolling overwrote it
    queue!(
        stdout,
        cursor::MoveTo(0, 0),
        terminal::Clear(terminal::ClearType::UntilNewLine),
        style::ResetColor,
        style::Print("Spectrogram (q to quit)")
    )?;

    // determine bottom row y coordinate
    let bottom_row_y = {
        let (_w, h) = terminal::size()?;
        h.saturating_sub(1)
    };

    let last_row = history.last().unwrap();

    queue!(stdout, cursor::MoveTo(0, bottom_row_y))?;
    for &v in last_row.iter().take(bins) {
        let color = color_for(v);
        queue!(stdout, style::SetForegroundColor(color), style::Print("█"))?;
    }
    queue!(stdout, style::ResetColor)?;
    stdout.flush()?;
    Ok(())
}

fn build_stream(
    device: &cpal::Device,
    cfg: &StreamConfig,
    fmt: SampleFormat,
    tx: mpsc::Sender<f32>,
) -> anyhow::Result<cpal::Stream> {
    let err_fn = |e| eprintln!("stream error {e}");
    Ok(match fmt {
        SampleFormat::F32 => device.build_input_stream(cfg, move |d: &[f32], _| {
            for &v in d {
                let _ = tx.send(v);
            }
        }, err_fn, None)?,
        SampleFormat::I16 => device.build_input_stream(cfg, move |d: &[i16], _| {
            for &v in d {
                let _ = tx.send(v as f32 / i16::MAX as f32);
            }
        }, err_fn, None)?,
        SampleFormat::U16 => device.build_input_stream(cfg, move |d: &[u16], _| {
            for &v in d {
                let sample = v as f32 - 32768.0;
                let _ = tx.send(sample / 32768.0);
            }
        }, err_fn, None)?,
        _ => anyhow::bail!("Unsupported sample format"),
    })
}

fn color_for(v: f32) -> style::Color {
    // Map 0 → black, 0.5 → purple, 1 → white using a two-segment gradient.
    let v = v.clamp(0.0, 1.0);
    if v < 0.5 {
        // Black → Purple
        let t = v * 2.0; // 0..1
        let r = (255.0 * 0.5 * t) as u8; // 0..127
        let g = 0u8;
        let b = (255.0 * 0.5 * t) as u8; // 0..127
        style::Color::Rgb { r, g, b }
    } else {
        // Purple → White
        let t = (v - 0.5) * 2.0; // 0..1
        let r = (127.0 + 128.0 * t) as u8; // 127..255
        let g = (0.0 + 255.0 * t) as u8; // 0..255
        let b = (127.0 + 128.0 * t) as u8; // 127..255
        style::Color::Rgb { r, g, b }
    }
} 