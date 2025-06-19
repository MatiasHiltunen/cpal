use std::io;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, StreamConfig};

use crossterm::{event, execute, terminal};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Layout};
use ratatui::style::{Color, Style};
use ratatui::text::{Span, Spans};
use ratatui::widgets::{Block, Paragraph};
use ratatui::Terminal;
use rustfft::{num_complex::Complex, FftPlanner};

const FFT_SIZE: usize = 1024;
const BINS: usize = 64;   // columns
const HISTORY: usize = 60; // rows
const REFRESH_MS: u64 = 80;

fn main() -> anyhow::Result<()> {
    // Setup terminal
    terminal::enable_raw_mode()?;
    execute!(io::stdout(), terminal::EnterAlternateScreen, event::EnableMouseCapture)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(io::stdout()))?;

    // Audio setup
    let host = cpal::default_host();
    let device = host.default_input_device().expect("No input device");
    let supported = device.default_input_config()?;
    let mut cfg: StreamConfig = supported.clone().into();

    let (tx, rx) = mpsc::channel::<f32>();
    let sample_format = supported.sample_format();

    let stream = build_stream(&device, &cfg, sample_format, tx.clone())?;
    stream.play()?;

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(FFT_SIZE);
    let mut buf = vec![0f32; FFT_SIZE];
    let mut pos = 0usize;
    let mut history: Vec<Vec<f32>> = Vec::with_capacity(HISTORY);

    let bin_hz = supported.sample_rate().0 as f32 / FFT_SIZE as f32;
    let step = (FFT_SIZE / 2) / BINS;

    loop {
        // Collect samples
        while let Ok(s) = rx.try_recv() {
            buf[pos] = s;
            pos += 1;
            if pos >= FFT_SIZE {
                // perform fft
                let mut complex: Vec<Complex<f32>> = buf.iter().map(|&v| Complex{re:v, im:0.0}).collect();
                fft.process(&mut complex);
                let magnitudes: Vec<f32> = complex.iter().take(FFT_SIZE/2).map(|c| c.norm()).collect();
                // compress bins
                let mut row = vec![0f32; BINS];
                for i in 0..BINS {
                    let slice = &magnitudes[i*step..(i+1)*step];
                    let avg = slice.iter().sum::<f32>() / slice.len() as f32;
                    row[i] = avg;
                }
                // normalize row 0..1 logarithmic
                let max_mag = row.iter().cloned().fold(0./0., f32::max).max(1e-6);
                for v in &mut row {
                    *v = (v.log10().max(-5.0)+5.0)/5.0; // 0..1
                }
                if history.len() == HISTORY {
                    history.remove(0);
                }
                history.push(row);
                pos = 0;
            }
        }

        // Draw
        terminal.draw(|f| {
            let size = f.size();
            let chunks = Layout::default().constraints([Constraint::Percentage(100)].as_ref()).split(size);

            // Build lines
            let mut lines: Vec<Spans> = Vec::new();
            for row in history.iter().rev() { // newest at bottom
                let mut spans: Vec<Span> = Vec::with_capacity(BINS);
                for &v in row {
                    spans.push(Span::styled("█", Style::default().fg(color_for(v))));
                }
                lines.push(Spans::from(spans));
            }
            // pad empty
            while lines.len()<HISTORY {
                lines.push(Spans::from(""));
            }
            lines.reverse();

            let para = Paragraph::new(lines).block(Block::default().title("Spectrogram (q to quit)"));
            f.render_widget(para, chunks[0]);
        })?;

        if event::poll(Duration::from_millis(REFRESH_MS))? {
            if let event::Event::Key(k) = event::read()? {
                if matches!(k.code, event::KeyCode::Char('q')|event::KeyCode::Esc){
                    break;
                }
            }
        }
    }

    execute!(io::stdout(), terminal::LeaveAlternateScreen, event::DisableMouseCapture)?;
    terminal::disable_raw_mode()?;
    Ok(())
}

fn build_stream(device: &cpal::Device, cfg:&StreamConfig, fmt:SampleFormat, tx:mpsc::Sender<f32>) -> anyhow::Result<cpal::Stream> {
    let err_fn = |e| eprintln!("stream error {e}");
    Ok(match fmt {
        SampleFormat::F32 => device.build_input_stream(cfg, move |d:&[f32],_|{for &v in d{let _=tx.send(v);}}, err_fn, None)?,
        SampleFormat::I16 => device.build_input_stream(cfg, move |d:&[i16],_|{for &v in d{let _=tx.send(v as f32 / i16::MAX as f32);}}, err_fn, None)?,
        SampleFormat::U16 => device.build_input_stream(cfg, move |d:&[u16],_|{for &v in d{let sample=v as f32 - 32768.0;let _=tx.send(sample/32768.0);}},err_fn,None)?,
        _ => anyhow::bail!("Unsupported"),
    })
}

fn color_for(v: f32) -> Color {
    // v in 0..1 => map to purple (low) to white (high)
    let v = v.clamp(0.0,1.0);
    let r = (191.0 + v*64.0) as u8; // 191..255
    let g = (0.0 + v*64.0) as u8;   // 0..64
    let b = (191.0 + v*64.0) as u8; // 191..255 (purple tint)
    Color::Rgb(r,g,b)
} 