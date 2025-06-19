use std::collections::VecDeque;
use std::io::{self, Stdout};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Sample, SampleFormat, StreamConfig};

use crossterm::{event, execute, terminal};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Layout};
use ratatui::style::Style;
use ratatui::symbols;
use ratatui::widgets::{Axis, Chart, Dataset};
use ratatui::Terminal;
use ratatui::text::Span;

const HISTORY_LEN: usize = 2048; // samples to keep for drawing

fn main() -> anyhow::Result<()> {
    // Initialise terminal UI.
    setup_terminal()?;
    let mut terminal = create_terminal()?;

    // CPAL host & device.
    let host = cpal::default_host();
    let device = match host.default_input_device() {
        Some(d) => d,
        None => {
            restore_terminal();
            eprintln!("No default input device available");
            return Ok(());
        }
    };

    // Print device information
    println!("Using input device: {}", device.name().unwrap_or_else(|_| "<Unknown>".into()));

    #[cfg(target_os = "macos")]
    if let cpal::platform::DeviceInner::CoreAudio(inner) = device.as_inner() {
        if let Ok(meta) = inner.metadata() {
            println!("  UID       : {:?}", meta.uid);
            println!("  Transport : {:?}", meta.transport_type);
        }
    }

    let config = device.default_input_config()?;

    // Channel to send amplitude samples from audio thread to UI thread.
    let (tx, rx) = mpsc::channel::<f32>();

    // Build and run the input stream.
    let err_fn = |err| eprintln!("Stream error: {}", err);

    let stream = match config.sample_format() {
        SampleFormat::F32 => {
            let cfg: StreamConfig = config.clone().into();
            device.build_input_stream(&cfg, move |data: &[f32], _| {
                for &sample in data {
                    let _ = tx.send(sample.abs());
                }
            }, err_fn, None)?
        }
        SampleFormat::I16 => {
            let cfg: StreamConfig = config.clone().into();
            device.build_input_stream(&cfg, move |data: &[i16], _| {
                for &s in data {
                    let val = (s as f32).abs() / i16::MAX as f32;
                    let _ = tx.send(val);
                }
            }, err_fn, None)?
        }
        SampleFormat::U16 => {
            let cfg: StreamConfig = config.clone().into();
            device.build_input_stream(&cfg, move |data: &[u16], _| {
                for &s in data {
                    let signed = s as i32 - 32768;
                    let val = (signed as f32).abs() / 32768.0;
                    let _ = tx.send(val);
                }
            }, err_fn, None)?
        }
        _ => unreachable!(),
    };

    stream.play()?;

    // History buffer for plotting.
    let mut history: VecDeque<(f64, f64)> = VecDeque::with_capacity(HISTORY_LEN);
    let start = Instant::now();

    loop {
        // Drain channel quickly.
        while let Ok(val) = rx.try_recv() {
            let t = start.elapsed().as_secs_f64();
            if history.len() == HISTORY_LEN {
                history.pop_front();
            }
            history.push_back((t, val as f64));
        }

        // Draw UI.
        terminal.draw(|f| {
            let size = f.size();
            let chunks = Layout::default()
                .constraints([Constraint::Percentage(100)].as_ref())
                .split(size);

            let data_vec: Vec<(f64, f64)> = history.iter().cloned().collect();

            // Determine x-axis bounds based on history.
            let (x_min, x_max) = if let (Some(first), Some(last)) = (history.front(), history.back()) {
                (first.0, last.0)
            } else {
                let now = start.elapsed().as_secs_f64();
                (now - 1.0, now)
            };

            let datasets = vec![Dataset::default()
                .name("amplitude")
                .marker(symbols::Marker::Braille)
                .style(Style::default())
                .data(&data_vec)];

            let chart = Chart::new(datasets)
                .x_axis(
                    Axis::default()
                        .bounds([x_min, x_max])
                        .labels(vec![Span::raw("now-1s"), Span::raw("now")])
                )
                .y_axis(
                    Axis::default()
                        .bounds([0.0, 1.0])
                        .labels(vec![Span::raw("0"), Span::raw("1")])
                )
                .block(ratatui::widgets::Block::default().title("Input Waveform (press q / Esc to quit)"));

            f.render_widget(chart, chunks[0]);
        })?;

        // Handle terminal events / exit.
        if event::poll(Duration::from_millis(50))? {
            if let event::Event::Key(k) = event::read()? {
                if k.code == event::KeyCode::Char('q') || k.code == event::KeyCode::Esc {
                    break;
                }
            }
        }

        // Debug: if no history yet, show message in main screen.
        if history.is_empty() {
            terminal.draw(|f| {
                let size = f.size();
                let block = ratatui::widgets::Block::default().title("Waiting for input stream...").title_alignment(ratatui::layout::Alignment::Center);
                f.render_widget(block, size);
            })?;
        }
    }

    // Restore terminal state before exit.
    restore_terminal();
    Ok(())
}

fn setup_terminal() -> anyhow::Result<()> {
    terminal::enable_raw_mode()?;
    execute!(io::stdout(), terminal::EnterAlternateScreen, event::EnableMouseCapture)?;
    Ok(())
}

fn restore_terminal() {
    let _ = execute!(io::stdout(), terminal::LeaveAlternateScreen, event::DisableMouseCapture);
    let _ = terminal::disable_raw_mode();
}

fn create_terminal() -> anyhow::Result<Terminal<CrosstermBackend<Stdout>>> {
    let stdout = io::stdout();
    let backend = CrosstermBackend::new(stdout);
    let terminal = Terminal::new(backend)?;
    Ok(terminal)
} 