//! Real-time audio spectrogram visualization using terminal graphics.
//!
//! This example demonstrates:
//! - Real-time audio capture and processing
//! - FFT-based frequency analysis
//! - Terminal-based visualization using Unicode characters
//! - Dynamic terminal resizing support
//! - Configurable audio processing parameters
//!
//! ## Usage
//! ```
//! cargo run --example visualize_spectrogram
//! ```
//!
//! ## Environment Variables
//! - `CPAL_WASAPI_REQUEST_FORCE_RAW=1` - On Windows, request raw (unprocessed) audio input

use std::io;
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, SampleFormat, Stream, StreamConfig};

use crossterm::{event, execute, terminal};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Layout};
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Paragraph};
use ratatui::Terminal;

use rustfft::{num_complex::Complex, Fft, FftPlanner};

// Configuration constants
mod config {
    use std::time::Duration;

    /// FFT window size - determines frequency resolution
    pub const FFT_SIZE: usize = 1024;
    
    /// Number of historical rows to display
    pub const HISTORY_ROWS: usize = 180;
    
    /// UI refresh rate
    pub const REFRESH_INTERVAL: Duration = Duration::from_millis(10);
    
    /// How often to push a new spectrogram row
    pub const ROW_UPDATE_INTERVAL: Duration = Duration::from_millis(48);
    
    /// High frequency emphasis factor (0.0 = no emphasis, 1.0 = 2x at highest frequency)
    pub const HIGH_FREQ_BOOST: f32 = 1.0;
    
    /// Minimum magnitude in dB for logarithmic scaling
    pub const MIN_DB: f32 = -50.0;
    
    /// Maximum magnitude in dB for logarithmic scaling
    pub const MAX_DB: f32 = 0.0;
}

/// Errors that can occur during spectrogram operation
#[derive(Debug)]
enum SpectrogramError {
    AudioDevice(String),
    Terminal(String),
    Channel(String),
}

impl std::fmt::Display for SpectrogramError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AudioDevice(msg) => write!(f, "Audio device error: {}", msg),
            Self::Terminal(msg) => write!(f, "Terminal error: {}", msg),
            Self::Channel(msg) => write!(f, "Channel error: {}", msg),
        }
    }
}

impl std::error::Error for SpectrogramError {}

/// Converts various error types to SpectrogramError
impl From<io::Error> for SpectrogramError {
    fn from(err: io::Error) -> Self {
        SpectrogramError::Terminal(err.to_string())
    }
}

impl From<cpal::DefaultStreamConfigError> for SpectrogramError {
    fn from(err: cpal::DefaultStreamConfigError) -> Self {
        SpectrogramError::AudioDevice(err.to_string())
    }
}

impl From<cpal::BuildStreamError> for SpectrogramError {
    fn from(err: cpal::BuildStreamError) -> Self {
        SpectrogramError::AudioDevice(err.to_string())
    }
}

/// Manages audio input and sample collection
struct AudioCapture {
    stream: Stream,
    receiver: Receiver<f32>,
}

impl AudioCapture {
    /// Creates a new audio capture instance using the default input device
    fn new() -> Result<Self, SpectrogramError> {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .ok_or_else(|| SpectrogramError::AudioDevice("No input device found".to_string()))?;
        
        let supported_config = device.default_input_config()?;
        let config: StreamConfig = supported_config.config();
        let sample_format = supported_config.sample_format();
        
        let (sender, receiver) = mpsc::channel::<f32>();
        let stream = Self::build_stream(&device, &config, sample_format, sender)?;
        
        Ok(Self { stream, receiver })
    }
    
    /// Starts audio capture
    fn start(&self) -> Result<(), SpectrogramError> {
        self.stream
            .play()
            .map_err(|e| SpectrogramError::AudioDevice(e.to_string()))
    }
    
    /// Attempts to receive audio samples without blocking
    fn try_recv(&self) -> Result<f32, TryRecvError> {
        self.receiver.try_recv()
    }
    
    /// Builds an audio input stream for the specified format
    fn build_stream(
        device: &Device,
        config: &StreamConfig,
        format: SampleFormat,
        sender: Sender<f32>,
    ) -> Result<Stream, SpectrogramError> {
        let error_callback = |err| eprintln!("Audio stream error: {}", err);
        
        let stream = match format {
            SampleFormat::F32 => Self::build_f32_stream(device, config, sender, error_callback),
            SampleFormat::I16 => Self::build_i16_stream(device, config, sender, error_callback),
            SampleFormat::U16 => Self::build_u16_stream(device, config, sender, error_callback),
            _ => return Err(SpectrogramError::AudioDevice(
                format!("Unsupported sample format: {:?}", format)
            )),
        }?;
        
        Ok(stream)
    }
    
    fn build_f32_stream(
        device: &Device,
        config: &StreamConfig,
        sender: Sender<f32>,
        error_callback: impl FnMut(cpal::StreamError) + Send + 'static,
    ) -> Result<Stream, cpal::BuildStreamError> {
        device.build_input_stream(
            config,
            move |data: &[f32], _: &_| {
                for &sample in data {
                    let _ = sender.send(sample);
                }
            },
            error_callback,
            None,
        )
    }
    
    fn build_i16_stream(
        device: &Device,
        config: &StreamConfig,
        sender: Sender<f32>,
        error_callback: impl FnMut(cpal::StreamError) + Send + 'static,
    ) -> Result<Stream, cpal::BuildStreamError> {
        device.build_input_stream(
            config,
            move |data: &[i16], _: &_| {
                for &sample in data {
                    let normalized = sample as f32 / i16::MAX as f32;
                    let _ = sender.send(normalized);
                }
            },
            error_callback,
            None,
        )
    }
    
    fn build_u16_stream(
        device: &Device,
        config: &StreamConfig,
        sender: Sender<f32>,
        error_callback: impl FnMut(cpal::StreamError) + Send + 'static,
    ) -> Result<Stream, cpal::BuildStreamError> {
        device.build_input_stream(
            config,
            move |data: &[u16], _: &_| {
                for &sample in data {
                    let centered = sample as f32 - 32768.0;
                    let normalized = centered / 32768.0;
                    let _ = sender.send(normalized);
                }
            },
            error_callback,
            None,
        )
    }
}

/// Performs FFT analysis on audio samples
struct FftAnalyzer {
    planner: FftPlanner<f32>,
    fft: Arc<dyn Fft<f32>>,
    buffer: Vec<f32>,
    position: usize,
}

impl FftAnalyzer {
    /// Creates a new FFT analyzer with the specified window size
    fn new(fft_size: usize) -> Self {
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(fft_size);
        
        Self {
            planner,
            fft,
            buffer: vec![0.0; fft_size],
            position: 0,
        }
    }
    
    /// Adds a sample to the buffer and returns frequency magnitudes if buffer is full
    fn add_sample(&mut self, sample: f32) -> Option<Vec<f32>> {
        self.buffer[self.position] = sample;
        self.position += 1;
        
        if self.position >= self.buffer.len() {
            self.position = 0;
            Some(self.compute_magnitudes())
        } else {
            None
        }
    }
    
    /// Computes frequency magnitudes from the current buffer
    fn compute_magnitudes(&self) -> Vec<f32> {
        let mut complex_buffer: Vec<Complex<f32>> = self.buffer
            .iter()
            .map(|&sample| Complex { re: sample, im: 0.0 })
            .collect();
        
        self.fft.process(&mut complex_buffer);
        
        // Return magnitudes for positive frequencies only
        complex_buffer
            .iter()
            .take(self.buffer.len() / 2)
            .map(|c| c.norm())
            .collect()
    }
}

/// Manages the spectrogram display history and binning
struct SpectrogramDisplay {
    history: Vec<Vec<f32>>,
    max_rows: usize,
    current_bins: usize,
    interval_maximums: Vec<f32>,
    last_row_time: Instant,
}

impl SpectrogramDisplay {
    /// Creates a new spectrogram display with the specified parameters
    fn new(max_rows: usize, initial_width: usize) -> Self {
        Self {
            history: Vec::with_capacity(max_rows),
            max_rows,
            current_bins: initial_width.max(1),
            interval_maximums: vec![0.0; initial_width.max(1)],
            last_row_time: Instant::now(),
        }
    }
    
    /// Updates the display with new frequency magnitudes
    fn update(&mut self, magnitudes: &[f32], terminal_width: usize) -> bool {
        // Handle terminal resize
        if terminal_width != self.current_bins && terminal_width > 0 {
            self.resize(terminal_width);
        }
        
        // Bin the frequency data to match terminal width
        let binned = self.bin_frequencies(magnitudes);
        
        // Apply logarithmic scaling
        let scaled = self.apply_log_scaling(&binned);
        
        // Update interval maximums
        for (i, &value) in scaled.iter().enumerate() {
            self.interval_maximums[i] = self.interval_maximums[i].max(value);
        }
        
        // Check if it's time to add a new row
        let should_update = self.last_row_time.elapsed() >= config::ROW_UPDATE_INTERVAL
            || self.history.is_empty();
        
        if should_update {
            self.add_row(self.interval_maximums.clone());
            self.interval_maximums.fill(0.0);
            self.last_row_time = Instant::now();
            true
        } else {
            false
        }
    }
    
    /// Resizes the display to match new terminal width
    fn resize(&mut self, new_width: usize) {
        self.current_bins = new_width;
        self.interval_maximums = vec![0.0; new_width];
        self.history.clear();
    }
    
    /// Bins frequency magnitudes to match display width
    fn bin_frequencies(&self, magnitudes: &[f32]) -> Vec<f32> {
        let mut binned = vec![0.0; self.current_bins];
        let step = magnitudes.len().max(1) as f32 / self.current_bins as f32;
        
        for i in 0..self.current_bins {
            let start = (i as f32 * step) as usize;
            let end = ((i + 1) as f32 * step) as usize;
            
            if start < magnitudes.len() {
                let end = end.min(magnitudes.len());
                let slice = &magnitudes[start..end];
                
                if !slice.is_empty() {
                    let avg = slice.iter().sum::<f32>() / slice.len() as f32;
                    
                    // Apply frequency-dependent gain
                    let freq_weight = 1.0 + config::HIGH_FREQ_BOOST * (i as f32) 
                        / (self.current_bins.saturating_sub(1).max(1) as f32);
                    
                    binned[i] = avg * freq_weight;
                }
            }
        }
        
        binned
    }
    
    /// Applies logarithmic scaling to magnitudes
    fn apply_log_scaling(&self, magnitudes: &[f32]) -> Vec<f32> {
        magnitudes
            .iter()
            .map(|&mag| {
                if mag > 0.0 {
                    let db = 20.0 * mag.log10();
                    let normalized = (db - config::MIN_DB) / (config::MAX_DB - config::MIN_DB);
                    normalized.clamp(0.0, 1.0)
                } else {
                    0.0
                }
            })
            .collect()
    }
    
    /// Adds a new row to the history
    fn add_row(&mut self, row: Vec<f32>) {
        if self.history.len() >= self.max_rows {
            self.history.remove(0);
        }
        self.history.push(row);
    }
    
    /// Returns an iterator over the display rows (newest at bottom)
    fn rows(&self) -> impl Iterator<Item = &[f32]> {
        self.history.iter().rev().map(|row| row.as_slice())
    }
}

/// Maps normalized values to colors for visualization
struct ColorMapper;

impl ColorMapper {
    /// Returns a color for the given normalized value (0.0 to 1.0)
    fn get_color(value: f32) -> Color {
        let value = value.clamp(0.0, 1.0);
        
        if value < 0.5 {
            // Black to purple gradient
            let t = value * 2.0;
            let r = (127.0 * t) as u8;
            let g = 0;
            let b = (127.0 * t) as u8;
            Color::Rgb(r, g, b)
        } else {
            // Purple to white gradient
            let t = (value - 0.5) * 2.0;
            let r = (127.0 + 128.0 * t) as u8;
            let g = (255.0 * t) as u8;
            let b = (127.0 + 128.0 * t) as u8;
            Color::Rgb(r, g, b)
        }
    }
}

/// Terminal UI manager
struct TerminalUI {
    terminal: Terminal<CrosstermBackend<io::Stdout>>,
}

impl TerminalUI {
    /// Creates and initializes the terminal UI
    fn new() -> Result<Self, SpectrogramError> {
        terminal::enable_raw_mode()?;
        execute!(
            io::stdout(),
            terminal::EnterAlternateScreen,
            event::EnableMouseCapture
        )?;
        
        let backend = CrosstermBackend::new(io::stdout());
        let terminal = Terminal::new(backend)?;
        
        Ok(Self { terminal })
    }
    
    /// Gets the current terminal width
    fn width(&self) -> Result<usize, SpectrogramError> {
        let size = self.terminal.size()?;
        Ok(size.width as usize)
    }
    
    /// Renders the spectrogram display
    fn render(&mut self, display: &SpectrogramDisplay) -> Result<(), SpectrogramError> {
        self.terminal.draw(|frame| {
            let area = frame.size();
            let chunks = Layout::default()
                .constraints([Constraint::Percentage(100)])
                .split(area);
            
            let lines: Vec<Line> = display
                .rows()
                .map(|row| {
                    let spans: Vec<Span> = row
                        .iter()
                        .map(|&value| {
                            Span::styled(
                                "█",
                                Style::default().fg(ColorMapper::get_color(value))
                            )
                        })
                        .collect();
                    Line::from(spans)
                })
                .collect();
            
            // Pad with empty lines if needed
            let mut padded_lines = lines;
            while padded_lines.len() < config::HISTORY_ROWS {
                padded_lines.push(Line::from(""));
            }
            
            let paragraph = Paragraph::new(padded_lines)
                .block(Block::default().title("Audio Spectrogram (Press 'q' or ESC to quit)"));
            
            frame.render_widget(paragraph, chunks[0]);
        })?;
        
        Ok(())
    }
    
    /// Checks for quit key press
    fn should_quit(&self) -> Result<bool, SpectrogramError> {
        if event::poll(config::REFRESH_INTERVAL)? {
            if let event::Event::Key(key) = event::read()? {
                return Ok(matches!(
                    key.code,
                    event::KeyCode::Char('q') | event::KeyCode::Esc
                ));
            }
        }
        Ok(false)
    }
}

impl Drop for TerminalUI {
    fn drop(&mut self) {
        // Restore terminal state
        let _ = execute!(
            io::stdout(),
            terminal::LeaveAlternateScreen,
            event::DisableMouseCapture
        );
        let _ = terminal::disable_raw_mode();
    }
}

/// Main application that coordinates all components
struct SpectrogramApp {
    audio_capture: AudioCapture,
    fft_analyzer: FftAnalyzer,
    display: Arc<Mutex<SpectrogramDisplay>>,
    ui: TerminalUI,
}

impl SpectrogramApp {
    /// Creates a new spectrogram application
    fn new() -> Result<Self, SpectrogramError> {
        let audio_capture = AudioCapture::new()?;
        let fft_analyzer = FftAnalyzer::new(config::FFT_SIZE);
        let ui = TerminalUI::new()?;
        
        let initial_width = ui.width()?;
        let display = Arc::new(Mutex::new(SpectrogramDisplay::new(
            config::HISTORY_ROWS,
            initial_width,
        )));
        
        Ok(Self {
            audio_capture,
            fft_analyzer,
            display,
            ui,
        })
    }
    
    /// Runs the main application loop
    fn run(mut self) -> Result<(), SpectrogramError> {
        self.audio_capture.start()?;
        
        loop {
            // Process audio samples
            while let Ok(sample) = self.audio_capture.try_recv() {
                if let Some(magnitudes) = self.fft_analyzer.add_sample(sample) {
                    let terminal_width = self.ui.width()?;
                    
                    let mut display = self.display.lock().unwrap();
                    display.update(&magnitudes, terminal_width);
                }
            }
            
            // Render UI
            {
                let display = self.display.lock().unwrap();
                self.ui.render(&display)?;
            }
            
            // Check for quit
            if self.ui.should_quit()? {
                break;
            }
        }
        
        Ok(())
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Print environment variable hint for Windows users
    #[cfg(target_os = "windows")]
    {
        if std::env::var("CPAL_WASAPI_REQUEST_FORCE_RAW").is_err() {
            eprintln!("Hint: Set CPAL_WASAPI_REQUEST_FORCE_RAW=1 to request raw audio input on Windows");
        }
    }
    
    let app = SpectrogramApp::new()?;
    app.run()?;
    
    Ok(())
} 