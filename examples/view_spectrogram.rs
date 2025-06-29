//! Spectrogram binary file viewer.
//!
//! This example loads a binary spectrogram file created by the `spectrogram`
//! example and visualizes it in the terminal.
//!
//! ## Usage
//!
//! First, record a spectrogram:
//! ```sh
//! cargo run --example spectrogram -- -o spectro.bin
//! ```
//!
//! Then, view it:
//! ```sh
//! cargo run --example view_spectrogram -- spectro.bin
//! ```
//!
//! ## Controls
//!
//! - `j` / `k`: Scroll line by line
//! - `u` / `d`: Scroll page by page
//! - `g`: Jump to the beginning
//! - `G`: Jump to the end
//! - `q` or `Ctrl+C`: Quit
//!
use std::io::{self, Read, Write, BufReader, Seek, SeekFrom};
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};
use std::fs::File;
use std::path::{Path, PathBuf};

// Global shutdown signal
static SHUTDOWN: AtomicBool = AtomicBool::new(false);

/// Cross-platform terminal control using ANSI escape sequences.
/// This module is a lightweight, dependency-free utility for terminal manipulation.
mod terminal {
    use std::io;
    #[cfg(unix)]
    use std::io::Read;

    #[cfg(windows)]
    #[link(name = "kernel32")]
    extern "system" {
        fn GetConsoleMode(handle: *mut std::ffi::c_void, mode: *mut u32) -> i32;
        fn SetConsoleMode(handle: *mut std::ffi::c_void, mode: u32) -> i32;
        fn GetConsoleScreenBufferInfo(h: *mut std::ffi::c_void, i: *mut CONSOLE_SCREEN_BUFFER_INFO) -> i32;
        fn GetNumberOfConsoleInputEvents(h: *mut std::ffi::c_void, n: *mut u32) -> i32;
        fn ReadConsoleInputA(h: *mut std::ffi::c_void, b: *mut INPUT_RECORD, l: u32, r: *mut u32) -> i32;
    }

    #[cfg(windows)]
    #[repr(C)]
    struct COORD { x: i16, y: i16 }
    #[cfg(windows)]
    #[repr(C)]
    struct SMALL_RECT { left: i16, top: i16, right: i16, bottom: i16 }
    #[cfg(windows)]
    #[repr(C)]
    struct CONSOLE_SCREEN_BUFFER_INFO {
        size: COORD, cursor_pos: COORD, attributes: u16, window: SMALL_RECT, max_window_size: COORD,
    }
    #[cfg(windows)]
    #[repr(C)]
    struct INPUT_RECORD { event_type: u16, _padding: u16, event: [u8; 16] }

    /// Enable raw mode.
    pub fn enable_raw_mode() -> io::Result<()> {
        #[cfg(unix)]
        {
            unsafe {
                let mut termios: libc::termios = std::mem::zeroed();
                if libc::tcgetattr(libc::STDIN_FILENO, &mut termios) != 0 {
                    return Err(io::Error::last_os_error());
                }
                let original = termios;
                termios.c_lflag &= !(libc::ICANON | libc::ECHO);
                if libc::tcsetattr(libc::STDIN_FILENO, libc::TCSANOW, &termios) != 0 {
                    return Err(io::Error::last_os_error());
                }
                // Store original termios to restore on exit
                std::mem::forget(original);
            }
        }
        
        #[cfg(windows)]
        {
            use std::os::windows::io::AsRawHandle;
            
            unsafe {
                const ENABLE_LINE_INPUT: u32 = 0x0002;
                const ENABLE_ECHO_INPUT: u32 = 0x0004;
                const ENABLE_VIRTUAL_TERMINAL_PROCESSING: u32 = 0x0004;

                // Set stdin to raw mode
                let stdin_handle = io::stdin().as_raw_handle();
                let mut stdin_mode = 0;
                if GetConsoleMode(stdin_handle as *mut _, &mut stdin_mode) == 0 {
                    return Err(io::Error::last_os_error());
                }
                let new_stdin_mode = stdin_mode & !(ENABLE_LINE_INPUT | ENABLE_ECHO_INPUT);
                if SetConsoleMode(stdin_handle as *mut _, new_stdin_mode) == 0 {
                    return Err(io::Error::last_os_error());
                }

                // Enable virtual terminal processing for stdout
                let stdout_handle = io::stdout().as_raw_handle();
                let mut stdout_mode = 0;
                if GetConsoleMode(stdout_handle as *mut _, &mut stdout_mode) == 0 {
                    return Err(io::Error::last_os_error());
                }
                let new_stdout_mode = stdout_mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING;
                if SetConsoleMode(stdout_handle as *mut _, new_stdout_mode) == 0 {
                    return Err(io::Error::last_os_error());
                }
            }
        }
        
        Ok(())
    }

    /// Disable raw mode
    pub fn disable_raw_mode() -> io::Result<()> {
        #[cfg(unix)]
        {
            unsafe {
                let mut termios: libc::termios = std::mem::zeroed();
                if libc::tcgetattr(libc::STDIN_FILENO, &mut termios) != 0 {
                    return Err(io::Error::last_os_error());
                }
                termios.c_lflag |= libc::ICANON | libc::ECHO;
                if libc::tcsetattr(libc::STDIN_FILENO, libc::TCSANOW, &termios) != 0 {
                     return Err(io::Error::last_os_error());
                }
            }
        }
        
        #[cfg(windows)]
        {
            use std::os::windows::io::AsRawHandle;
            
            unsafe {
                let handle = io::stdin().as_raw_handle();
                
                const PROCESSED_INPUT: u32 = 0x0001;
                const LINE_INPUT: u32 = 0x0002;
                const ECHO_INPUT: u32 = 0x0004;
                SetConsoleMode(handle as *mut _, PROCESSED_INPUT | LINE_INPUT | ECHO_INPUT);
            }
        }
        
        Ok(())
    }

    /// Get terminal size
    pub fn size() -> io::Result<(u16, u16)> {
        #[cfg(unix)]
        {
            unsafe {
                let mut size: libc::winsize = std::mem::zeroed();
                if libc::ioctl(libc::STDOUT_FILENO, libc::TIOCGWINSZ, &mut size) == 0 {
                    Ok((size.ws_col, size.ws_row))
                } else {
                    Ok((80, 24)) // Default fallback
                }
            }
        }
        
        #[cfg(windows)]
        {
            use std::os::windows::io::AsRawHandle;
            
            unsafe {
                let handle = io::stdout().as_raw_handle();
                let mut info: CONSOLE_SCREEN_BUFFER_INFO = std::mem::zeroed();
                
                if GetConsoleScreenBufferInfo(handle as *mut _, &mut info) != 0 {
                    let width = info.window.right - info.window.left + 1;
                    let height = info.window.bottom - info.window.top + 1;
                    Ok((width as u16, height as u16))
                } else {
                    Ok((80, 24)) // Default fallback
                }
            }
        }
        
        #[cfg(not(any(unix, windows)))] { Ok((80, 24)) }
    }

    /// Check if a key was pressed (non-blocking)
    pub fn key_pressed() -> io::Result<Option<char>> {
        #[cfg(unix)]
        {
            unsafe {
                let mut fds = libc::pollfd { fd: libc::STDIN_FILENO, events: libc::POLLIN, revents: 0 };
                if libc::poll(&mut fds, 1, 0) > 0 {
                    let mut buf = [0u8; 1];
                    if io::stdin().read(&mut buf)? > 0 {
                        return Ok(Some(buf[0] as char));
                    }
                }
            }
        }
        
        #[cfg(windows)]
        {
            use std::os::windows::io::AsRawHandle;
            unsafe {
                let handle = io::stdin().as_raw_handle();
                let mut event_count = 0u32;
                if GetNumberOfConsoleInputEvents(handle as *mut _, &mut event_count) != 0 && event_count > 0 {
                    let mut buffer: INPUT_RECORD = std::mem::zeroed();
                    let mut read = 0u32;
                    if ReadConsoleInputA(handle as *mut _, &mut buffer, 1, &mut read) != 0 && read > 0 && buffer.event_type == 1 {
                        let key_event = &buffer.event;
                        let key_down = key_event[0] != 0;
                        if key_down {
                           let ascii_char = key_event[10];
                           if ascii_char != 0 {
                               return Ok(Some(ascii_char as char));
                           }
                        }
                    }
                }
            }
        }
        Ok(None)
    }

    /// ANSI escape sequences
    pub const CURSOR_HOME: &str = "\x1b[H";
    pub const HIDE_CURSOR: &str = "\x1b[?25l";
    pub const SHOW_CURSOR: &str = "\x1b[?25h";
    pub const ALTERNATE_SCREEN: &str = "\x1b[?1049h";
    pub const NORMAL_SCREEN: &str = "\x1b[?1049l";
    pub const RESET_COLOR: &str = "\x1b[0m";
    pub const CLEAR_LINE: &str = "\x1b[K";
    
    pub fn set_color(r: u8, g: u8, b: u8) -> String {
        format!("\x1b[38;2;{};{};{}m", r, g, b)
    }
}

/// A RAII guard for managing the terminal state.
/// It sets up the terminal on creation and cleans up on drop.
struct TerminalUi;

impl TerminalUi {
    fn new() -> io::Result<Self> {
        terminal::enable_raw_mode()?;
        let mut stdout = io::stdout();
        stdout.write_all(terminal::ALTERNATE_SCREEN.as_bytes())?;
        stdout.write_all(terminal::HIDE_CURSOR.as_bytes())?;
        stdout.flush()?;
        Ok(Self)
    }
    
    fn cleanup() {
        let _ = terminal::disable_raw_mode();
        let mut stdout = io::stdout();
        let _ = stdout.write_all(terminal::NORMAL_SCREEN.as_bytes());
        let _ = stdout.write_all(terminal::SHOW_CURSOR.as_bytes());
        let _ = stdout.write_all(terminal::RESET_COLOR.as_bytes());
        let _ = stdout.flush();
    }
}

impl Drop for TerminalUi {
    fn drop(&mut self) {
        Self::cleanup();
    }
}

/// Reads and parses a spectrogram binary file.
/// This reader builds an in-memory index of row locations, allowing for
/// efficient, on-demand loading of spectrogram data from disk. This avoids
/// loading the entire file into RAM.
struct SpectrogramReader {
    path: PathBuf,
    reader: BufReader<File>,
    /// Index of (file_offset, timestamp_micros, bin_count) for each row.
    index: Vec<(u64, u64, u16)>,
    total_duration: Duration,
}

impl SpectrogramReader {
    /// Creates a new reader and builds an index of the spectrogram data.
    ///
    /// The binary format for each row is:
    /// * `u64` – microseconds elapsed since recording start (little-endian)
    /// * `u16` – number of magnitude bins in this row (little-endian)
    /// * `bins * f32` – raw magnitude values (little-endian)
    fn new<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let path_buf = path.as_ref().to_path_buf();
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        let mut index = Vec::new();
        let mut last_timestamp_micros = 0;
        let mut current_offset = 0;

        loop {
            let mut u64_bytes = [0; 8];
            match reader.read_exact(&mut u64_bytes) {
                Ok(_) => {}
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e),
            }
            let timestamp_micros = u64::from_le_bytes(u64_bytes);

            let mut u16_bytes = [0; 2];
            if reader.read_exact(&mut u16_bytes).is_err() {
                // Handle case where file is truncated mid-header
                break;
            }
            let len = u16::from_le_bytes(u16_bytes);

            index.push((current_offset, timestamp_micros, len));
            
            let bins_bytes = len as i64 * 4;
            current_offset += 8 + 2 + bins_bytes as u64;

            // Seek past the bin data to the next record header
            if reader.seek_relative(bins_bytes).is_err() {
                 // Handle case where file is truncated mid-bins
                break;
            }
            
            last_timestamp_micros = timestamp_micros;
        }
        
        Ok(Self {
            path: path_buf,
            reader,
            index,
            total_duration: Duration::from_micros(last_timestamp_micros),
        })
    }

    /// Returns the total number of rows in the spectrogram.
    fn row_count(&self) -> usize {
        self.index.len()
    }
    
    /// Returns the timestamp for a given row index.
    fn get_timestamp(&self, index: usize) -> Option<Duration> {
        self.index.get(index).map(|&(_, ts, _)| Duration::from_micros(ts))
    }

    /// Reads a single spectrogram row from the file by its index.
    fn get_row(&mut self, index: usize) -> io::Result<Option<Vec<f32>>> {
        let (offset, _, len) = match self.index.get(index) {
            Some(&data) => data,
            None => return Ok(None),
        };
        
        let len = len as usize;
        self.reader.seek(SeekFrom::Start(offset))?;

        // Skip timestamp (u64) and len (u16) fields to get to the bin data
        self.reader.seek_relative(8 + 2)?;

        let mut bins = vec![0.0f32; len];
        let mut bin_bytes = vec![0u8; len * 4];
        self.reader.read_exact(&mut bin_bytes)?;

        for (i, chunk) in bin_bytes.chunks_exact(4).enumerate() {
            bins[i] = f32::from_le_bytes(chunk.try_into().unwrap());
        }

        Ok(Some(bins))
    }
}

/// Main application for viewing the spectrogram.
/// Manages UI state, rendering, and user input.
struct ViewerApp {
    reader: SpectrogramReader,
    view_offset: usize,
    render_buffer: String,
    _ui: TerminalUi,
}

impl ViewerApp {
    fn new(reader: SpectrogramReader) -> io::Result<Self> {
        let (width, height) = terminal::size()?;
        Ok(Self {
            reader,
            view_offset: 0,
            render_buffer: String::with_capacity((width as usize) * (height as usize)),
            _ui: TerminalUi::new()?,
        })
    }
    
    /// Main application loop.
    fn run(&mut self) -> io::Result<()> {
        let mut last_render = Instant::now();
        let mut needs_render = true;

        loop {
            if SHUTDOWN.load(Ordering::Relaxed) {
                break;
            }
            
            if self.handle_input()? {
                needs_render = true;
            }
            
            // Render on changes or periodically
            if needs_render || last_render.elapsed() > Duration::from_millis(100) {
                self.render()?;
                last_render = Instant::now();
                needs_render = false;
            }

            thread::sleep(Duration::from_millis(10));
        }
        Ok(())
    }

    /// Handles user keyboard input for navigation.
    fn handle_input(&mut self) -> io::Result<bool> {
        let (_width, height) = terminal::size()?;
        let page_size = height.saturating_sub(3) as usize; // Header/footer
        let max_offset = self.reader.row_count().saturating_sub(page_size);

        if let Some(key) = terminal::key_pressed()? {
            match key {
                'q' | '\x1b' => SHUTDOWN.store(true, Ordering::SeqCst),
                'k' => self.view_offset = self.view_offset.saturating_sub(1),
                'j' => self.view_offset = (self.view_offset + 1).min(max_offset),
                'u' => self.view_offset = self.view_offset.saturating_sub(page_size),
                'd' => self.view_offset = (self.view_offset + page_size).min(max_offset),
                'g' => self.view_offset = 0,
                'G' => self.view_offset = max_offset,
                _ => return Ok(false),
            }
            return Ok(true);
        }
        Ok(false)
    }

    /// Renders the entire terminal UI.
    fn render(&mut self) -> io::Result<()> {
        let (width, height) = terminal::size()?;
        self.render_buffer.clear();
        
        self.render_header(width);
        self.render_spectrogram(width, height)?;
        self.render_footer(width, height);

        let mut stdout = io::stdout();
        stdout.write_all(self.render_buffer.as_bytes())?;
        stdout.flush()
    }

    fn render_header(&mut self, width: u16) {
        self.render_buffer.push_str(terminal::CURSOR_HOME);
        let filename = self.reader.path.file_name().unwrap_or_default().to_string_lossy();
        let title = format!("Spectrogram Viewer: {}", filename);
        self.render_buffer.push_str(&title);
        self.render_buffer.push_str(&" ".repeat(width.saturating_sub(title.len() as u16) as usize));
        self.render_buffer.push_str("\r\n");
        
        let separator = "-".repeat(width as usize);
        self.render_buffer.push_str(&separator);
        self.render_buffer.push_str("\r\n");
    }

    fn render_spectrogram(&mut self, width: u16, height: u16) -> io::Result<()> {
        let spectrogram_height = height.saturating_sub(3); // 2 for header, 1 for footer
        
        let start_index = self.view_offset;
        let end_index = (start_index + spectrogram_height as usize).min(self.reader.row_count());

        for i in start_index..end_index {
            if let Some(bins) = self.reader.get_row(i)? {
                let resampled = resample_bins(&bins, width as usize);
                for &value in &resampled {
                    let (r, g, b) = value_to_rgb(value);
                    self.render_buffer.push_str(&terminal::set_color(r, g, b));
                    self.render_buffer.push('█');
                }
                self.render_buffer.push_str(terminal::RESET_COLOR);
            }
            self.render_buffer.push_str(terminal::CLEAR_LINE);
            self.render_buffer.push_str("\r\n");
        }
        Ok(())
    }

    fn render_footer(&mut self, width: u16, height: u16) {
        let spectrogram_height = height.saturating_sub(3);
        let num_rendered = self.reader.row_count().saturating_sub(self.view_offset).min(spectrogram_height as usize);

        // Fill empty lines
        for _ in num_rendered..spectrogram_height as usize {
            self.render_buffer.push_str(terminal::CLEAR_LINE);
            self.render_buffer.push_str("\r\n");
        }
        
        let current_time = self.reader.get_timestamp(self.view_offset).unwrap_or_default();
        
        let time_info = format!(
            "Time: {:.2}s / {:.2}s",
            current_time.as_secs_f32(),
            self.reader.total_duration.as_secs_f32()
        );
        let row_info = format!(
            "Row: {} / {}",
            self.view_offset,
            self.reader.row_count().saturating_sub(1)
        );
        let controls = "[j/k, u/d, g/G] Navigate, [q] Quit";
        
        let left = format!("{} | {}", time_info, row_info);
        let right = controls;
        let padding = width.saturating_sub(left.len() as u16).saturating_sub(right.len() as u16);

        self.render_buffer.push_str(&left);
        self.render_buffer.push_str(&" ".repeat(padding as usize));
        self.render_buffer.push_str(right);
        self.render_buffer.push_str(terminal::CLEAR_LINE);
    }
}

/// Resamples a slice of bins to a new target width by averaging.
fn resample_bins(bins: &[f32], target_width: usize) -> Vec<f32> {
    if bins.is_empty() || target_width == 0 {
        return vec![0.0; target_width];
    }
    if bins.len() == target_width {
        return bins.to_vec();
    }
    
    let mut resampled = vec![0.0; target_width];
    let step = bins.len() as f32 / target_width as f32;

    for i in 0..target_width {
        let start = (i as f32 * step) as usize;
        let end = ((i + 1) as f32 * step) as usize;
        let slice = &bins[start..end.min(bins.len())];
        
        if !slice.is_empty() {
            resampled[i] = slice.iter().sum::<f32>() / slice.len() as f32;
        }
    }
    resampled
}

/// Convert normalized value (0.0-1.0) to RGB color using a heatmap gradient.
fn value_to_rgb(value: f32) -> (u8, u8, u8) {
    let value = value.clamp(0.0, 1.0);
    // Simple black -> purple -> white gradient
    if value < 0.5 {
        let t = value * 2.0; // 0..1
        ((127.0 * t) as u8, 0, (127.0 * t) as u8)
    } else {
        let t = (value - 0.5) * 2.0; // 0..1
        (
            (127.0 + 128.0 * t) as u8,
            (255.0 * t) as u8,
            (127.0 + 128.0 * t) as u8,
        )
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Graceful shutdown on Ctrl+C
    #[cfg(unix)]
    {
        // On Unix, we can use a signal handler
        extern "C" fn handle_sigint(_: libc::c_int) {
            SHUTDOWN.store(true, Ordering::SeqCst);
        }
        unsafe {
            let mut action: libc::sigaction = std::mem::zeroed();
            action.sa_sigaction = handle_sigint as libc::sighandler_t;
            libc::sigaction(libc::SIGINT, &action, std::ptr::null_mut());
        }
    }
    #[cfg(windows)]
    {
        // On Windows, we use a console control handler
        unsafe extern "system" fn ctrl_handler(ctrl_type: u32) -> i32 {
            if ctrl_type == 0 || ctrl_type == 1 { // CTRL_C_EVENT or CTRL_BREAK_EVENT
                SHUTDOWN.store(true, Ordering::SeqCst);
                1 // Handled
            } else {
                0 // Not handled
            }
        }
        unsafe {
            #[link(name = "kernel32")]
            extern "system" {
                fn SetConsoleCtrlHandler(h: Option<unsafe extern "system" fn(u32) -> i32>, add: i32) -> i32;
            }
            SetConsoleCtrlHandler(Some(ctrl_handler), 1);
        }
    }
    
    // Parse command-line arguments
    let path = std::env::args().nth(1).ok_or("Usage: view_spectrogram <path_to_file.bin>")?;
    
    let reader = SpectrogramReader::new(&path)?;
    if reader.row_count() == 0 {
        // Use eprint! here because terminal is not yet in raw mode.
        eprint!("Spectrogram file is empty or invalid. ");
        eprintln!("To generate one, run: cargo run --example spectrogram -- -o spectro.bin");
        return Ok(());
    }
    
    let mut app = ViewerApp::new(reader)?;
    if let Err(e) = app.run() {
        // Cleanup is handled by Drop trait, but we should print the error
        // eprintln does not work well with alternate screen buffer, so we clean up first
        TerminalUi::cleanup();
        eprintln!("\nAn error occurred: {}", e);
    }
    
    Ok(())
} 