#![allow(dead_code)]

mod constants;
mod decoder;
mod encoder;
mod filters;

use std::io::{Read, Write};
use std::process::{Command, Stdio};
use std::time::Instant;

use clap::{Parser, Subcommand};
use indicatif::{ProgressBar, ProgressStyle};

use crate::decoder::Decoder;
use crate::encoder::Encoder;

#[derive(Parser)]
#[command(name = "ntsc-composite-simulator")]
#[command(about = "NTSC Composite Video Simulator (Rust)")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Roundtrip a single image through the NTSC composite pipeline
    Image {
        /// Input image file (PNG, JPG, etc.)
        input: String,
        /// Output image file
        #[arg(short, long, default_value = "output.png")]
        output: String,
        /// Output width (default: same as input)
        #[arg(long)]
        width: Option<u32>,
        /// Output height (default: same as input)
        #[arg(long)]
        height: Option<u32>,
        /// Use 1H line-delay comb filter
        #[arg(long)]
        comb_1h: bool,
    },
    /// Encode video to composite and decode back (roundtrip)
    Roundtrip {
        /// Input video file
        input: String,
        /// Output video file
        #[arg(short, long, default_value = "output.mp4")]
        output: String,
        /// Output width
        #[arg(long, default_value = "640")]
        width: u32,
        /// Output height
        #[arg(long, default_value = "480")]
        height: u32,
        /// Use 1H line-delay comb filter
        #[arg(long)]
        comb_1h: bool,
        /// x264 CRF quality (0=lossless, 51=worst)
        #[arg(long, default_value = "17")]
        crf: u32,
        /// x264 preset
        #[arg(long, default_value = "fast")]
        preset: String,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Image {
            input,
            output,
            width,
            height,
            comb_1h,
        } => cmd_image(&input, &output, width, height, comb_1h),
        Commands::Roundtrip {
            input,
            output,
            width,
            height,
            comb_1h,
            crf,
            preset,
        } => cmd_roundtrip(&input, &output, width, height, comb_1h, crf, &preset),
    }
}

fn cmd_image(input: &str, output: &str, width: Option<u32>, height: Option<u32>, comb_1h: bool) {
    let img = image::open(input).unwrap_or_else(|e| {
        eprintln!("Error: Cannot open image '{}': {}", input, e);
        std::process::exit(1);
    });

    let img_rgb = img.to_rgb8();
    let (w, h) = (img_rgb.width() as usize, img_rgb.height() as usize);
    let pixels = img_rgb.as_raw();

    let out_w = width.map(|v| v as usize).unwrap_or(w);
    let out_h = height.map(|v| v as usize).unwrap_or(h);

    eprintln!("Input: {} ({}x{})", input, w, h);
    eprintln!("Encoding to composite signal...");

    let encoder = Encoder::new();
    let t0 = Instant::now();
    let signal = encoder.encode_frame(pixels, w, h, 0);
    let encode_ms = t0.elapsed().as_secs_f64() * 1000.0;
    eprintln!("  Encode: {:.1} ms", encode_ms);

    eprintln!("Decoding from composite signal...");
    let decoder = Decoder::new();
    let t0 = Instant::now();
    let result_rgb = decoder.decode_frame(&signal, 0, out_w, out_h, comb_1h);
    let decode_ms = t0.elapsed().as_secs_f64() * 1000.0;
    eprintln!("  Decode: {:.1} ms", decode_ms);
    eprintln!(
        "  Total:  {:.1} ms ({:.1} fps)",
        encode_ms + decode_ms,
        1000.0 / (encode_ms + decode_ms)
    );

    // Save output
    let out_img =
        image::RgbImage::from_raw(out_w as u32, out_h as u32, result_rgb).expect("image creation");
    out_img.save(output).unwrap_or_else(|e| {
        eprintln!("Error: Cannot save image '{}': {}", output, e);
        std::process::exit(1);
    });

    eprintln!("Output: {} ({}x{})", output, out_w, out_h);
}

fn cmd_roundtrip(
    input: &str,
    output: &str,
    width: u32,
    height: u32,
    comb_1h: bool,
    crf: u32,
    preset: &str,
) {
    let out_w = width as usize;
    let out_h = height as usize;

    // Probe input video for dimensions, fps, and frame count
    let (in_w, in_h, fps, total_frames) = ffprobe_video(input);
    eprintln!(
        "Input: {} ({}x{} @ {:.3} fps, {} frames)",
        input, in_w, in_h, fps, total_frames
    );
    eprintln!(
        "Output: {} ({}x{} @ {:.3} fps)",
        output, out_w, out_h, fps
    );

    let frame_bytes = in_w * in_h * 3;

    // Spawn ffmpeg reader: decode input to raw RGB
    let mut reader = Command::new("ffmpeg")
        .args([
            "-i",
            input,
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-v",
            "error",
            "pipe:1",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to spawn ffmpeg reader. Is ffmpeg installed?");

    // Spawn ffmpeg writer: encode raw RGB to output
    let mut writer = Command::new("ffmpeg")
        .args([
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            &format!("{}x{}", out_w, out_h),
            "-r",
            &format!("{}", fps),
            "-i",
            "pipe:0",
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            &format!("{}", crf),
            "-pix_fmt",
            "yuv420p",
            "-v",
            "error",
            output,
        ])
        .stdin(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to spawn ffmpeg writer. Is ffmpeg installed?");

    let reader_stdout = reader.stdout.take().unwrap();
    let mut reader_buf = std::io::BufReader::new(reader_stdout);
    let writer_stdin = writer.stdin.take().unwrap();
    let mut writer_buf = std::io::BufWriter::new(writer_stdin);

    let encoder = Encoder::new();
    let decoder = Decoder::new();

    let mut frame_buf = vec![0u8; frame_bytes];
    let mut frame_num = 0u32;
    let total_start = Instant::now();
    let mut total_encode_ms = 0.0f64;
    let mut total_decode_ms = 0.0f64;

    let pb = if total_frames > 0 {
        let pb = ProgressBar::new(total_frames as u64);
        pb.set_style(
            ProgressStyle::with_template(
                "Processing {bar:40.cyan/blue} {pos}/{len} [{elapsed_precise}<{eta_precise}, {msg}]",
            )
            .unwrap()
            .progress_chars("##-"),
        );
        pb
    } else {
        let pb = ProgressBar::new_spinner();
        pb.set_style(
            ProgressStyle::with_template("Processing {spinner} {pos} frames [{elapsed_precise}, {msg}]")
                .unwrap(),
        );
        pb
    };

    loop {
        // Read one frame
        match reader_buf.read_exact(&mut frame_buf) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => {
                eprintln!("Read error: {}", e);
                break;
            }
        }

        // Encode
        let t0 = Instant::now();
        let signal = encoder.encode_frame(&frame_buf, in_w, in_h, frame_num);
        total_encode_ms += t0.elapsed().as_secs_f64() * 1000.0;

        // Decode
        let t0 = Instant::now();
        let result_rgb = decoder.decode_frame(&signal, frame_num, out_w, out_h, comb_1h);
        total_decode_ms += t0.elapsed().as_secs_f64() * 1000.0;

        // Write frame
        writer_buf.write_all(&result_rgb).expect("write failed");

        frame_num += 1;
        pb.inc(1);

        if frame_num % 10 == 0 {
            let elapsed = total_start.elapsed().as_secs_f64();
            let fps_actual = frame_num as f64 / elapsed;
            pb.set_message(format!("{:.1} fps", fps_actual));
        }
    }

    pb.finish_and_clear();

    drop(writer_buf);
    let _ = reader.wait();
    let _ = writer.wait();

    let total_elapsed = total_start.elapsed().as_secs_f64();
    eprintln!(
        "Done: {} frames in {:.1}s ({:.1} fps, enc {:.1}ms + dec {:.1}ms avg)",
        frame_num,
        total_elapsed,
        frame_num as f64 / total_elapsed,
        total_encode_ms / frame_num.max(1) as f64,
        total_decode_ms / frame_num.max(1) as f64,
    );

    // Mux audio from source
    mux_audio(input, output);
}

/// Use ffprobe to get video width, height, fps, and frame count.
fn ffprobe_video(path: &str) -> (usize, usize, f64, usize) {
    let output = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,nb_frames",
            "-of",
            "csv=p=0",
            path,
        ])
        .output()
        .expect("Failed to run ffprobe. Is ffmpeg installed?");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let parts: Vec<&str> = stdout.trim().split(',').collect();

    if parts.len() < 3 {
        eprintln!("Error: Cannot probe video '{}'. ffprobe output: {}", path, stdout);
        std::process::exit(1);
    }

    let w: usize = parts[0].parse().unwrap_or(640);
    let h: usize = parts[1].parse().unwrap_or(480);

    // r_frame_rate comes as "num/den"
    let fps = if let Some((num, den)) = parts[2].split_once('/') {
        let n: f64 = num.parse().unwrap_or(30000.0);
        let d: f64 = den.parse().unwrap_or(1001.0);
        n / d
    } else {
        parts[2].parse().unwrap_or(29.97)
    };

    // nb_frames may be "N/A" for some containers
    let total_frames: usize = parts
        .get(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    (w, h, fps, total_frames)
}

/// Copy audio from source into the output video file.
fn mux_audio(source: &str, video_path: &str) {
    // Check if source has audio
    let probe = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "csv=p=0",
            source,
        ])
        .output();

    let has_audio = match probe {
        Ok(out) => String::from_utf8_lossy(&out.stdout).contains("audio"),
        Err(_) => false,
    };

    if !has_audio {
        return;
    }

    let tmp = format!("{}.mux.mp4", video_path);
    let result = Command::new("ffmpeg")
        .args([
            "-y",
            "-i",
            video_path,
            "-i",
            source,
            "-map",
            "0:v",
            "-map",
            "1:a",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            &tmp,
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();

    match result {
        Ok(status) if status.success() => {
            let _ = std::fs::rename(&tmp, video_path);
        }
        _ => {
            let _ = std::fs::remove_file(&tmp);
        }
    }
}
