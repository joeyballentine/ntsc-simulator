mod constants;
mod decoder;
mod encoder;
mod filters;

use std::io::{Read, Write};
use std::process::{Command, Stdio};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;

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
        /// Number of parallel worker threads (default: all logical cores)
        #[arg(long)]
        threads: Option<usize>,
    },
}

fn main() -> Result<()> {
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
            threads,
        } => cmd_roundtrip(&input, &output, width, height, comb_1h, crf, &preset, threads),
    }
}

fn cmd_image(input: &str, output: &str, width: Option<u32>, height: Option<u32>, comb_1h: bool) -> Result<()> {
    let img = image::open(input).with_context(|| format!("Cannot open image '{}'", input))?;

    let img_rgb = img.to_rgb8();
    let (w, h) = (img_rgb.width() as usize, img_rgb.height() as usize);
    let pixels = img_rgb.as_raw();

    let out_w = width.map(|v| v as usize).unwrap_or(w);
    let out_h = height.map(|v| v as usize).unwrap_or(h);

    eprintln!("Input: {} ({}x{})", input, w, h);
    eprintln!("Encoding to composite signal...");

    let mut encoder = Encoder::new();
    let t0 = Instant::now();
    let signal = encoder.encode_frame(pixels, w, h, 0);
    let encode_ms = t0.elapsed().as_secs_f64() * 1000.0;
    eprintln!("  Encode: {:.1} ms", encode_ms);

    eprintln!("Decoding from composite signal...");
    let mut decoder = Decoder::new();
    let t0 = Instant::now();
    let result_rgb = decoder.decode_frame(signal, out_w, out_h, comb_1h);
    let decode_ms = t0.elapsed().as_secs_f64() * 1000.0;
    eprintln!("  Decode: {:.1} ms", decode_ms);
    eprintln!(
        "  Total:  {:.1} ms ({:.1} fps)",
        encode_ms + decode_ms,
        1000.0 / (encode_ms + decode_ms)
    );

    // Save output
    let out_img =
        image::RgbImage::from_raw(out_w as u32, out_h as u32, result_rgb.to_vec())
            .context("Failed to create output image")?;
    out_img.save(output).with_context(|| format!("Cannot save image '{}'", output))?;

    eprintln!("Output: {} ({}x{})", output, out_w, out_h);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn cmd_roundtrip(
    input: &str,
    output: &str,
    width: u32,
    height: u32,
    comb_1h: bool,
    crf: u32,
    preset: &str,
    threads: Option<usize>,
) -> Result<()> {
    let out_w = width as usize;
    let out_h = height as usize;

    let (in_w, in_h, fps, total_frames) = ffprobe_video(input)?;
    eprintln!(
        "Input: {} ({}x{} @ {:.3} fps, {} frames)",
        input, in_w, in_h, fps, total_frames
    );
    eprintln!(
        "Output: {} ({}x{} @ {:.3} fps)",
        output, out_w, out_h, fps
    );

    let frame_bytes = in_w * in_h * 3;

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
        .context("Failed to spawn ffmpeg reader. Is ffmpeg installed?")?;

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
        .context("Failed to spawn ffmpeg writer. Is ffmpeg installed?")?;

    let reader_stdout = reader.stdout.take().unwrap();
    let mut reader_buf = std::io::BufReader::new(reader_stdout);
    let writer_stdin = writer.stdin.take().unwrap();
    let mut writer_buf = std::io::BufWriter::new(writer_stdin);

    let num_cpus = threads.unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
    });
    let batch_size = num_cpus;
    eprintln!("  Batch size: {} (parallel frames)", batch_size);

    rayon::ThreadPoolBuilder::new()
        .num_threads(num_cpus)
        .build_global()
        .ok();

    let mut frame_num = 0u32;
    let total_start = Instant::now();

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

    // Thread-local encoder/decoder — each rayon thread gets its own with
    // pre-allocated scratch buffers, avoiding cross-thread contention.
    use std::cell::RefCell;
    thread_local! {
        static TL_ENCODER: RefCell<Encoder> = RefCell::new(Encoder::new());
        static TL_DECODER: RefCell<Decoder> = RefCell::new(Decoder::new());
    }

    // Pre-allocate frame read buffers for the batch to avoid per-frame allocation
    let mut frame_pool: Vec<Vec<u8>> = (0..batch_size)
        .map(|_| vec![0u8; frame_bytes])
        .collect();

    loop {
        // Read a batch of frames, reusing pooled buffers
        let mut batch_count = 0usize;
        for buf in frame_pool.iter_mut() {
            match reader_buf.read_exact(buf) {
                Ok(()) => {
                    batch_count += 1;
                }
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => {
                    eprintln!("Read error: {}", e);
                    break;
                }
            }
        }

        if batch_count == 0 {
            break;
        }

        // Build references for the active batch items
        let batch_items: Vec<(&[u8], u32)> = (0..batch_count)
            .map(|i| (frame_pool[i].as_slice(), frame_num + i as u32))
            .collect();

        // Process all frames in the batch in parallel, each thread
        // uses its own encoder/decoder with reused scratch buffers
        let results: Vec<Vec<u8>> = batch_items
            .par_iter()
            .map(|(frame_buf, fnum)| {
                TL_ENCODER.with(|enc| {
                    TL_DECODER.with(|dec| {
                        let mut enc_ref = enc.borrow_mut();
                        let signal = enc_ref.encode_frame(frame_buf, in_w, in_h, *fnum);
                        let mut dec_ref = dec.borrow_mut();
                        let result = dec_ref.decode_frame(signal, out_w, out_h, comb_1h);
                        result.to_vec()
                    })
                })
            })
            .collect();

        // Write results in order
        for result in &results {
            writer_buf.write_all(result).context("write failed")?;
        }

        frame_num += batch_count as u32;
        pb.inc(batch_count as u64);

        let elapsed = total_start.elapsed().as_secs_f64();
        let fps_actual = frame_num as f64 / elapsed;
        pb.set_message(format!("{:.1} fps", fps_actual));
    }

    pb.finish_and_clear();

    drop(writer_buf);
    let _ = reader.wait();
    let _ = writer.wait();

    let total_elapsed = total_start.elapsed().as_secs_f64();
    eprintln!(
        "Done: {} frames in {:.1}s ({:.1} fps)",
        frame_num,
        total_elapsed,
        frame_num as f64 / total_elapsed,
    );

    // Mux audio from source
    mux_audio(input, output);
    Ok(())
}

fn ffprobe_video(path: &str) -> Result<(usize, usize, f64, usize)> {
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
        .context("Failed to run ffprobe. Is ffmpeg installed?")?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    let parts: Vec<&str> = stdout.trim().split(',').collect();

    if parts.len() < 3 {
        bail!("Cannot probe video '{}'. ffprobe output: {}", path, stdout);
    }

    let w: usize = parts[0].parse().unwrap_or(640);
    let h: usize = parts[1].parse().unwrap_or(480);

    let fps = if let Some((num, den)) = parts[2].split_once('/') {
        let n: f64 = num.parse().unwrap_or(30000.0);
        let d: f64 = den.parse().unwrap_or(1001.0);
        n / d
    } else {
        parts[2].parse().unwrap_or(29.97)
    };

    let total_frames: usize = parts
        .get(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    Ok((w, h, fps, total_frames))
}

fn mux_audio(source: &str, video_path: &str) {
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
