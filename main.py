"""CLI entry point for the NTSC Composite Video Simulator."""

import argparse
import multiprocessing
import os
import shutil
import subprocess
import sys

import numpy as np
from tqdm import tqdm


class FFmpegWriter:
    """Write video frames by piping raw RGB into ffmpeg.

    Supports interlaced output with proper field flags.
    """

    def __init__(self, filepath, width, height, fps=29.97, interlaced=False,
                 crf=17, preset='fast'):
        self.width = width
        self.height = height

        cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo',
            '-pix_fmt', 'rgb24',
            '-s', f'{width}x{height}',
            '-r', str(fps),
            '-i', 'pipe:0',
        ]

        if interlaced:
            # Set interlaced flags: top-field-first, interleaved fields
            cmd += [
                '-vf', 'setfield=tff',
                '-flags', '+ilme+ildct',
                '-top', '1',
            ]

        cmd += [
            '-c:v', 'libx264',
            '-preset', preset,
            '-crf', str(crf),
            '-pix_fmt', 'yuv420p',
            filepath,
        ]

        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    def write(self, frame_rgb):
        """Write one frame (H x W x 3 uint8 RGB array)."""
        self.proc.stdin.write(frame_rgb.tobytes())

    def release(self):
        self.proc.stdin.close()
        self.proc.wait()


class CV2Writer:
    """Fallback video writer using OpenCV (no interlace flags)."""

    def __init__(self, filepath, width, height, fps=29.97):
        import cv2
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(filepath, fourcc, fps, (width, height))

    def write(self, frame_rgb):
        import cv2
        self.out.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

    def release(self):
        self.out.release()


def _make_writer(filepath, width, height, fps=29.97, interlaced=False,
                 crf=17, preset='fast'):
    """Create a video writer, preferring ffmpeg for interlaced output."""
    if shutil.which('ffmpeg'):
        return FFmpegWriter(filepath, width, height, fps, interlaced,
                            crf=crf, preset=preset)
    if interlaced:
        print("Warning: ffmpeg not found, interlace flags will not be set")
    return CV2Writer(filepath, width, height, fps)


def _mux_audio(source_path, video_path):
    """Copy audio from source into the output video file using ffmpeg.

    Replaces video_path in-place if audio is found.
    """
    if not shutil.which('ffmpeg'):
        return

    # Check if source has audio
    probe = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'a',
         '-show_entries', 'stream=codec_type', '-of', 'csv=p=0',
         source_path],
        capture_output=True, text=True,
    )
    if 'audio' not in probe.stdout:
        return

    # Mux: copy video from our output + copy audio from source
    tmp = video_path + '.mux.mp4'
    result = subprocess.run(
        ['ffmpeg', '-y',
         '-i', video_path,
         '-i', source_path,
         '-map', '0:v', '-map', '1:a',
         '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k',
         '-shortest',
         tmp],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    if result.returncode == 0:
        os.replace(tmp, video_path)
    else:
        # Clean up temp file on failure
        if os.path.exists(tmp):
            os.remove(tmp)


def _read_input(path):
    """Open a video file with OpenCV, return (capture, frame_count)."""
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file '{path}'")
        sys.exit(1)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, total


def _read_frame_rgb(cap):
    """Read one frame as RGB, or return None at end."""
    import cv2
    ret, frame = cap.read()
    if not ret:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def cmd_encode(args):
    """Encode a video file to a composite NTSC signal (.npy)."""
    from ntsc_simulator.encoder import encode_frame
    from ntsc_simulator.signal_io import export_signal, export_wav

    cap, total_frames = _read_input(args.input)
    print(f"Encoding {args.input} ({total_frames} frames) to composite signal...")

    all_signals = []
    frame_num = 0

    for _ in tqdm(range(total_frames), unit='frame', desc='Encoding'):
        frame_rgb = _read_frame_rgb(cap)
        if frame_rgb is None:
            break
        all_signals.append(encode_frame(frame_rgb, frame_number=frame_num))
        frame_num += 1

    cap.release()

    if not all_signals:
        print("Error: No frames read from video")
        sys.exit(1)

    full_signal = np.concatenate(all_signals)
    print(f"Exporting signal ({len(full_signal)} samples, "
          f"{len(full_signal) / 14318180:.2f}s)...")
    export_signal(full_signal, args.output)
    print(f"Done: {args.output}")

    if args.wav:
        export_wav(full_signal, args.wav)
        print(f"WAV: {args.wav}")


def _build_pipeline(args):
    """Build a SignalPipeline from CLI args, or return None if no effects."""
    effects = _build_effects_dict(args)
    if not effects:
        return None
    return _pipeline_from_dict(effects)


def _pipeline_from_dict(effects):
    """Build a SignalPipeline from an effects dict."""
    from ntsc_simulator.pipeline import SignalPipeline
    from ntsc_simulator.effects import (add_noise, add_ghosting,
                                        add_attenuation, add_jitter)

    factories = {
        'noise': add_noise,
        'ghost': add_ghosting,
        'attenuation': add_attenuation,
        'jitter': add_jitter,
    }

    pipeline = SignalPipeline()
    for name, params in effects.items():
        pipeline.add(factories[name](**params))
    return pipeline


def _build_effects_dict(args):
    """Build an effects dict from CLI args (plain data, picklable)."""
    effects = {}
    if getattr(args, 'noise', None):
        effects['noise'] = {'amplitude': args.noise}
    if getattr(args, 'ghost', None):
        effects['ghost'] = {'amplitude': args.ghost,
                            'delay_us': args.ghost_delay}
    if getattr(args, 'attenuation', None):
        effects['attenuation'] = {'strength': args.attenuation}
    if getattr(args, 'jitter', None):
        effects['jitter'] = {'amplitude': args.jitter}
    return effects


def cmd_decode(args):
    """Decode a composite NTSC signal (.npy) back to video."""
    from ntsc_simulator.decoder import decode_frame
    from ntsc_simulator.signal_io import import_signal
    from ntsc_simulator.constants import TOTAL_LINES, SAMPLES_PER_LINE

    print(f"Importing signal from {args.input}...")
    signal, sample_rate = import_signal(args.input)
    print(f"  {len(signal)} samples at {sample_rate} Hz")

    samples_per_frame = TOTAL_LINES * SAMPLES_PER_LINE
    num_frames = len(signal) // samples_per_frame
    print(f"  {num_frames} frames detected")

    pipeline = _build_pipeline(args)
    if pipeline:
        print(f"  Applying {len(pipeline)} signal effect(s)")
        signal = pipeline.process(signal, sample_rate)

    width = args.width
    height = args.height
    out = _make_writer(args.output, width, height,
                       crf=args.crf, preset=args.preset)

    comb_1h = getattr(args, 'comb_1h', False)
    for i in tqdm(range(num_frames), unit='frame', desc='Decoding'):
        frame_signal = signal[i * samples_per_frame:(i + 1) * samples_per_frame]
        frame_rgb = decode_frame(frame_signal, frame_number=i,
                                 output_width=width, output_height=height,
                                 comb_1h=comb_1h)
        out.write(frame_rgb)

    out.release()
    print(f"Done: {args.output}")


def _ntsc_worker(args):
    """Worker process: encode frame through NTSC composite and decode back.

    Must be at module level for pickling by multiprocessing.
    """
    from ntsc_simulator.encoder import encode_frame
    from ntsc_simulator.decoder import decode_frame

    frame_rgb, frame_number, width, height, field2_frame, comb_1h, effects = args
    signal = encode_frame(frame_rgb, frame_number=frame_number,
                          field2_frame=field2_frame)
    if effects:
        pipeline = _pipeline_from_dict(effects)
        signal = pipeline.process(signal)
    return decode_frame(signal, frame_number=frame_number,
                        output_width=width, output_height=height,
                        comb_1h=comb_1h)


def _get_num_workers():
    """Get number of parallel workers (leave one core free for I/O)."""
    return max(1, os.cpu_count() - 1)


def cmd_roundtrip(args):
    """Encode video to composite signal and decode back."""
    cap, total_frames = _read_input(args.input)
    width = args.width
    height = args.height
    workers = _get_num_workers()

    comb_1h = getattr(args, 'comb_1h', False)

    crf = args.crf
    preset = args.preset

    # Build effects dict (plain data, safe for multiprocessing pickling)
    effects = _build_effects_dict(args)
    if effects:
        print(f"  Signal effects: {', '.join(effects.keys())}")

    if args.telecine:
        out = _make_writer(args.output, width, height, fps=29.97, interlaced=True,
                           crf=crf, preset=preset)
        print(f"Roundtrip (3:2 telecine 480i): {args.input} -> composite -> {args.output}")
        print(f"  Output: {width}x{height} 29.97fps interlaced (TFF), {workers} workers")
        _roundtrip_telecine(cap, out, width, height, total_frames, workers,
                            comb_1h, effects)
    else:
        out = _make_writer(args.output, width, height, fps=29.97, interlaced=False,
                           crf=crf, preset=preset)
        print(f"Roundtrip (progressive): {args.input} -> composite -> {args.output}")
        print(f"  Output: {width}x{height} 29.97fps progressive, {workers} workers")
        _roundtrip_progressive(cap, out, width, height, total_frames, workers,
                               comb_1h, effects)

    cap.release()
    out.release()

    # Mux audio from source into output
    _mux_audio(args.input, args.output)


def _roundtrip_progressive(cap, out, width, height, total_frames, workers,
                           comb_1h=False, effects=None):
    """Progressive roundtrip with parallel processing."""
    frame_num = 0
    batch_size = workers * 2
    pbar = tqdm(total=total_frames, unit='frame', desc='Processing')

    with multiprocessing.Pool(workers) as pool:
        while True:
            # Read a batch of frames
            batch = []
            for _ in range(batch_size):
                frame_rgb = _read_frame_rgb(cap)
                if frame_rgb is None:
                    break
                batch.append((frame_rgb, frame_num + len(batch), width, height,
                              None, comb_1h, effects))

            if not batch:
                break

            # Process batch in parallel, results come back in order
            results = pool.map(_ntsc_worker, batch)

            for result in results:
                out.write(result)
                frame_num += 1

            pbar.update(len(results))

    pbar.close()
    print(f"Done: {frame_num} output frames")


def _roundtrip_telecine(cap, out, width, height, total_frames, workers,
                        comb_1h=False, effects=None):
    """3:2 pulldown telecine with parallel processing.

    Pulldown pattern per group of 4 film frames A, B, C, D:
      NTSC frame 1: field1=A, field2=A  (clean)
      NTSC frame 2: field1=B, field2=B  (clean)
      NTSC frame 3: field1=B, field2=C  (combed)
      NTSC frame 4: field1=C, field2=D  (combed)
      NTSC frame 5: field1=D, field2=D  (clean)
    """
    ntsc_num = 0
    film_idx = 0
    # Process multiple groups at once: read N groups, expand to NTSC jobs, process in parallel
    groups_per_batch = max(1, workers)  # N groups -> 5N NTSC frames per batch
    pbar = tqdm(total=total_frames, unit='film frame', desc='Processing')

    with multiprocessing.Pool(workers) as pool:
        while True:
            # Read groups_per_batch * 4 film frames
            film_buf = []
            for _ in range(groups_per_batch * 4):
                frame_rgb = _read_frame_rgb(cap)
                if frame_rgb is None:
                    break
                film_buf.append(frame_rgb)

            if not film_buf:
                break

            # Expand film frames into NTSC (field1, field2) jobs
            jobs = []
            fi = 0
            while fi + 3 < len(film_buf):
                a, b, c, d = film_buf[fi], film_buf[fi+1], film_buf[fi+2], film_buf[fi+3]
                for f1, f2 in [(a, a), (b, b), (b, c), (c, d), (d, d)]:
                    jobs.append((f1, ntsc_num + len(jobs), width, height, f2,
                                 comb_1h, effects))
                fi += 4

            # Handle remaining < 4 frames as progressive
            while fi < len(film_buf):
                f = film_buf[fi]
                jobs.append((f, ntsc_num + len(jobs), width, height, None,
                             comb_1h, effects))
                fi += 1

            if not jobs:
                break

            # Process all NTSC frames in this batch in parallel
            results = pool.map(_ntsc_worker, jobs)

            for result in results:
                out.write(result)

            film_idx += len(film_buf)
            ntsc_num += len(results)
            pbar.update(len(film_buf))

    pbar.close()
    print(f"Done: {film_idx} film frames -> {ntsc_num} NTSC frames")


def cmd_image(args):
    """Roundtrip a single image through the NTSC composite pipeline."""
    import cv2
    from ntsc_simulator.encoder import encode_frame
    from ntsc_simulator.decoder import decode_frame
    from ntsc_simulator.signal_io import export_signal, export_wav

    frame_bgr = cv2.imread(args.input)
    if frame_bgr is None:
        print(f"Error: Cannot open image '{args.input}'")
        sys.exit(1)

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    print(f"Input: {args.input} ({frame_rgb.shape[1]}x{frame_rgb.shape[0]})")

    print("Encoding to composite signal...")
    signal = encode_frame(frame_rgb, frame_number=0)

    if args.signal:
        export_signal(signal, args.signal)
        print(f"Signal saved: {args.signal}")
    if args.wav:
        export_wav(signal, args.wav)
        print(f"WAV: {args.wav}")

    pipeline = _build_pipeline(args)
    if pipeline:
        print(f"Applying {len(pipeline)} signal effect(s)...")
        signal = pipeline.process(signal)

    print("Decoding from composite signal...")
    width = args.width or frame_rgb.shape[1]
    height = args.height or frame_rgb.shape[0]
    comb_1h = getattr(args, 'comb_1h', False)
    result_rgb = decode_frame(signal, frame_number=0,
                              output_width=width, output_height=height,
                              comb_1h=comb_1h)

    result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.output, result_bgr)
    print(f"Output: {args.output} ({width}x{height})")


def cmd_colorbars(args):
    """Generate SMPTE color bars and encode to composite signal."""
    from ntsc_simulator.colorbars import generate_colorbars
    from ntsc_simulator.encoder import encode_frame
    from ntsc_simulator.signal_io import export_signal, export_wav

    print("Generating SMPTE color bars...")
    bars = generate_colorbars(640, 480)

    print("Encoding to composite signal...")
    signal = encode_frame(bars, frame_number=0)

    print(f"Exporting signal ({len(signal)} samples)...")
    export_signal(signal, args.output)
    print(f"Done: {args.output}")

    if args.wav:
        export_wav(signal, args.wav)
        print(f"WAV: {args.wav}")

    if args.save_png:
        import cv2
        cv2.imwrite(args.save_png, cv2.cvtColor(bars, cv2.COLOR_RGB2BGR))
        print(f"Saved source pattern: {args.save_png}")


def _add_effect_args(parser):
    """Add signal degradation flags to an argparse subparser."""
    group = parser.add_argument_group('signal effects')
    group.add_argument('--noise', type=float, default=None,
                       help='Snow amplitude (e.g. 0.05=subtle, 0.2=heavy)')
    group.add_argument('--ghost', type=float, default=None,
                       help='Ghost amplitude 0-1 (multipath echo)')
    group.add_argument('--ghost-delay', type=float, default=2.0,
                       help='Ghost delay in microseconds (default: 2.0)')
    group.add_argument('--attenuation', type=float, default=None,
                       help='Signal attenuation 0-1 (washed-out picture)')
    group.add_argument('--jitter', type=float, default=None,
                       help='Horizontal jitter in samples (timing instability)')


def main():
    parser = argparse.ArgumentParser(
        description="NTSC Composite Video Simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python main.py encode input.mp4 -o signal.npy
  python main.py decode signal.npy -o output.mp4
  python main.py roundtrip input.mp4 -o output.mp4
  python main.py roundtrip input.mp4 -o output.mp4 --telecine
  python main.py image photo.png -o ntsc_photo.png
  python main.py colorbars -o colorbars.npy
        """)

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # encode
    p_enc = subparsers.add_parser('encode', help='Encode video to composite signal')
    p_enc.add_argument('input', help='Input video file')
    p_enc.add_argument('-o', '--output', default='signal.npy', help='Output signal file (.npy)')
    p_enc.add_argument('--wav', default=None, help='Also export as WAV (stretched to 48 kHz for audio editors)')

    # decode
    p_dec = subparsers.add_parser('decode', help='Decode composite signal to video')
    p_dec.add_argument('input', help='Input signal file (.npy)')
    p_dec.add_argument('-o', '--output', default='output.mp4', help='Output video file')
    p_dec.add_argument('--width', type=int, default=640, help='Output width')
    p_dec.add_argument('--height', type=int, default=480, help='Output height')
    p_dec.add_argument('--comb-1h', action='store_true',
                       help='Use 1H line-delay comb filter (reduces rainbow, adds hanging dots)')
    p_dec.add_argument('--crf', type=int, default=17, help='x264 CRF quality (0=lossless, 51=worst, default: 17)')
    p_dec.add_argument('--preset', default='fast',
                       help='x264 preset (ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow, default: fast)')
    _add_effect_args(p_dec)

    # roundtrip
    p_rt = subparsers.add_parser('roundtrip', help='Video -> composite -> video')
    p_rt.add_argument('input', help='Input video file')
    p_rt.add_argument('-o', '--output', default='output.mp4', help='Output video file')
    p_rt.add_argument('--width', type=int, default=640, help='Output width')
    p_rt.add_argument('--height', type=int, default=480, help='Output height')
    p_rt.add_argument('--telecine', action='store_true',
                      help='Simulate 3:2 pulldown telecine (480i, 4 film frames -> 5 NTSC frames)')
    p_rt.add_argument('--comb-1h', action='store_true',
                      help='Use 1H line-delay comb filter (reduces rainbow, adds hanging dots)')
    p_rt.add_argument('--crf', type=int, default=17, help='x264 CRF quality (0=lossless, 51=worst, default: 17)')
    p_rt.add_argument('--preset', default='fast',
                      help='x264 preset (ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow, default: fast)')
    _add_effect_args(p_rt)

    # image
    p_img = subparsers.add_parser('image', help='Roundtrip a single image through NTSC')
    p_img.add_argument('input', help='Input image file (PNG, JPG, etc.)')
    p_img.add_argument('-o', '--output', default='output.png', help='Output image file')
    p_img.add_argument('--width', type=int, default=None, help='Output width (default: same as input)')
    p_img.add_argument('--height', type=int, default=None, help='Output height (default: same as input)')
    p_img.add_argument('--comb-1h', action='store_true',
                       help='Use 1H line-delay comb filter (reduces rainbow, adds hanging dots)')
    p_img.add_argument('--signal', default=None, help='Also export composite signal (.npy)')
    p_img.add_argument('--wav', default=None, help='Also export as WAV (stretched to 48 kHz for audio editors)')
    _add_effect_args(p_img)

    # colorbars
    p_cb = subparsers.add_parser('colorbars', help='Generate color bar test signal')
    p_cb.add_argument('-o', '--output', default='colorbars.npy', help='Output signal file (.npy)')
    p_cb.add_argument('--wav', default=None, help='Also export as WAV (stretched to 48 kHz for audio editors)')
    p_cb.add_argument('--save-png', default=None, help='Also save source pattern as PNG')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    commands = {
        'encode': cmd_encode,
        'decode': cmd_decode,
        'roundtrip': cmd_roundtrip,
        'image': cmd_image,
        'colorbars': cmd_colorbars,
    }
    commands[args.command](args)


if __name__ == '__main__':
    main()
