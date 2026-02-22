"""CLI entry point for the NTSC Composite Video Simulator."""

import argparse
import shutil
import subprocess
import sys

import numpy as np


class FFmpegWriter:
    """Write video frames by piping raw RGB into ffmpeg.

    Supports interlaced output with proper field flags.
    """

    def __init__(self, filepath, width, height, fps=29.97, interlaced=False):
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
            '-preset', 'fast',
            '-crf', '17',
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


def _make_writer(filepath, width, height, fps=29.97, interlaced=False):
    """Create a video writer, preferring ffmpeg for interlaced output."""
    if shutil.which('ffmpeg'):
        return FFmpegWriter(filepath, width, height, fps, interlaced)
    if interlaced:
        print("Warning: ffmpeg not found, interlace flags will not be set")
    return CV2Writer(filepath, width, height, fps)


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
    """Encode a video file to a composite NTSC WAV signal."""
    from ntsc_simulator.encoder import encode_frame
    from ntsc_simulator.wav_io import export_wav

    cap, total_frames = _read_input(args.input)
    print(f"Encoding {args.input} ({total_frames} frames) to composite signal...")

    all_signals = []
    frame_num = 0

    while True:
        frame_rgb = _read_frame_rgb(cap)
        if frame_rgb is None:
            break
        all_signals.append(encode_frame(frame_rgb, frame_number=frame_num))
        frame_num += 1
        if frame_num % 10 == 0:
            print(f"  Encoded frame {frame_num}/{total_frames}")

    cap.release()

    if not all_signals:
        print("Error: No frames read from video")
        sys.exit(1)

    full_signal = np.concatenate(all_signals)
    print(f"Exporting WAV ({len(full_signal)} samples, "
          f"{len(full_signal) / 14318180:.2f}s)...")
    export_wav(full_signal, args.output)
    print(f"Done: {args.output}")


def cmd_decode(args):
    """Decode a composite NTSC WAV signal back to video."""
    from ntsc_simulator.decoder import decode_frame
    from ntsc_simulator.wav_io import import_wav
    from ntsc_simulator.constants import TOTAL_LINES, SAMPLES_PER_LINE

    print(f"Importing WAV signal from {args.input}...")
    signal, sample_rate = import_wav(args.input)
    print(f"  {len(signal)} samples at {sample_rate} Hz")

    samples_per_frame = TOTAL_LINES * SAMPLES_PER_LINE
    num_frames = len(signal) // samples_per_frame
    print(f"  {num_frames} frames detected")

    width = args.width
    height = args.height
    out = _make_writer(args.output, width, height)

    for i in range(num_frames):
        frame_signal = signal[i * samples_per_frame:(i + 1) * samples_per_frame]
        frame_rgb = decode_frame(frame_signal, frame_number=i,
                                 output_width=width, output_height=height)
        out.write(frame_rgb)
        if (i + 1) % 10 == 0:
            print(f"  Decoded frame {i + 1}/{num_frames}")

    out.release()
    print(f"Done: {args.output}")


def cmd_roundtrip(args):
    """Encode video to composite signal and decode back."""
    from ntsc_simulator.encoder import encode_frame
    from ntsc_simulator.decoder import decode_frame
    from ntsc_simulator.pipeline import SignalPipeline

    cap, total_frames = _read_input(args.input)
    width = args.width
    height = args.height
    pipeline = SignalPipeline()

    if args.telecine:
        out = _make_writer(args.output, width, height, fps=29.97, interlaced=True)
        print(f"Roundtrip (3:2 telecine 480i): {args.input} -> composite -> {args.output}")
        print(f"  Output: {width}x{height} 29.97fps interlaced (TFF)")
        _roundtrip_telecine(cap, out, pipeline, encode_frame, decode_frame,
                            width, height, total_frames)
    else:
        out = _make_writer(args.output, width, height, fps=29.97, interlaced=False)
        print(f"Roundtrip (progressive): {args.input} -> composite -> {args.output}")
        print(f"  Output: {width}x{height} 29.97fps progressive")
        _roundtrip_progressive(cap, out, pipeline, encode_frame, decode_frame,
                               width, height, total_frames)

    cap.release()
    out.release()


def _roundtrip_progressive(cap, out, pipeline, encode_frame, decode_frame,
                           width, height, total_frames):
    """Progressive roundtrip: each input frame -> one NTSC frame."""
    frame_num = 0
    while True:
        frame_rgb = _read_frame_rgb(cap)
        if frame_rgb is None:
            break
        signal = encode_frame(frame_rgb, frame_number=frame_num)
        if len(pipeline) > 0:
            signal = pipeline.process(signal)
        result = decode_frame(signal, frame_number=frame_num,
                              output_width=width, output_height=height)
        out.write(result)
        frame_num += 1
        if frame_num % 10 == 0:
            print(f"  Frame {frame_num}/{total_frames}")
    print(f"Done: {frame_num} output frames")


def _roundtrip_telecine(cap, out, pipeline, encode_frame, decode_frame,
                        width, height, total_frames):
    """3:2 pulldown telecine: 4 input frames -> 5 NTSC frames (480i).

    Streams frames in groups of 4 to avoid loading entire video into memory.

    Pulldown pattern per group of 4 film frames A, B, C, D:
      NTSC frame 1: field1=A, field2=A  (clean)
      NTSC frame 2: field1=B, field2=B  (clean)
      NTSC frame 3: field1=B, field2=C  (combed)
      NTSC frame 4: field1=C, field2=D  (combed)
      NTSC frame 5: field1=D, field2=D  (clean)
    """
    def _encode_decode(f1, f2, ntsc_num):
        signal = encode_frame(f1, frame_number=ntsc_num, field2_frame=f2)
        if len(pipeline) > 0:
            signal = pipeline.process(signal)
        return decode_frame(signal, frame_number=ntsc_num,
                            output_width=width, output_height=height)

    ntsc_num = 0
    film_idx = 0
    buf = []  # Rolling buffer of up to 4 frames

    while True:
        # Fill buffer to 4 frames
        while len(buf) < 4:
            frame_rgb = _read_frame_rgb(cap)
            if frame_rgb is None:
                break
            buf.append(frame_rgb)

        if len(buf) >= 4:
            a, b, c, d = buf[0], buf[1], buf[2], buf[3]

            for f1, f2 in [(a, a), (b, b), (b, c), (c, d), (d, d)]:
                out.write(_encode_decode(f1, f2, ntsc_num))
                ntsc_num += 1

            film_idx += 4
            buf = buf[4:]  # Advance past the group

            if ntsc_num % 25 == 0:
                print(f"  NTSC frame {ntsc_num} (film frame {film_idx}/{total_frames})")
        else:
            # Fewer than 4 remaining — output as progressive
            for frame in buf:
                out.write(_encode_decode(frame, frame, ntsc_num))
                ntsc_num += 1
                film_idx += 1
            break

    print(f"Done: {film_idx} film frames -> {ntsc_num} NTSC frames")


def cmd_image(args):
    """Roundtrip a single image through the NTSC composite pipeline."""
    import cv2
    from ntsc_simulator.encoder import encode_frame
    from ntsc_simulator.decoder import decode_frame
    from ntsc_simulator.wav_io import export_wav, export_wav_preview

    frame_bgr = cv2.imread(args.input)
    if frame_bgr is None:
        print(f"Error: Cannot open image '{args.input}'")
        sys.exit(1)

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    print(f"Input: {args.input} ({frame_rgb.shape[1]}x{frame_rgb.shape[0]})")

    print("Encoding to composite signal...")
    signal = encode_frame(frame_rgb, frame_number=0)

    if args.wav:
        export_wav(signal, args.wav)
        print(f"Signal WAV: {args.wav}")
    if args.preview:
        export_wav_preview(signal, args.preview)
        print(f"Preview WAV: {args.preview}")

    print("Decoding from composite signal...")
    width = args.width or frame_rgb.shape[1]
    height = args.height or frame_rgb.shape[0]
    result_rgb = decode_frame(signal, frame_number=0,
                              output_width=width, output_height=height)

    result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.output, result_bgr)
    print(f"Output: {args.output} ({width}x{height})")


def cmd_colorbars(args):
    """Generate SMPTE color bars and encode to composite signal."""
    from ntsc_simulator.colorbars import generate_colorbars
    from ntsc_simulator.encoder import encode_frame
    from ntsc_simulator.wav_io import export_wav, export_wav_preview

    print("Generating SMPTE color bars...")
    bars = generate_colorbars(640, 480)

    print("Encoding to composite signal...")
    signal = encode_frame(bars, frame_number=0)

    print(f"Exporting WAV ({len(signal)} samples)...")
    export_wav(signal, args.output)
    print(f"Done: {args.output}")

    if args.preview:
        print("Exporting 48 kHz preview WAV...")
        export_wav_preview(signal, args.preview)
        print(f"Preview: {args.preview}")

    if args.save_png:
        import cv2
        cv2.imwrite(args.save_png, cv2.cvtColor(bars, cv2.COLOR_RGB2BGR))
        print(f"Saved source pattern: {args.save_png}")


def main():
    parser = argparse.ArgumentParser(
        description="NTSC Composite Video Simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python main.py encode input.mp4 -o signal.wav
  python main.py decode signal.wav -o output.mp4
  python main.py roundtrip input.mp4 -o output.mp4
  python main.py roundtrip input.mp4 -o output.mp4 --telecine
  python main.py image photo.png -o ntsc_photo.png
  python main.py colorbars -o bars.wav
        """)

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # encode
    p_enc = subparsers.add_parser('encode', help='Encode video to composite WAV')
    p_enc.add_argument('input', help='Input video file')
    p_enc.add_argument('-o', '--output', default='signal.wav', help='Output WAV file')

    # decode
    p_dec = subparsers.add_parser('decode', help='Decode composite WAV to video')
    p_dec.add_argument('input', help='Input WAV file')
    p_dec.add_argument('-o', '--output', default='output.mp4', help='Output video file')
    p_dec.add_argument('--width', type=int, default=640, help='Output width')
    p_dec.add_argument('--height', type=int, default=480, help='Output height')

    # roundtrip
    p_rt = subparsers.add_parser('roundtrip', help='Video -> composite -> video')
    p_rt.add_argument('input', help='Input video file')
    p_rt.add_argument('-o', '--output', default='output.mp4', help='Output video file')
    p_rt.add_argument('--width', type=int, default=640, help='Output width')
    p_rt.add_argument('--height', type=int, default=480, help='Output height')
    p_rt.add_argument('--telecine', action='store_true',
                      help='Simulate 3:2 pulldown telecine (480i, 4 film frames -> 5 NTSC frames)')

    # image
    p_img = subparsers.add_parser('image', help='Roundtrip a single image through NTSC')
    p_img.add_argument('input', help='Input image file (PNG, JPG, etc.)')
    p_img.add_argument('-o', '--output', default='output.png', help='Output image file')
    p_img.add_argument('--width', type=int, default=None, help='Output width (default: same as input)')
    p_img.add_argument('--height', type=int, default=None, help='Output height (default: same as input)')
    p_img.add_argument('--wav', default=None, help='Also export full-rate composite WAV')
    p_img.add_argument('--preview', default=None, help='Also export 48 kHz preview WAV')

    # colorbars
    p_cb = subparsers.add_parser('colorbars', help='Generate color bar test signal')
    p_cb.add_argument('-o', '--output', default='colorbars.wav', help='Output WAV file')
    p_cb.add_argument('--preview', default=None, help='Export 48 kHz preview WAV for audio editors')
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
