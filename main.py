"""CLI entry point for the NTSC Composite Video Simulator."""

import argparse
import sys

import numpy as np


def cmd_encode(args):
    """Encode a video file to a composite NTSC WAV signal."""
    import cv2
    from ntsc_simulator.encoder import encode_frame
    from ntsc_simulator.wav_io import export_wav
    from ntsc_simulator.constants import TOTAL_LINES, SAMPLES_PER_LINE

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"Error: Cannot open video file '{args.input}'")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Encoding {args.input} ({total_frames} frames) to composite signal...")

    all_signals = []
    frame_num = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # OpenCV reads BGR, convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        signal = encode_frame(frame_rgb, frame_number=frame_num)
        all_signals.append(signal)

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
    import cv2
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

    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, 29.97, (width, height))

    for i in range(num_frames):
        frame_signal = signal[i * samples_per_frame:(i + 1) * samples_per_frame]
        frame_rgb = decode_frame(frame_signal, frame_number=i,
                                 output_width=width, output_height=height)

        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

        if (i + 1) % 10 == 0:
            print(f"  Decoded frame {i + 1}/{num_frames}")

    out.release()
    print(f"Done: {args.output}")


def cmd_roundtrip(args):
    """Encode video to composite signal and decode back."""
    import cv2
    from ntsc_simulator.encoder import encode_frame
    from ntsc_simulator.decoder import decode_frame
    from ntsc_simulator.pipeline import SignalPipeline

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"Error: Cannot open video file '{args.input}'")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = args.width
    height = args.height

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, 29.97, (width, height))

    pipeline = SignalPipeline()
    # Future: add transforms to pipeline here

    print(f"Roundtrip: {args.input} -> composite -> {args.output}")
    print(f"  Output resolution: {width}x{height}")
    frame_num = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Encode
        signal = encode_frame(frame_rgb, frame_number=frame_num)

        # Apply pipeline transforms
        if len(pipeline) > 0:
            signal = pipeline.process(signal)

        # Decode
        result_rgb = decode_frame(signal, frame_number=frame_num,
                                  output_width=width, output_height=height)
        result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
        out.write(result_bgr)

        frame_num += 1
        if frame_num % 10 == 0:
            print(f"  Frame {frame_num}/{total_frames}")

    cap.release()
    out.release()
    print(f"Done: {args.output} ({frame_num} frames)")


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

    # Export signal WAV if requested
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

    # Export a 48 kHz preview WAV for inspection in audio editors
    if args.preview:
        print(f"Exporting 48 kHz preview WAV...")
        export_wav_preview(signal, args.preview)
        print(f"Preview: {args.preview}")

    # Also save the source pattern as PNG if requested
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
