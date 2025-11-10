#!/usr/bin/env python3
"""
Isolated RealSense reader:
- Lists devices and selects one (or uses --serial).
- Streams color (+ optional depth) with alignment to color.
- Sets manual exposure and disables auto-exposure if requested.
- Limits internal frames queue size to reduce stalls.
- Uses wait_for_frames(timeout_ms=...) to avoid indefinite blocking.
- Shows FPS and provides clean shutdown on Ctrl+C or 'q'.
"""

import argparse
import time
import sys
import signal
from collections import deque

import numpy as np
import cv2
import pyrealsense2 as rs


def list_serials():
    ctx = rs.context()
    return [d.get_info(rs.camera_info.serial_number) for d in ctx.devices]


def apply_sensor_options(device, exposure=None, disable_auto_exposure=True, frames_queue_size=2):
    # A device can have multiple sensors (RGB, depth, IR). Apply options where supported.
    for s in device.query_sensors():
        # Queue size
        if s.supports(rs.option.frames_queue_size):
            try:
                s.set_option(rs.option.frames_queue_size, float(frames_queue_size))
            except Exception as e:
                print(f"[warn] frames_queue_size not set on {s.get_info(rs.camera_info.name)}: {e}")

        # Auto exposure
        if disable_auto_exposure and s.supports(rs.option.enable_auto_exposure):
            try:
                s.set_option(rs.option.enable_auto_exposure, 0.0)
            except Exception as e:
                print(f"[warn] enable_auto_exposure not set: {e}")

        # Manual exposure (only relevant for color sensor typically)
        if exposure is not None and s.supports(rs.option.exposure):
            try:
                s.set_option(rs.option.exposure, float(exposure))
            except Exception as e:
                print(f"[warn] exposure not set: {e}")


def visualize_depth(depth_frame):
    """Convert a 16-bit depth frame to a colorized 8-bit image for display."""
    depth_np = np.asanyarray(depth_frame.get_data())
    # Scale for visualization (tune alpha for your typical depth range)
    depth_8u = cv2.convertScaleAbs(depth_np, alpha=0.03)
    depth_color = cv2.applyColorMap(depth_8u, cv2.COLORMAP_JET)
    return depth_color


def main():
    parser = argparse.ArgumentParser(description="Isolated Intel RealSense color(+depth) reader")
    parser.add_argument("--serial", type=str, default=None, help="Device serial number to use")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--depth", type=int, default=1, help="Enable depth stream (1=yes, 0=no)")
    parser.add_argument("--exposure", type=int, default=20000, help="Manual exposure (µs) for color")
    parser.add_argument("--no_disable_auto_exposure", action="store_true", help="Do NOT disable auto-exposure")
    parser.add_argument("--timeout_ms", type=int, default=1000, help="Frame wait timeout (ms)")
    args = parser.parse_args()

    # Device discovery
    serials = list_serials()
    if not serials:
        print("ERROR: No Intel RealSense devices detected.")
        sys.exit(1)

    if args.serial:
        if args.serial not in serials:
            print(f"ERROR: Serial {args.serial} not found. Available: {serials}")
            sys.exit(1)
        serial = args.serial
    else:
        serial = serials[0]
        print(f"[info] Using first available device: {serial}")

    # Pipeline & config
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    if args.depth:
        cfg.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)

    # Start streaming
    try:
        profile = pipe.start(cfg)
    except Exception as e:
        print(f"ERROR: Failed to start pipeline: {e}")
        sys.exit(1)

    device = profile.get_device()
    apply_sensor_options(
        device,
        exposure=args.exposure,
        disable_auto_exposure=(not args.no_disable_auto_exposure),
        frames_queue_size=2,
    )

    # Align depth to color (even if depth disabled, we keep object for simplicity)
    align_to = rs.stream.color
    align = rs.align(align_to)

    # FPS measurement
    times = deque(maxlen=60)
    last_print = time.time()

    # Graceful exit handling
    stop_flag = {"stop": False}
    def _handle_sigint(sig, frame):
        stop_flag["stop"] = True
    signal.signal(signal.SIGINT, _handle_sigint)

    print("[info] Streaming... Press 'q' in the window or Ctrl+C to quit.")

    try:
        while not stop_flag["stop"]:
            try:
                frames = pipe.wait_for_frames(timeout_ms=args.timeout_ms)
            except RuntimeError:
                print("[warn] Timeout waiting for frames; continuing...")
                continue

            aligned = align.process(frames)

            color_frame = aligned.get_color_frame()
            if not color_frame or not color_frame.is_video_frame():
                print("[warn] No valid color frame.")
                continue

            color_img = np.asanyarray(color_frame.get_data())

            if args.depth:
                depth_frame = aligned.get_depth_frame()
                if depth_frame and depth_frame.is_depth_frame():
                    depth_vis = visualize_depth(depth_frame)
                    # Stack displays side by side (resize to same size for safety)
                    if depth_vis.shape[:2] != color_img.shape[:2]:
                        depth_vis = cv2.resize(depth_vis, (color_img.shape[1], color_img.shape[0]))
                    disp = np.hstack([color_img, depth_vis])
                else:
                    disp = color_img
            else:
                disp = color_img
                print(disp[0][0])

            cv2.imshow("RealSense Color (+Depth)", disp)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            # FPS tracking
            now = time.time()
            times.append(now)
            if now - last_print >= 1.0 and len(times) > 1:
                fps = (len(times) - 1) / (times[-1] - times[0])
                print(f"[fps] ~{fps:.1f}")
                last_print = now

    finally:
        print("[info] Stopping pipeline and cleaning up...")
        pipe.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

