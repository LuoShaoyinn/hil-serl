import cv2
import time
import numpy as np
from rs_capture import RSCapture  # assuming your class is saved in rscapture.py

def main():
    # 1. Discover available devices
    cam = RSCapture(name="Camera", serial_number="fuck", depth=True)  # temporary init to query devices
    serials = cam.get_device_serial_numbers()
    cam.close()

    if not serials:
        print("❌ No Intel RealSense devices detected.")
        return
    else:
        print(f"✅ Found devices: {serials}")

    # 2. Select the first device
    serial = serials[1]
    print(f"🔧 Using device serial: {serial}")

    # 3. Initialize camera (color + optional depth)
    rs_cam = RSCapture(
        name="MainCam",
        serial_number=serial,
        dim=(640, 480),
        fps=30,
        depth=True,        # enable depth stream if available
        exposure=20000     # adjust manually if lighting is bright/dim
    )

    print("🎥 Starting capture. Press 'q' to quit.")

    # 4. Main loop
    try:
        while True:
            ret, frame = rs_cam.read()
            if not ret:
                print("⚠️ Frame not available.")
                continue

            print(frame.shape)

            # If depth is enabled, last channel contains depth
            if frame.shape[2] == 4:  # BGR + depth
                color_img = frame[:, :, :3].astype(np.uint8)
                depth_img = frame[:, :, 3]

                # Normalize depth for display
                depth_vis = cv2.convertScaleAbs(depth_img, alpha=0.03)
                depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
            
                print(color_img[0][0])
                print("----------------")

                cv2.imshow("Color", color_img)
                cv2.imshow("Depth", depth_vis)
            else:
                cv2.imshow("Color", frame)

            # Exit condition
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            

    except KeyboardInterrupt:
        print("⏹ Interrupted by user.")
    finally:
        print("🧹 Releasing resources...")
        rs_cam.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

