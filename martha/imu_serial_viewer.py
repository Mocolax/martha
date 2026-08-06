import argparse
import math
import os
import sys
import threading
import time
from dataclasses import dataclass


@dataclass
class ImuSample:
    accel_x: float
    accel_y: float
    accel_z: float
    gyro_x: float
    gyro_y: float
    gyro_z: float
    roll: float
    pitch: float
    yaw: float
    stationary: bool


def parse_imu_line(line):
    line = line.strip()
    if not line.startswith("imu,"):
        return None

    parts = line.split(",")
    if len(parts) != 11:
        return None

    try:
        return ImuSample(
            accel_x=float(parts[1]),
            accel_y=float(parts[2]),
            accel_z=float(parts[3]),
            gyro_x=float(parts[4]),
            gyro_y=float(parts[5]),
            gyro_z=float(parts[6]),
            roll=float(parts[7]),
            pitch=float(parts[8]),
            yaw=float(parts[9]),
            stationary=bool(int(parts[10])),
        )
    except ValueError:
        return None


def rotation_matrix(roll, pitch, yaw):
    cr = math.cos(roll)
    sr = math.sin(roll)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cy = math.cos(yaw)
    sy = math.sin(yaw)

    return (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )


def rotate_point(matrix, point):
    x, y, z = point
    return (
        matrix[0][0] * x + matrix[0][1] * y + matrix[0][2] * z,
        matrix[1][0] * x + matrix[1][1] * y + matrix[1][2] * z,
        matrix[2][0] * x + matrix[2][1] * y + matrix[2][2] * z,
    )


def set_axes_equal(ax):
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(-1.5, 1.5)
    ax.set_box_aspect((1, 1, 1))


class ImuViewer:
    def __init__(self):
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/martha-matplotlib")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        self.plt = plt
        self.Poly3DCollection = Poly3DCollection

        self.fig = plt.figure("Martha IMU viewer")
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.view_init(elev=25, azim=-45)
        set_axes_equal(self.ax)

        self.title = self.ax.set_title("Waiting for IMU data")
        self.body = None
        self.axis_lines = []
        self.text = self.ax.text2D(0.02, 0.96, "", transform=self.ax.transAxes)

        plt.ion()
        plt.show(block=False)

    def on_key_press(self, callback):
        self.fig.canvas.mpl_connect("key_press_event", callback)

    def draw(self, sample, roll_offset=0.0, pitch_offset=0.0, yaw_offset=0.0):
        if self.body is not None:
            self.body.remove()
            self.body = None

        for line in self.axis_lines:
            line.remove()
        self.axis_lines = []

        roll = sample.roll - roll_offset
        pitch = sample.pitch - pitch_offset
        yaw = sample.yaw - yaw_offset
        matrix = rotation_matrix(roll, pitch, yaw)

        sx, sy, sz = 0.8, 0.45, 0.12
        corners = [
            (-sx, -sy, -sz),
            (sx, -sy, -sz),
            (sx, sy, -sz),
            (-sx, sy, -sz),
            (-sx, -sy, sz),
            (sx, -sy, sz),
            (sx, sy, sz),
            (-sx, sy, sz),
        ]
        points = [rotate_point(matrix, point) for point in corners]
        faces = [
            [points[i] for i in face]
            for face in (
                (0, 1, 2, 3),
                (4, 5, 6, 7),
                (0, 1, 5, 4),
                (2, 3, 7, 6),
                (1, 2, 6, 5),
                (0, 3, 7, 4),
            )
        ]

        self.body = self.Poly3DCollection(
            faces,
            facecolors=(0.1, 0.45, 0.85, 0.35),
            edgecolors=(0.05, 0.12, 0.18, 0.9),
            linewidths=1.0,
        )
        self.ax.add_collection3d(self.body)

        self._draw_axis(matrix, (1.2, 0, 0), "tab:red", "X")
        self._draw_axis(matrix, (0, 1.2, 0), "tab:green", "Y")
        self._draw_axis(matrix, (0, 0, 1.2), "tab:blue", "Z")

        self.title.set_text("Martha IMU orientation")
        self.text.set_text(
            "roll: {:+6.1f} deg\npitch: {:+6.1f} deg\nyaw: {:+6.1f} deg\n"
            "accel: {:+.2f}, {:+.2f}, {:+.2f} m/s2\n"
            "gyro: {:+.3f}, {:+.3f}, {:+.3f} rad/s\n"
            "stationary: {}".format(
                math.degrees(sample.roll),
                math.degrees(sample.pitch),
                math.degrees(sample.yaw),
                sample.accel_x,
                sample.accel_y,
                sample.accel_z,
                sample.gyro_x,
                sample.gyro_y,
                sample.gyro_z,
                "yes" if sample.stationary else "no",
            )
        )

        self.plt.pause(0.001)

    def _draw_axis(self, matrix, vector, color, label):
        end = rotate_point(matrix, vector)
        line = self.ax.plot(
            [0, end[0]],
            [0, end[1]],
            [0, end[2]],
            color=color,
            linewidth=3,
        )[0]
        text = self.ax.text(end[0], end[1], end[2], label, color=color)
        self.axis_lines.extend((line, text))


class SharedImuState:
    def __init__(self):
        self.lock = threading.Lock()
        self.latest_sample = None
        self.lines_received = 0
        self.invalid_lines = 0
        self.frames_drawn = 0
        self.reader_error = None
        self.stop_event = threading.Event()

    def set_sample(self, sample):
        with self.lock:
            self.latest_sample = sample
            self.lines_received += 1

    def add_invalid_line(self):
        with self.lock:
            self.invalid_lines += 1

    def set_reader_error(self, message):
        with self.lock:
            self.reader_error = message
            self.stop_event.set()

    def snapshot(self):
        with self.lock:
            return (
                self.latest_sample,
                self.lines_received,
                self.invalid_lines,
                self.frames_drawn,
                self.reader_error,
            )

    def add_frame(self):
        with self.lock:
            self.frames_drawn += 1


def open_serial(port, baud):
    try:
        import serial
    except ImportError as exc:
        raise SystemExit("Falta pyserial. Instala python3-serial.") from exc

    return serial.Serial(port, baudrate=baud, timeout=0.05)


def read_lines_to_state(source, state, echo_invalid=True, stop_on_eof=False):
    while not state.stop_event.is_set():
        try:
            raw_line = source.readline()
        except Exception as exc:
            state.set_reader_error(str(exc))
            return

        if not raw_line:
            if stop_on_eof:
                state.stop_event.set()
                return
            continue

        if isinstance(raw_line, bytes):
            line = raw_line.decode("utf-8", errors="replace").strip()
        else:
            line = raw_line.strip()

        sample = parse_imu_line(line)
        if sample is None:
            state.add_invalid_line()
            if echo_invalid:
                print(line)
            continue

        state.set_sample(sample)


def print_diagnostics(state, previous_counts, now):
    _, lines_received, invalid_lines, frames_drawn, _ = state.snapshot()
    previous_time, previous_lines, previous_invalid, previous_frames = previous_counts
    dt = max(now - previous_time, 1e-6)

    serial_hz = (lines_received - previous_lines) / dt
    render_fps = (frames_drawn - previous_frames) / dt
    invalid_delta = invalid_lines - previous_invalid

    print(
        "viewer: serial={:.1f} Hz render={:.1f} FPS invalid={}".format(
            serial_hz,
            render_fps,
            invalid_delta,
        )
    )

    return now, lines_received, invalid_lines, frames_drawn


def main():
    parser = argparse.ArgumentParser(
        description="Visualiza la orientacion del IMU por serial."
    )
    parser.add_argument(
        "--port",
        default="/dev/ttyUSB0",
        help="Puerto serial del ESP32.",
    )
    parser.add_argument(
        "--baud",
        type=int,
        default=115200,
        help="Baudios del sketch Arduino.",
    )
    parser.add_argument(
        "--stdin",
        action="store_true",
        help="Lee datos desde stdin en vez de serial.",
    )
    parser.add_argument(
        "--render-rate",
        type=float,
        default=25.0,
        help="FPS maximos del visor.",
    )
    parser.add_argument(
        "--no-diagnostics",
        action="store_true",
        help="No imprime Hz/FPS por consola.",
    )
    args = parser.parse_args()

    if args.render_rate <= 0.0:
        raise SystemExit("--render-rate debe ser mayor que 0")

    viewer = ImuViewer()
    state = SharedImuState()
    offsets = {"roll": 0.0, "pitch": 0.0, "yaw": 0.0}

    def handle_key_press(event):
        latest_sample, _, _, _, _ = state.snapshot()
        if event.key == "r" and latest_sample is not None:
            offsets["roll"] = latest_sample.roll
            offsets["pitch"] = latest_sample.pitch
            offsets["yaw"] = latest_sample.yaw
            print("VIEWER_RESET")

    viewer.on_key_press(handle_key_press)

    serial_source = None
    if args.stdin:
        source = sys.stdin
    else:
        serial_source = open_serial(args.port, args.baud)
        time.sleep(2.0)
        serial_source.reset_input_buffer()
        source = serial_source

    reader = threading.Thread(
        target=read_lines_to_state,
        args=(source, state, True, args.stdin),
        daemon=True,
    )
    reader.start()

    render_period = 1.0 / args.render_rate
    next_render_time = time.monotonic()
    last_drawn_sample = None
    previous_counts = (time.monotonic(), 0, 0, 0)
    next_diagnostic_time = time.monotonic() + 1.0

    try:
        while not state.stop_event.is_set():
            now = time.monotonic()

            if now >= next_render_time:
                sample, _, _, _, reader_error = state.snapshot()
                if reader_error:
                    raise SystemExit(reader_error)

                if sample is not None and sample is not last_drawn_sample:
                    viewer.draw(
                        sample,
                        roll_offset=offsets["roll"],
                        pitch_offset=offsets["pitch"],
                        yaw_offset=offsets["yaw"],
                    )
                    state.add_frame()
                    last_drawn_sample = sample
                else:
                    viewer.plt.pause(0.001)

                next_render_time = now + render_period

            if not args.no_diagnostics and now >= next_diagnostic_time:
                previous_counts = print_diagnostics(state, previous_counts, now)
                next_diagnostic_time = now + 1.0

            time.sleep(0.001)
    except KeyboardInterrupt:
        state.stop_event.set()
    finally:
        state.stop_event.set()
        if serial_source is not None:
            serial_source.close()
        reader.join(timeout=1.0)


if __name__ == "__main__":
    main()
