import matplotlib.pyplot as plt
import numpy as np
from rplidar import RPLidar
from matplotlib.animation import FuncAnimation

PORT_NAME = '/dev/ttyUSB0'
lidar = RPLidar(PORT_NAME, baudrate=115200)

fig = plt.figure()
ax = plt.subplot(111, projection='polar')
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)
points, = ax.plot([], [], 'b.', markersize=2)

def update(frame):
    scan = next(lidar.iter_scans())
    angles = []
    dists = []
    for quality, angle, dist in scan:
        if dist > 0:
            angles.append(np.deg2rad(angle))
            dists.append(dist/1000.0)  # metros
    points.set_data(angles, dists)
    return points,

ani = FuncAnimation(fig, update, interval=50, blit=True)

try:
    print("Radar en tiempo real (Ctrl+C para salir)")
    plt.show()
except KeyboardInterrupt:
    pass
finally:
    print("Cerrando Lidar...")
    lidar.stop()
    lidar.stop_motor()
    lidar.disconnect()