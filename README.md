# Martha: simulación, hardware y navegación PPO

Martha usa el mismo contrato ROS 2 en Gazebo y en el robot físico. El backend
se selecciona en el launch; teleoperación, RViz, SLAM y la política PPO no
cambian de tópicos.

| Interfaz | Simulación | Hardware |
|---|---|---|
| Comando | `/cmd_vel` (`Twist`) | `/cmd_vel` (`Twist`) |
| LiDAR | `/scan` | `/scan` |
| IMU | `/imu/data` | `/imu/data` |
| Odometría para consumidores | `/odometry/filtered` | `/odometry/filtered` |
| Meta | `/goal_pose` | `/goal_pose` |
| TF dinámico | `odom -> base_link` por EKF | `odom -> base_link` por EKF |
| Reloj | simulado | sistema |

SLAM Toolbox es la única autoridad de `map -> odom` cuando `mapping:=true`.
El controlador y el firmware entregan las mediciones de ruedas; el EKF es la
única autoridad dinámica de `odom -> base_link`.

## Construcción

Requiere Ubuntu 22.04 y ROS 2 Humble con Gazebo Classic. Desde el workspace:

```bash
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
python3 -m pip install -r src/martha/requirements-ppo.txt
colcon build --packages-select martha --symlink-install
source install/setup.bash
```

## Simulación

Arranque común sin SLAM:

```bash
ros2 launch martha bringup.launch.py mode:=sim mapping:=false
```

Los launches históricos `gazebo.launch.py`, `gazebo_rviz.launch.py` y
`gz_rviz_map.launch.py` se conservan como aliases del mismo pipeline canónico;
ya no crean una segunda fuente de TF u odometría.

Teleoperación, en otra terminal con el workspace cargado:

```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard \
  --ros-args -r cmd_vel:=/cmd_vel
```

Mapeo y RViz:

```bash
ros2 launch martha bringup.launch.py mode:=sim mapping:=true
```

Para guardar el mapa construido:

```bash
ros2 service call /slam_toolbox/save_map slam_toolbox/srv/SaveMap \
  "{name: {data: '/ruta/absoluta/mapa_martha'}}"
```

## Hardware

Antes del launch deben cumplirse estas condiciones:

1. La ESP32 ejecuta el firmware de `arduino/` y aparece en un puerto estable,
   preferiblemente `/dev/serial/by-id/...`.
2. El driver del LiDAR físico ya publica un `LaserScan` casi completo de 360°
   en `/scan`, con timestamps válidos y un `frame_id` conectado por TF a
   `base_link`.
3. La polaridad de motores y encoders, el radio de rueda y la geometría
   mecanum se comprobaron físicamente con las ruedas levantadas.

El modelo exacto del LiDAR no está declarado en este repositorio; por eso su
driver se inicia aparte. Después:

```bash
ros2 launch martha bringup.launch.py \
  mode:=hardware \
  port:=/dev/serial/by-id/PUERTO_DE_LA_ESP32 \
  mapping:=true
```

El mismo comando de `teleop_twist_keyboard` usado en simulación mueve el
hardware. Comprobaciones mínimas:

```bash
ros2 topic hz /scan
ros2 topic hz /odometry/filtered
ros2 topic echo --once /hardware/motor_fault
ros2 run tf2_ros tf2_echo odom base_link
ros2 run tf2_ros tf2_echo base_link lidar
```

El puente limita los comandos a `vx=0.35 m/s`, `vy=0.35 m/s` y
`wz=0.80 rad/s`; rechaza valores no finitos, vigila la telemetría serial y
publica el fault latched `/hardware/motor_fault`. Para rearmar una protección:

```bash
ros2 service call \
  /cmd_vel_serial_bridge/reset_motor_protection \
  std_srvs/srv/Trigger "{}"
ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist "{}"
```

El segundo comando cero debe enviarse después del reconocimiento de la ESP32.

## Entrenamiento PPO en Gazebo

Los archivos `worlds/mundo_1.world` a `mundo_9.world` son escenarios SDF, no
mapas de ocupación. `MarthaEnv` intercambia sus obstáculos, toma inicios y
metas aleatorias dentro de espacio libre conectado, teletransporta el robot y
realinea el EKF en cada episodio.

La observación de 45 valores contiene 36 sectores LiDAR, distancia/rumbo a la
meta, velocidad y acción previa. No expone ground truth a la política; Gazebo
ground truth solo se usa para recompensa, colisión y métricas. La recompensa
combina avance geodésico, penalización por paso, esfuerzo, cambios bruscos,
proximidad, colisión y llegada.

Primero inicia el simulador sin RViz ni SLAM:

```bash
ros2 launch martha simulation.launch.py gui:=false
```

En otra terminal:

```bash
cd ~/ros2_ws
source install/setup.bash
ros2 run martha ppo_train \
  --episodes 2000 \
  --max-steps 300 \
  --map-mode random \
  --device auto
```

Los resultados quedan por defecto en `./ppo_runs/<run>/`: `metrics.csv`,
`last_model.pt` y `best_model.pt`. El mejor modelo se elige por tasa de éxito,
SPL, menor tasa de colisión y, al final, recompensa. Para continuar:

```bash
ros2 run martha ppo_train \
  --resume /ruta/ppo_runs/run/last_model.pt \
  --episodes 4000
```

Evaluación determinista en los nueve escenarios:

```bash
ros2 run martha ppo_evaluate \
  --checkpoint /ruta/ppo_runs/run/best_model.pt \
  --episodes-per-map 10 \
  --csv /ruta/resultados.csv
```

## Ejecutar una política entrenada

El mismo launch sirve para ambos backends y la política permanece detenida
hasta recibir una meta desde la herramienta **2D Goal Pose** de RViz:

```bash
# Gazebo
ros2 launch martha ppo_navigation.launch.py \
  mode:=sim checkpoint:=/ruta/best_model.pt mapping:=true

# Robot físico; el driver de /scan debe estar activo antes
ros2 launch martha ppo_navigation.launch.py \
  mode:=hardware \
  port:=/dev/serial/by-id/PUERTO_DE_LA_ESP32 \
  checkpoint:=/ruta/best_model.pt \
  mapping:=true
```

Sensores obsoletos, salida inválida, obstáculo dentro de la huella o un fault
del motor detienen y enclavan la política. Una meta nueva no borra el fault.
Después de resolver su causa:

```bash
ros2 service call /ppo_policy/rearm std_srvs/srv/Trigger "{}"
```

Luego se debe publicar una meta nueva.

No ejecutes `teleop_twist_keyboard` y `ppo_policy` al mismo tiempo: ambos son
productores directos de `/cmd_vel` y este proyecto todavía no incluye un mux
de prioridad. Usa teleoperación con `bringup.launch.py` o autonomía con
`ppo_navigation.launch.py`, de forma mutuamente exclusiva.

El entorno también permite entrenamiento/evaluación física, pero está
bloqueado por defecto. Requiere `--backend hardware --allow-hardware --goal X
Y`, operador presente, parada de emergencia y confirmación manual antes de
cada episodio. Para aprender normalmente se debe entrenar en Gazebo y usar en
hardware únicamente el nodo de inferencia.

## Validación

Pruebas puras y contractuales:

```bash
python3 -m pytest -q
```

Con el launch canónico de simulación ya activo, la prueba live de reset,
`SetPose`, sensores y un step se habilita explícitamente:

```bash
MARTHA_GAZEBO_SMOKE=1 python3 -m pytest -q -s \
  test/test_gazebo_env_smoke.py
```

Que el proyecto compile o pase en Gazebo no valida por sí solo la polaridad,
escala de encoders, deriva de IMU ni posición real del LiDAR. Esas cuatro
comprobaciones se deben cerrar sobre el robot físico antes de navegación
autónoma.
