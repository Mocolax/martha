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

Este es el launcher canónico. `simulation.launch.py` y `hardware.launch.py`
son los backends que compone internamente.

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
2. El RPLIDAR A2M8 está conectado y tiene disponible el enlace estable
   `/dev/rplidar`.
3. La polaridad de motores y encoders, el radio de rueda y la geometría
   mecanum se comprobaron físicamente con las ruedas levantadas.

En cada computadora anfitriona, instala una vez la regla udev incluida en
`rplidar_ros` y vuelve a conectar el LiDAR:

```bash
sudo cp ~/ros2_ws/src/rplidar_ros/scripts/rplidar.rules \
  /etc/udev/rules.d/99-rplidar.rules
sudo udevadm control --reload-rules
sudo udevadm trigger
ls -l /dev/rplidar
```

El launch de hardware inicia `rplidar_node` con el perfil A2M8: 115200
baudios, modo `Sensitivity`, compensación angular, 10 Hz, tópico `/scan` y
frame `lidar`. Arranque completo:

```bash
ros2 launch martha bringup.launch.py \
  mode:=hardware \
  port:=/dev/serial/by-id/PUERTO_DE_LA_ESP32 \
  lidar_port:=/dev/rplidar \
  mapping:=true
```

El puerto puede sobrescribirse con `lidar_port:=...`. Si la ESP32 también usa
un adaptador CP210x, utiliza rutas distintas de `/dev/serial/by-id`; el launch
rechaza ambos argumentos si terminan apuntando al mismo dispositivo. Para usar
un driver iniciado externamente, añade `start_lidar:=false`.

El URDF sitúa el plano del LiDAR al frente y centrado, aproximadamente a
`x=0.2325 m` de `base_link`. El montaje físico debe respetar esa posición; si
cambia, deben actualizarse juntos el URDF y el offset usado por la protección
de huella PPO.

El mismo comando de `teleop_twist_keyboard` usado en simulación mueve el
hardware. Comprobaciones mínimas:

```bash
ros2 topic hz /scan
ros2 topic echo --once /scan
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

Los escenarios canónicos son `four_rooms`, `hall`, `multi`, `roblab`, `room`
y `tube`. El entrenamiento genera un único mundo temporal con las seis arenas
en una cuadrícula 3×2, separadas 30 m, y lanza ocho robots namespaced
(`martha_0` a `martha_7`) dentro de un solo `gzserver`.

Los inicios y metas se editan únicamente en `config/training_points.yaml`.
Cada coordenada es local al `.world`; el coordinador aplica el desplazamiento
de su arena. `x` e `y` son obligatorios y `yaw: null` pide una orientación de
inicio aleatoria. Un mismo punto puede ser inicio o meta, pero no ambos para la
misma Martha. Antes de lanzar Gazebo se validan IDs, valores finitos, espacio
libre, conectividad, distancia mínima y ocho inicios simultáneos separados. No
hay generación automática: mientras una sección tenga menos de ocho puntos
válidos, `ppo_train` termina indicando el mapa problemático.

La observación contiene cuatro frames distribuidos durante un segundo de tiempo
ROS. Cada frame reúne 36 sectores LiDAR, distancia normalizada y seno/coseno del
rumbo hacia un waypoint local, además de velocidad odométrica `[Vx, Vy, W]`; el
vector completo tiene 168 valores. La distancia se codifica sin recorte como
`d / (d + 3 m)`. En Gazebo, el planificador privilegiado usa el mapa y la pose
real para actualizar un waypoint geodésico a 0.50 m; la política solo recibe
sus tres características relativas, no el mapa ni las coordenadas globales.
La llegada y las métricas siguen midiéndose contra la meta final. En el robot
real, un planificador global con localización debe producir el mismo waypoint.
La recompensa combina avance, penalización por paso, esfuerzo, cambios bruscos,
proximidad, colisión y llegada.

Cada robot publica contactos de su huella elevada en
`/martha_N/contacts` mediante `libgazebo_ros_bumper.so`. Los contactos con
suelo o con el propio modelo se ignoran; una pared u otra Martha finaliza el
episodio. Si chocan dos robots, ambos terminan. El LiDAR no decide colisiones:
se conserva para observación, recompensa de proximidad y validación del reset.

La cantidad original de puntos de `/scan` no cambia el tamaño del modelo: los
scans de densidad variable del A2M8 se agrupan angularmente por mínimo en los
mismos 36 sectores y se limitan al alcance común de 8 m.

El actor y el crítico tienen extractores multirrama independientes. La variable
`OBSERVATION_ENCODER_MODE` de `martha/PPO/network.py` selecciona `"present"`
(solo el frame más reciente) o `"history"` (los cuatro frames) como entrada de
las ramas de LiDAR, distancia, orientación y velocidad. El modo normal es
`"present"`; el entorno conserva el vector canónico de 168 valores, pero el
extractor ignora los tres frames antiguos y deja la temporalidad al LSTM. Las
ramas se fusionan en 384 unidades y luego el actor y el crítico usan su propia
LSTM de 128 unidades. Cambiar el modo exige entrenar un checkpoint nuevo.
Cada robot conserva memoria recurrente independiente; se borra al iniciar un
episodio, cambiar de meta, llegar o entrar en fault. PPO entrena fragmentos
ordenados de 32 pasos con máscaras de padding y de fin de episodio, sin barajar
transiciones temporales aisladas. Sus gradientes se recortan por separado. La
recompensa guardada en el
buffer se multiplica por `reward_scale` (`0.01` por defecto), pero
`episode_reward` conserva la recompensa original y `episode_scaled_reward`
registra el valor usado por PPO.

Los valores de entrenamiento estan reunidos al principio de
`martha/PPO/train.py`, en `TrainingDefaults`. Para cambiar permanentemente el
numero de robots, velocidad, episodios o hiperparametros, edita ese unico
bloque. El unico argumento de consola es `--resume`.

Las ocho Marthas comparten un escenario durante una ronda. Sus inicios no se
repiten; cada meta difiere del inicio propio. Al terminar por meta, contacto o
timeout, el robot se detiene, pierde su marcador y se teletransporta a una
plaza exclusiva alrededor de `y=-43 m`. Allí espera sin bloquear la arena.
Cuando termina la última Martha activa comienza la ronda siguiente. Cada ciclo
baraja los seis escenarios con la semilla del entrenamiento y usa cada uno una
sola vez. `map_index` fija todas las rondas a un único escenario. `episodes`
sigue contando episodios individuales, por lo que la última ronda puede activar
menos de ocho robots.

La recompensa se configura en `martha/PPO/reward.py`, dentro de
`RewardConfig`. Sigue la función completa de Jestel et al.,
[*Obtaining Robust Control and Navigation Policies for Multi-Robot Navigation
via Deep Reinforcement Learning*](https://arxiv.org/pdf/2209.03097). Para un
step no terminal se calcula:

```text
delta_d = distancia_euclidea_anterior - distancia_euclidea_actual
distancia = delta_d * (0.002 si delta_d < 0, de otro modo 0.01)
orientacion = (1 - 2 * abs(angulo_meta) / pi) *
              0.001, solo si Martha apunta a ±90° de la meta; si no, 0
record = 0.05 * (mejor_distancia_del_episodio - distancia_actual)
         solo cuando establece un nuevo récord
laser = -0.01 * (0.65 - minimo_laser), solo bajo 0.65 m
zigzag = -0.01, si hay más de 3 reversos directos izquierda-derecha
          durante las últimas 10 acciones
reward = costo_temporal + distancia + orientacion + record + laser + zigzag
```

La distancia para recompensa es euclídea; la distancia geodésica de Gazebo se
mantiene únicamente para SPL y las métricas de ruta. Cada transición no terminal
aplica un costo temporal de `-0.0002`. Los finales son exclusivos: meta `+1.0`,
colisión/motor fault/fuera del mapa `-0.75` y timeout `-0.5`.
Una colisión conserva prioridad sobre una señal simultánea de meta. El zigzag
usa la velocidad angular aplicada: izquierda por encima de `+0.2 rad/s`,
derecha por debajo de `-0.2 rad/s`; una acción recta corta la secuencia. Esos
tres parámetros no fueron publicados por el paper y son la adaptación acordada
para Martha. Los checkpoints nuevos guardan `reward_config`; al reanudar o
evaluar uno antiguo se usan los valores actuales y aparece una advertencia.

El entrenamiento siempre administra su propia instancia de Gazebo. No
inicies `simulation.launch.py` antes de `ppo_train`, ni siquiera cuando
`num_envs=1`. El perfil PPO usa por defecto un paso físico de 2 ms, 180 rayos
LiDAR reducidos a los mismos 36 sectores y cinemática planar. La simulación
normal conserva por defecto el modelo detallado de 48 rodillos por robot, el
paso de 1 ms y 360 rayos. Por tanto, el contrato de observación y los tópicos
ROS son comunes, pero el backend de entrenamiento evita simular articulaciones
que no aportan información a la política.

Si el factor supera la capacidad del equipo, Gazebo simplemente no alcanzara
la velocidad solicitada y pueden aparecer jitter, colas de mensajes o
`sensor_timeout` en las metricas. En ese caso reduce el factor; aumentar el
valor mas alla del rendimiento sostenible no acelera el aprendizaje.

Para entrenar:

```bash
cd ~/ros2_ws
source install/setup.bash
ros2 run martha ppo_train
```

`TrainingDefaults.num_envs=4` significa cantidad de robots, no cantidad de
servidores. En cada paso vectorizado el coordinador publica todas las acciones,
reanuda la física una vez, recibe sensores/contactos frescos y la pausa una
vez. Cuando uno termina, su plaza se recicla en el mismo mapa; el intervalo de
asentamiento se registra como transición de parada para los robots activos y
queda enmascarado del loss del actor, aunque sí entrena al crítico. El único log
de simulación queda en `<run>/gazebo.log`. Con `gazebo_gui=True`, una ventana
muestra las seis arenas y la flota.
El arranque es secuencial para evitar carreras de Gazebo y puede tardar varios
minutos con los modelos detallados; `gazebo_startup_timeout` vale 240 s.

Las evaluaciones periódicas esperan el final de un bloque. Los demás robots quedan
aparcados y `martha_0` evalúa la política dentro del mismo `gzserver`. Cambia
`eval_every`, `eval_map_count`, `num_envs`, `map_batch_episodes` y
`sim_speed_factor` en
`TrainingDefaults`; `eval_every=0` desactiva la evaluación periódica. En
entrenamiento no abras otro `simulation.launch.py`: el coordinador es dueño de
la única simulación.

Cada intento deja de asignar episodios 30 minutos antes del límite de 24 horas,
drena los episodios activos y guarda un checkpoint. Si alcanza el límite duro,
los episodios restantes se cierran como truncamientos con bootstrap. El CSV
incluye tiempo acumulado de física, reset, PPO, evaluación y checkpoint, además
de `training_steps_per_second`.

Cuando no haya un entrenamiento activo, compara 1, 2 y 4 robots con una carga
fija y sin actualizar la política:

```bash
ros2 run martha ppo_benchmark
```

El benchmark se niega a competir por recursos si detecta `ppo_train` activo.
Para validar ademas el reciclaje de un robot mientras sus pares siguen activos:

```bash
ros2 run martha ppo_benchmark --robot-counts 4 --steps 40 --recycle-smoke
```

Los resultados quedan por defecto en
`~/ros2_ws/src/martha/martha/PPO/ppo_runs/<run>/`: `metrics.csv`,
`last_model.pt`, `best_model.pt`, `learning_report.png`,
`ppo_diagnostics.png` y `training_summary.txt`. Los tres informes se generan
automáticamente al finalizar correctamente. El mejor modelo se elige por tasa de éxito,
SPL, menor tasa de colisión y, al final, recompensa. La ubicación se configura
con `TrainingDefaults.runs_dir`. `metrics.csv` incluye `approx_kl`,
`clip_fraction`, `explained_variance`, `policy_std`, `actor_inactive_relu` y
`critic_inactive_relu` para detectar inestabilidad antes de que la politica
deje de responder a sus observaciones. Para continuar un checkpoint se debe
conservar el mismo `reward_scale`:

```bash
ros2 run martha ppo_train \
  --resume /ruta/ppo_runs/run/last_model.pt
```

Antes de reanudar, establece en `TrainingDefaults.episodes` el episodio final
deseado. El CSV del run debe usar el esquema actual. La arquitectura LSTM usa
el contrato de política versión 5; los checkpoints feed-forward anteriores se
rechazan explícitamente y requieren comenzar un run nuevo o una transferencia
de pesos diseñada por separado.

Si el entrenamiento fue interrumpido o quieres volver a graficar un run
anterior, el siguiente comando usa el run modificado más recientemente:

```bash
ros2 run martha ppo_plot
```

También acepta directamente una carpeta de run o su `metrics.csv`:

```bash
ros2 run martha ppo_plot /ruta/ppo_runs/ppo_martha_YYYYMMDD_HHMMSS
```

La ventana de la media móvil se cambia en `REPORT_WINDOW`, al principio de
`martha/PPO/analytics.py`. Los runs nuevos guardan además la contribución
acumulada de cada término de `RewardConfig`, para identificar si dominan el
progreso, el costo temporal, el zigzag, el clearance o las recompensas terminales.

La exploración PPO también tiene tres fases configurables en `TrainingDefaults`:
`entropy_coef` se mantiene durante `entropy_exploration_fraction` (60 % por
defecto) de los episodios, disminuye linealmente hasta cero durante
`entropy_decay_fraction` (30 %), y queda en cero durante el porcentaje
restante. No modifica la recompensa, arquitectura ni learning rate; solo deja
de incentivar entropía para que PPO pueda consolidar una menor desviación
estándar cuando sus ventajas lo indiquen. El estado de la fase depende del
progreso relativo de los episodios, por lo que la misma configuración conserva
la fase al reanudar un checkpoint.

Evaluación determinista en los seis escenarios (requiere la simulación normal
ya iniciada en otra terminal):

```bash
ros2 launch martha simulation.launch.py gui:=false \
  sim_speed_factor:=5.0 physics_step_size:=0.002 \
  training_kinematic:=true lidar_samples:=180 lidar_visualize:=false

# En otra terminal, con el mismo workspace cargado:
ros2 run martha ppo_evaluate \
  --checkpoint /ruta/ppo_runs/run/best_model.pt
```

Configura backend, mapas, repeticiones, meta y CSV en `EvaluationDefaults`, al
principio de `martha/PPO/evaluate.py`.

## Ejecutar una política entrenada

El mismo launch sirve para ambos backends. Este checkpoint es un controlador
local: `/goal_pose` debe contener un waypoint aproximadamente 0.50 m por delante
sobre una ruta global, no la meta final seleccionada directamente con
**2D Goal Pose**. El planificador debe actualizarlo antes de que quede a 0.25 m;
las actualizaciones durante una navegación conservan la memoria LSTM.

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

El entorno también permite entrenamiento y evaluación física. Configura
`backend="hardware"` y `goal=(X, Y)` en el bloque de valores correspondiente.
Cada episodio exige una confirmación interactiva, un operador presente y una
parada de emergencia. Para aprender normalmente se recomienda Gazebo y usar en
hardware el nodo de inferencia.

## Validación

Pruebas puras y contractuales:

```bash
python3 -m pytest -q
```

La prueba live administra su propio Gazebo y comprueba un `gzserver`, las seis
arenas, ocho Marthas y sus ocho tópicos de contacto independientes:

```bash
MARTHA_GAZEBO_SMOKE=1 python3 -m pytest -q -s \
  test/test_gazebo_env_smoke.py
```

Que el proyecto compile o pase en Gazebo no valida por sí solo la polaridad,
escala de encoders, deriva de IMU ni posición real del LiDAR. Esas cuatro
comprobaciones se deben cerrar sobre el robot físico antes de navegación
autónoma.
