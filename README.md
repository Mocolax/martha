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
`x=0.255 m` de `base_link`. El montaje físico debe respetar esa posición; si
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

Los archivos `worlds/mundo_1.world` a `mundo_9.world` son escenarios SDF, no
mapas de ocupación. `MarthaEnv` intercambia sus obstáculos, toma inicios y
metas aleatorias dentro de espacio libre conectado, teletransporta el robot y
realinea el EKF en cada episodio.

La observación contiene cuatro frames distribuidos durante un segundo de tiempo
ROS. Cada frame reúne 36 sectores LiDAR, distancia normalizada, seno/coseno del
rumbo y velocidad odométrica `[Vx, Vy, W]`; el vector completo tiene 168
valores. No expone ground truth a la política; Gazebo ground truth solo se usa
para recompensa, colisión y métricas. La recompensa combina avance geodésico,
penalización por paso, esfuerzo, cambios bruscos, proximidad, colisión y llegada.

La cantidad original de puntos de `/scan` no cambia el tamaño del modelo: los
scans de densidad variable del A2M8 se agrupan angularmente por mínimo en los
mismos 36 sectores y se limitan al alcance común de 8 m.

El actor y el crítico tienen extractores multirrama independientes. Cada uno
procesa los cuatro LiDAR con dos convoluciones 1D, codifica distancia,
orientación y velocidad en ramas densas, y fusiona las representaciones en 384
unidades. Sus gradientes se recortan por separado. La recompensa guardada en el
buffer se multiplica por `reward_scale` (`0.01` por defecto), pero
`episode_reward` conserva la recompensa original y `episode_scaled_reward`
registra el valor usado por PPO.

Los valores de entrenamiento estan reunidos al principio de
`martha/PPO/train.py`, en `TrainingDefaults`. Para cambiar permanentemente el
numero de Gazebos, velocidad, episodios o hiperparametros, edita ese unico
bloque. El unico argumento de consola es `--resume`.

Durante entrenamiento Gazebo, `episodes_per_map` (20 por defecto) mantiene el
mismo mundo durante ese número de episodios globales antes de rotar. Cada ronda
baraja los nueve mundos con la semilla del entrenamiento y los usa una sola vez;
la secuencia se reconstruye por número de episodio al reanudar. `map_index`, si
se configura, conserva su prioridad y desactiva la rotación.

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
reward = distancia + orientacion + record + laser + zigzag
```

La distancia para recompensa es euclídea; la distancia geodésica de Gazebo se
mantiene únicamente para SPL y las métricas de ruta. Los finales son exclusivos:
meta `+1.0`, colisión/motor fault/fuera del mapa `-0.75` y timeout `0.0`.
Una colisión conserva prioridad sobre una señal simultánea de meta. El zigzag
usa la velocidad angular aplicada: izquierda por encima de `+0.2 rad/s`,
derecha por debajo de `-0.2 rad/s`; una acción recta corta la secuencia. Esos
tres parámetros no fueron publicados por el paper y son la adaptación acordada
para Martha. Los checkpoints nuevos guardan `reward_config`; al reanudar o
evaluar uno antiguo se usan los valores actuales y aparece una advertencia.

El entrenamiento siempre administra sus propias instancias de Gazebo. No
inicies `simulation.launch.py` antes de `ppo_train`, ni siquiera cuando
`num_envs=1`. El factor `TrainingDefaults.sim_speed_factor` acepta valores
mayores que cero hasta `20.0`. Se recomienda probar primero `2.0` y luego
`4.0`. No cambia el paso fisico de 1 ms ni el LiDAR de 10 Hz en tiempo
simulado: solo intenta ejecutar esos segundos simulados mas rapido que el reloj
de pared.

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

Con `num_envs=1`, `ppo_train` crea un Gazebo. Con valores mayores crea todas las
instancias aisladas necesarias. Las evaluaciones periodicas se hacen, por
defecto, cada 50 episodios sobre 3 mundos fijos, con 1 episodio y como maximo
400 steps por mundo. Cambia `eval_every`, `eval_map_count`, `num_envs` y
`sim_speed_factor` en `TrainingDefaults`; `eval_every=0` desactiva la
evaluacion periodica.

Para recolectar experiencia en paralelo, el entrenador crea y cierra una
instancia aislada de Gazebo por worker, cada una con su propio dominio ROS y
puerto maestro. Configura `TrainingDefaults.num_envs` y
`TrainingDefaults.sim_speed_factor` antes de ejecutar `ppo_train`.

`TrainingDefaults.gazebo_gui=True` hace visible solamente el worker 0; los
demás continúan ejecutándose sin interfaz gráfica. La ventana muestra un único
episodio activo representativo, no una composición de todos los workers. En
entrenamiento nunca se debe abrir otro `simulation.launch.py`: la ventana
visible también la crea y administra `ppo_train`. Para maximizar velocidad usa
`gazebo_gui=False`.

Empieza con `num_envs=2`; cada instancia adicional
consume memoria y CPU, por lo que un numero excesivo puede ser mas lento por
contencion o swap. `sim_speed_factor` aplica a cada worker paralelo. La red y
el optimizador siguen siendo un unico PPO (en CPU o GPU): solo se paraleliza la
recoleccion de experiencia. Los steps y los resets de los workers que terminan
en el mismo lote se despachan concurrentemente; cada reset sigue aislado en su
propio dominio ROS y puerto Gazebo. Los logs de Gazebo quedan en
`<run>/parallel_logs/`. Si los dominios o puertos por defecto estan ocupados,
edita `ros_domain_base` y `gazebo_port_base`.

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
deseado. El CSV del run debe usar el esquema actual.

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
progreso, el costo por step, el clearance o las recompensas terminales.

La exploración PPO también tiene tres fases configurables en `TrainingDefaults`:
`entropy_coef` se mantiene durante `entropy_exploration_fraction` (60 % por
defecto) de los episodios, disminuye linealmente hasta cero durante
`entropy_decay_fraction` (30 %), y queda en cero durante el porcentaje
restante. No modifica la recompensa, arquitectura ni learning rate; solo deja
de incentivar entropía para que PPO pueda consolidar una menor desviación
estándar cuando sus ventajas lo indiquen. El estado de la fase depende del
progreso relativo de los episodios, por lo que la misma configuración conserva
la fase al reanudar un checkpoint.

Evaluación determinista en los nueve escenarios:

```bash
ros2 run martha ppo_evaluate \
  --checkpoint /ruta/ppo_runs/run/best_model.pt
```

Configura backend, mapas, repeticiones, meta y CSV en `EvaluationDefaults`, al
principio de `martha/PPO/evaluate.py`.

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
