# Mapas de ocupación

Los archivos `worlds/*.world` describen escenarios físicos de Gazebo; no son
mapas que AMCL o un map server puedan cargar en el robot real.

Guarda aquí los pares de mapa (`.yaml` + `.pgm`) o los pose graphs serializados
por SLAM Toolbox. Un mapa real solo es reutilizable si corresponde al entorno
físico y a la calibración vigente del LiDAR.
