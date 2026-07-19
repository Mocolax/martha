from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'martha'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Instalar todos los launch files automáticamente
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*.xacro')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.world')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mocolax',
    maintainer_email='mocolax@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'cmd_vel_to_twist_stamped = martha.cmd_vel_to_twist_stamped:main',
            'odom_tf_broadcaster = martha.odom_tf_broadcaster:main',
            'imu_serial_viewer = martha.imu_serial_viewer:main',
        ],
    },
)
