from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'pick_drop_nav'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        (os.path.join('share', package_name, 'rviz'), glob(os.path.join('rviz', '*.rviz'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='eduardohufg',
    maintainer_email='eduardochavezmartin10@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mission_coordinator = pick_drop_nav.main_controller:main',
            'bug0 = pick_drop_nav.bug0:main',
            'bug2 = pick_drop_nav.bug2:main',
            'center_and_approach = pick_drop_nav.center_and_approach:main',
            'localisation = pick_drop_nav.localisation:main',
            'center2 = pick_drop_nav.center2:main',
        ],
    },
)
