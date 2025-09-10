from distutils.core import setup

setup(
    name='xromotion',
    version='0.0.2',
    packages=['expressive_motion_generation'],
    package_dir={'': 'src'},
    maintainer='Marvin Wiebe',
    maintainer_email='mwiebe@techfak.uni-bielefeld.de',
    description='The xRomotion framework uses MoveIt to generate motion trajectories and allows the user to amplify them with expressive information. Prerecorded motion in the form of animations can also be executed.',
    install_requires=['setuptools'],
    zip_safe=True,
    license='TODO',
    data_files=[
        ('share/ament_index/resource_index/packages',
        ['resource/xromotion']),
        ('share/xromotion', ['package.xml']),
    ],
    entry_points={},
)
