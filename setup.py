from setuptools import setup

package_name = 'fbot_robot_learning'

setup(
 name=package_name,
 version='0.0.0',
 packages=[package_name,],
 data_files=[
     ('share/ament_index/resource_index/packages',
             ['resource/' + package_name]),
     ('share/' + package_name, ['package.xml']),
   ],
 install_requires=['setuptools'],
 zip_safe=True,
 maintainer='TODO',
 maintainer_email='TODO',
 description='TODO: Package description',
 license='TODO: License declaration',
 tests_require=['pytest'],
 entry_points={
     'console_scripts': [
             'data_collection = fbot_robot_learning.doris_gazebo_env:main'
     ],
   },
)