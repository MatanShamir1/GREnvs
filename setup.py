from setuptools import setup, find_packages
import versioneer

setup(
    name='gr_libs',  # Replace with your package name
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=find_packages(),
    # entry_points={
    #     "gymnasium.envs": ["__root__ = gr_libs.minigrid_scripts.__init__:register_minigrid_envs"]
    # },
    python_requires=">=3.11",
    install_requires=[
        'minigrid',
        #'highway-env==1.9.1',
		'highway-env',
        'tensorboardX',
        'torchvision',
        'panda_gym',
        'rl_zoo3',
        'gymnasium',
        'gymnasium-robotics',
        'stable_baselines3[extra]',
        'sb3_contrib'
    ],
    include_package_data=True,
    description='Package to receive goal-directed environments',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/BenNageris/GoalRecognitionLibs/',  # Replace with your repository URL
    author='Ben Nageris, Osher Elhadad, Matan Shamir',
    author_email='',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
