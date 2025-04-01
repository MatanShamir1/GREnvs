# GREnvs
Gym Environments adjusted to Goal Recognition tasks.

## Installation
This repo is installable.
The name of the package is gr_envs.
The package serves as an extension with multiple gym environments and registration bundles that specifically fit GR frameworks, namely they are goal-conditioned.

The repo is distributed to Pypi.
to install the repo:
`pip install gr_envs`

Installing the repo registers the environments to gym, effectively enabling you to run your script\framework having the environments existing out-of-the-box.

If you're on windows and using vscode, you will need Microsoft Visual C++ 14.0 or greater. you can download a latest version here: https://visualstudio.microsoft.com/visual-cpp-build-tools/

### Installing Extras
This package offers additional environments via optional extras. To install a specific environment extra, include it in the pip install command:

- **Minigrid Environment:**  
  Installs the `minigrid` dependency.
  ```bash
  pip install gr_envs[minigrid]
  ```

- **Panda Environment:**  
  Installs the `panda_gym` dependency.
  ```bash
  pip install gr_envs[panda]
  ```

- **Parking Environment:**  
  *(Corresponds to the `highway-env` dependency.)*  
  ```bash
  pip install gr_envs[highway]
  ```

- **Point-Maze Environment:**  
  *(Corresponds to the `gymnasium-robotics` dependency.)*  
  ```bash
  pip install gr_envs[maze]
  ```

## Supported envs:
- **Minigrid**
- **Panda**
- **Parking**
- **Point-Maze**

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue if you have any suggestions or improvements.

## License
This project is licensed under the MIT License.
