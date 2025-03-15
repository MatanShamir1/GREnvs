# GREnvs
Gym Environments adjusted to Goal Recognition tasks.

## Installation
This repo is installable.
The name of the package is gr_envs.
The package serves as an extension with multiple gym environments and registration bundles that specifically fit GR frameworks, namely they are goal-conditioned.
You currently can only install the package by cloning the repo, it isn't distributed elsewhere.
to install regularly:
`pip install .`
to install in editable mode:
`pip install -e .`

You can also go to the dist folder which has a built version and install it, for example:
`pip install dist/gr_libs-0.1-py3-none-any.whl`

Make sure you have the package afterwards:
`pip list | grep gr-libs`

Installing the repo registers the environments to gym, effectively enabling you to run your script\framework having the environments existing out-of-the-box.

If you're on windows and using vscode (like me), you will need Microsoft Visual C++ 14.0 or greater. you can download a latest version here: https://visualstudio.microsoft.com/visual-cpp-build-tools/

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue if you have any suggestions or improvements.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

## For Developers
### Releasing a new version
1. Go to GitHub → Releases → Draft a new release.
2. Set the new version (e.g., v0.2.0).
3. Publish the release.
