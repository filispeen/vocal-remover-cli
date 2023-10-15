from setuptools import find_packages, setup

setup(
    name='vc-remover-cli',
    packages=find_packages(),
    version='5.0.0',
    description='This is a deep-learning-based tool to extract instrumental track from your songs.',
    license='MIT',
    author='FILISPEEN',
    long_description = """Nine""",
    install_requires=['opencv_python', 'tqdm', 'librosa'],
    entry_points={
        "console_scripts": [
            "vc-remover = f_voice-remover.cli:main"],},
    url = 'https://github.com/filispeen/so-vits-svc-discord-webhook-notification',
    keywords = ['vc', 'remover', 'cli', 'voice-removal', 'voice', 'vc-remover', 'voice-remover'],
      classifiers=[
    'Development Status :: 5 - Production/Stable',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
  ],
)