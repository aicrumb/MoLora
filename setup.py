from setuptools import setup, find_packages

setup(
  name = 'molora',
  packages = find_packages(exclude=[]),
  version = '0.0.1',
  license='MIT',
  description = 'MoLora - The first lora-based multi-expert-composed system, by crumb',
  author = 'Maxine Reams',
  author_email = 'aicrumbmail@gmail.com',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/aicrumb/MoLora',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'transformers',
    'lora',
    'peft'
  ],
  # todo
  #install_requires=[ 
  #],
)