from distutils.version import LooseVersion

import pydantic

USING_PYDANTIC_LEGACY = int(LooseVersion(pydantic.__version__).version[0]) < 2
