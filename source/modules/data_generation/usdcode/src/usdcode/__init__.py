## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

import importlib
import os

from .usd_meta_functions_get import *
from .usd_meta_functions_set import *

from .metafunction_modules import MFAr
from .metafunction_modules import MFGf
from .metafunction_modules import MFKind
from .metafunction_modules import MFNdr
from .metafunction_modules import MFPcp
from .metafunction_modules import MFPlug
from .metafunction_modules import MFSdf
from .metafunction_modules import MFSdr
from .metafunction_modules import MFTf
from .metafunction_modules import MFTrace
from .metafunction_modules import MFUsd
from .metafunction_modules import MFUsdGeom
from .metafunction_modules import MFUsdLux
from .metafunction_modules import MFUsdPhysics
from .metafunction_modules import MFUsdShade
from .metafunction_modules import MFUsdSkel
from .metafunction_modules import MFUsdUtils
from .metafunction_modules import MFVt

from .setup import add_functions_from_file
