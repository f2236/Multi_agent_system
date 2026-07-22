# Compatibility shim - use domain.py instead
# This module existed but now just redirects to avoid naming conflicts with stdlib 'types'
from uav_optimizer.domain import *  # noqa: F401, F403
