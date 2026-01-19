# Models package
from .tdnn_generator import TDNNGenerator, TDNNGeneratorQAT
from .discriminator import Discriminator
from .pa_digital_twin import PADigitalTwin, VolterraPA

# Phase-normalized architecture (matches ARCHITECTURE.md v3.0)
from .pn_tdnn_generator import (
    PNTDNNGenerator,
    PhaseNormalizedFeatureExtraction,
    Discriminator as PNDiscriminator,
    create_pn_tdnn_generator,
    create_discriminator,
)

__all__ = [
    # Legacy (deprecated - use PN versions)
    'TDNNGenerator',
    'TDNNGeneratorQAT', 
    'Discriminator',
    'PADigitalTwin',
    'VolterraPA',
    # Phase-normalized (production)
    'PNTDNNGenerator',
    'PhaseNormalizedFeatureExtraction',
    'PNDiscriminator',
    'create_pn_tdnn_generator',
    'create_discriminator',
]
