"""Stable channel contract for the eight-channel representation."""

from symmlearn.features.base import FeatureSpecification


FEATURE_IDENTIFIER = "eight_channel_v1"

CHANNEL_NAMES = (
    "image",
    "reflection_strength",
    "reflection_sin_2theta",
    "reflection_cos_2theta",
    "rotation_2_fold",
    "rotation_3_fold",
    "rotation_4_fold",
    "rotation_6_fold",
)

FEATURE_SPECIFICATION = FeatureSpecification(
    identifier=FEATURE_IDENTIFIER,
    channel_names=CHANNEL_NAMES,
)
