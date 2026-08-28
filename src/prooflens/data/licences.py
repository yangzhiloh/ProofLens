from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetAttribution:
    """Verified licence identifier and attribution details for a dataset."""

    dataset_name: str
    licence_identifier: str
    attribution: str


SID_SET = DatasetAttribution(
    dataset_name="sid_set",
    licence_identifier="CC-BY-4.0",
    attribution="SID-Set: source-material attribution is required.",
)
WILDFAKE = DatasetAttribution(
    dataset_name="wildfake",
    licence_identifier="REQUIRES-VERIFICATION",
    attribution="WildFake: retain the official paper and acquisition-source attribution.",
)
CIFAKE = DatasetAttribution(
    dataset_name="cifake_stress",
    licence_identifier="MIT",
    attribution="CIFAKE: retain required dataset citations.",
)
