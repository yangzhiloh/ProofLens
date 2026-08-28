from dataclasses import dataclass
from pathlib import Path

WILDFAKE_REPOSITORY_URL = "https://github.com/hy-zpg/AIGC-Image-Detection-Dataset"
WILDFAKE_MODELSCOPE_URL = "https://modelscope.cn/datasets/hy2628982280/WildFake/summary"


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


def wildfake_manual_acquisition_message(root: Path) -> str:
    return (
        f"WildFake export root is missing or empty: {root}. WildFake acquisition is manual. "
        f"Follow the official repository at {WILDFAKE_REPOSITORY_URL} and obtain the dataset "
        f"from ModelScope at {WILDFAKE_MODELSCOPE_URL}, then point the configuration root to "
        "the extracted export."
    )
