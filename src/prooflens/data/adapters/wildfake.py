from collections.abc import Iterator, Mapping
from pathlib import Path

from prooflens.data.licences import WILDFAKE
from prooflens.data.schema import ManifestRecord
from prooflens.errors import DataIntegrityError


class WildFakeAdapter:
    """Read WildFake labels and generator identities from an explicit hierarchy."""

    def __init__(
        self,
        root: Path,
        version: str = "main",
        real_directories: Mapping[str, str] | None = None,
        fake_directories: Mapping[str, str] | None = None,
    ) -> None:
        self.root = root
        self.version = version
        self.real_directories = dict(real_directories or {"real": "authentic"})
        self.fake_directories = dict(fake_directories or {"fake": "generator"})

    def scan(self) -> Iterator[ManifestRecord]:
        if not self.root.is_dir():
            raise DataIntegrityError(f"WildFake root is missing: {self.root}. Mount the approved export root.")
        records: list[ManifestRecord] = []
        for directory_name, family in sorted(self.real_directories.items()):
            directory = self._required_directory(directory_name)
            paths = list(_image_paths(directory))
            if not paths:
                raise DataIntegrityError(f"WildFake real directory has no images: {directory}")
            records.extend(self._record(path, 0, family) for path in paths)
        for directory_name, identity in sorted(self.fake_directories.items()):
            root = self._required_directory(directory_name)
            fake_records: list[ManifestRecord] = []
            if identity == "generator":
                for generator_dir in sorted(
                    (path for path in root.iterdir() if path.is_dir()), key=lambda p: p.name
                ):
                    for path in _image_paths(generator_dir):
                        fake_records.append(self._record(path, 1, generator_dir.name))
            else:
                fake_records.extend(self._record(path, 1, identity) for path in _image_paths(root))
            if not fake_records:
                raise DataIntegrityError(f"WildFake fake directory has no images: {root}")
            records.extend(fake_records)
        yield from records

    def _required_directory(self, directory_name: str) -> Path:
        directory = self.root / directory_name
        if not directory.is_dir():
            raise DataIntegrityError(
                f"WildFake required directory is missing: {directory}. Check the configured hierarchy."
            )
        return directory

    def _record(self, path: Path, label: int, generator_family: str) -> ManifestRecord:
        sample_id = path.relative_to(self.root).as_posix()
        return ManifestRecord(
            sample_id=sample_id,
            path=path,
            label=label,
            dataset_name=WILDFAKE.dataset_name,
            dataset_version=self.version,
            generator_family=generator_family,
            source_group_id=sample_id,
            original_image_id=path.stem,
            width=1,
            height=1,
            file_format="UNKNOWN",
            licence_identifier=WILDFAKE.licence_identifier,
        )


def _image_paths(root: Path) -> Iterator[Path]:
    if not root.exists():
        return iter(())
    return iter(sorted((path for path in root.rglob("*") if path.is_file()), key=lambda p: p.as_posix()))
