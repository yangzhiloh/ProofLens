class ProofLensError(Exception):
    """Base class for expected project failures."""


class UserInputError(ProofLensError):
    """The user supplied an invalid argument or input image."""


class DataIntegrityError(ProofLensError):
    """Dataset contents violate a required invariant."""


class DatasetAcquisitionError(DataIntegrityError):
    """Dataset acquisition could not produce the requested complete subset."""


class DatasetPolicyError(DataIntegrityError):
    """Dataset contents violate an approved source policy."""


class ManifestBuildError(DataIntegrityError):
    """A canonical manifest could not be built safely."""


class ImageDecodeError(DataIntegrityError):
    """An image cannot be decoded as RGB pixels."""


class LeakageError(DataIntegrityError):
    """Related source groups occur in multiple partitions."""


class MetricPartitionError(DataIntegrityError):
    """A requested metric partition is invalid."""


class TrainingError(ProofLensError):
    """Training or checkpoint recovery failed."""


class ExportError(ProofLensError):
    """Model export or numerical parity validation failed."""
