# Task 5 implementation report

## Result

Task 5 is implemented in commit `9a1362bffe820af244478b043a176628eeddb30e`.

## Files

1. `src/prooflens/data/transforms.py`
2. `tests/unit/data/test_transforms.py`

## Behavioral decisions

1. `TransformSpec` is a frozen, slotted dataclass. Construction copies each parameter mapping into a read-only `MappingProxyType`, so neither attribute assignment nor mutation of a caller-owned source dictionary can alter a spec.
2. The registry contains exactly 14 condition IDs in the locked order: four JPEG, three blur, two resize, three noise, one color jitter, and one center crop. Registry validation rejects missing families, duplicate IDs, and changes to condition order.
3. `group_specs_by_family()` follows the six-family contract order and preserves severity order within each tuple. `training_condition_probabilities()` assigns every family exactly `1/6` total mass and divides that mass uniformly among the family's conditions. It validates the total and each family mass instead of renormalizing an invalid registry.
4. `get_spec()` raises `UserInputError` for blank or unknown IDs. Invalid transform construction, unsupported family data, malformed image objects, and invalid seeds also raise `UserInputError`. Corrupted registry or output invariants raise `DataIntegrityError`.
5. Every transform first converts a copy of the input to RGB. It returns a distinct RGB image at the original dimensions and leaves the source object unchanged. Grayscale, RGBA, `1x1`, `1xN`, and `Nx1` images are supported.
6. JPEG uses an in-memory Pillow round trip with the canonical quality and `subsampling=2`. No filesystem path is involved.
7. Gaussian blur uses an exact separable NumPy Gaussian kernel with size `2 * ceil(3 * sigma) + 1`, edge padding, and the canonical sigma. This preserves the locked odd-kernel behavior on one-pixel dimensions, where torchvision's reflect padding raises a runtime error.
8. Resize uses bicubic downsampling and bicubic restoration, rounds scaled side lengths to the nearest integer, and clamps each downsampled dimension to at least one pixel.
9. Noise uses `numpy.random.default_rng(seed)` in normalized `[0,1]` space, independent Gaussian samples, clipping, and deterministic conversion back to bytes.
10. Color jitter samples independent brightness, contrast, and saturation factors from `[0.8,1.2]` with `default_rng(seed)`, then applies them in that order. Hue is unchanged.
11. Center crop rounds each 80 percent side length to the nearest integer, takes the centered box, and restores the original size with bicubic interpolation.

## TDD evidence

### RED

The complete behavioral test module was created before `src/prooflens/data/transforms.py` existed.

Command:

```powershell
$env:PYTHONPATH='src'
& 'C:\Users\Loh Yang Zhi\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m pytest tests\unit\data\test_transforms.py -v -p no:cacheprovider --basetemp '.pytest-tmp\task5-red-import'
```

Exact result: collection stopped with `0 items / 1 error` and `ModuleNotFoundError: No module named 'prooflens.data.transforms'`. The failure was caused by missing Task 5 production behavior. Python, pytest, project import resolution, and dependencies had initialized successfully.

### First GREEN correction

The first implementation run also stopped with `0 items / 1 error`. Canonical specs were instantiated before the module-level validation helper was defined, producing `NameError: name '_validated_parameter_copy' is not defined`. The module initialization was moved below its helper definitions. No contract or test expectation changed.

The next focused run produced `39 passed in 0.26s`.

## Final verification

Focused Task 5 verification:

```powershell
$env:PYTHONPATH='src'
& 'C:\Users\Loh Yang Zhi\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m pytest tests\unit\data\test_transforms.py -v -p no:cacheprovider --basetemp '.pytest-tmp\task5-precommit'
```

Evidence: `39 passed in 0.20s`, exit code 0.

Changed-file Ruff verification:

```powershell
& 'C:\Users\Loh Yang Zhi\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m ruff check --no-cache src\prooflens\data\transforms.py tests\unit\data\test_transforms.py
```

Evidence: `All checks passed!`, exit code 0.

Staged diff verification:

```powershell
git diff --cached --check
```

Evidence: no output, exit code 0.

## Coverage

The 39 focused cases cover exact registry order and uniqueness, external immutability, typed ID errors, deterministic family grouping, exact family probability mass, missing and duplicate registry failures, dimensions, mode, distinct output objects, source nonmutation, repeat determinism, stochastic seed sensitivity, JPEG quality direction, blur edge-energy direction, resize detail direction, noise variance direction, exact crop semantics, jitter factor independence and bounds, blur kernel sizes, grayscale, RGBA, tiny dimensions, malformed images, invalid seeds, malformed specs, and absence of filesystem or network access.

## Deferred gates and concerns

1. Per the user-approved milestone protocol, the full repository suite and independent review are deferred to the Milestone 1 gate after Task 6.
2. The editable installation still points at the retired worktree. Every pytest command explicitly set `PYTHONPATH=src` for this delivery repository and used workspace-local temporary paths with the cache provider disabled.
3. The approved brief mentioned torchvision Gaussian blur. The implementation instead uses the same locked kernel-size and sigma contract with a separable NumPy kernel because torchvision reflect padding fails on the required one-pixel inputs. Pixel-for-pixel parity with torchvision is not claimed. Determinism and severity behavior are covered directly.
4. Rounding for odd scaled or cropped dimensions was not specified. The implementation uses nearest-integer rounding with a one-pixel minimum. A later contract that requires floor semantics would change boundary pixels for some odd dimensions.
5. No full suite, remote push, merge, or amendment was performed.
