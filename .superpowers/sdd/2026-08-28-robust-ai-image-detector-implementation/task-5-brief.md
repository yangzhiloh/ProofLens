### Task 5: Canonical transformation registry

**Files:**

- Create: `src/prooflens/data/transforms.py`
- Create: `tests/unit/data/test_transforms.py`

**Interfaces:**

- Consumes: `PIL.Image.Image`, `TransformSpec`, integer seed.
- Produces: `canonical_specs() -> tuple[TransformSpec, ...]`, `training_condition_probabilities() -> dict[str, float]`, `apply_transform(image, spec, seed) -> Image`, stable `condition_id` strings.

- [ ] **Step 1: Write registry completeness tests**

```python
def test_canonical_registry_has_all_required_conditions():
    ids = {spec.condition_id for spec in canonical_specs()}
    assert ids == {
        "jpeg_q90", "jpeg_q70", "jpeg_q50", "jpeg_q30",
        "blur_s0.5", "blur_s1.0", "blur_s2.0",
        "resize_x0.5", "resize_x0.25",
        "noise_s0.02", "noise_s0.05", "noise_s0.10",
        "color_jitter_20", "center_crop_80",
    }


def test_every_transform_preserves_original_dimensions(rgb_fixture):
    for spec in canonical_specs():
        transformed = apply_transform(rgb_fixture, spec, seed=17)
        assert transformed.size == rgb_fixture.size
        assert transformed.mode == "RGB"


def test_training_probabilities_weight_families_equally():
    probabilities = training_condition_probabilities()
    specs = {spec.condition_id: spec for spec in canonical_specs()}
    family_mass = {
        family: sum(
            probability for condition, probability in probabilities.items()
            if specs[condition].family == family
        )
        for family in {spec.family for spec in specs.values()}
    }
    assert family_mass == pytest.approx({family: 1 / 6 for family in family_mass})
```

- [ ] **Step 2: Write deterministic and severity tests**

```python
def test_noise_is_seed_deterministic(rgb_fixture):
    spec = get_spec("noise_s0.05")
    assert np.array_equal(
        np.asarray(apply_transform(rgb_fixture, spec, 9)),
        np.asarray(apply_transform(rgb_fixture, spec, 9)),
    )


def test_stronger_blur_reduces_edge_energy(rgb_fixture):
    mild = np.asarray(apply_transform(rgb_fixture, get_spec("blur_s0.5"), 1))
    strong = np.asarray(apply_transform(rgb_fixture, get_spec("blur_s2.0"), 1))
    assert edge_energy(strong) < edge_energy(mild)
```

- [ ] **Step 3: Implement typed specs and exact transformation functions**

```python
@dataclass(frozen=True)
class TransformSpec:
    family: Literal["jpeg", "blur", "resize", "noise", "color_jitter", "center_crop"]
    condition_id: str
    severity: float
    parameters: Mapping[str, float | int | str]
```

Use an in-memory Pillow JPEG round-trip with recorded `quality` and `subsampling=2`; torchvision Gaussian blur with an odd `2 * ceil(3 * sigma) + 1` kernel; Pillow bicubic resize; NumPy `default_rng(seed)` for `[0,1]` Gaussian noise; seed-driven brightness, contrast, and saturation factors; and an 80 percent side-length center crop followed by bicubic restoration.

Training samples one of the six families uniformly, then one severity uniformly within that family. Do not sample uniformly over all 14 condition IDs because that would overweight JPEG and noise relative to crop and color jitter, while the primary metric weights families equally.

- [ ] **Step 4: Run transform tests and commit**

Run: `python -m pytest tests/unit/data/test_transforms.py -v`

Expected: PASS for registry, dimensions, determinism, and severity ordering.

```bash
git add src/prooflens/data/transforms.py tests/unit/data/test_transforms.py
git commit -m "feat: implement canonical robustness transformations"
```

