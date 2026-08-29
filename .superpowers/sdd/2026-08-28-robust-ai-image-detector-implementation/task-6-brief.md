### Task 6: Manifest dataset and paired batch collation

**Files:**

- Create: `src/prooflens/data/dataset.py`
- Create: `src/prooflens/data/sampling.py`
- Create: `src/prooflens/data/collate.py`
- Create: `src/prooflens/inference/preprocess.py`
- Create: `tests/unit/data/test_dataset.py`
- Create: `tests/unit/data/test_sampling.py`
- Create: `tests/unit/data/test_collate.py`

**Interfaces:**

- Consumes: Assigned manifest rows, transformation sampler, DINOv2 image processor.
- Produces: `SourceImageDataset`, source-balanced sampling weights, `PairedBatch`, `PairedBatchCollator`, tensors shaped `[batch, 3, 224, 224]`.

- [ ] **Step 1: Write source loading and corrupt-image tests**

```python
def test_source_dataset_returns_image_label_and_manifest_metadata(manifest_fixture):
    item = SourceImageDataset(manifest_fixture)[0]
    assert item.image.mode == "RGB"
    assert item.label in (0, 1)
    assert item.sample_id == manifest_fixture.iloc[0].sample_id


def test_source_dataset_raises_typed_decode_error(corrupt_manifest):
    with pytest.raises(ImageDecodeError):
        SourceImageDataset(corrupt_manifest)[0]


def test_sampling_weights_balance_labels_and_fake_generators(imbalanced_manifest):
    weights = compute_sampling_weights(imbalanced_manifest)
    weighted = imbalanced_manifest.assign(weight=weights)
    assert weighted.groupby("label").weight.sum().to_dict() == pytest.approx({0: 0.5, 1: 0.5})
    fake = weighted[weighted.label == 1]
    generator_mass = fake.groupby(["dataset_name", "generator_family"]).weight.sum()
    assert generator_mass.max() == pytest.approx(generator_mass.min())
```

- [ ] **Step 2: Write paired collator tests with a network-free fake processor**

```python
def test_paired_collator_keeps_labels_and_shapes(source_items, fake_processor):
    collator = PairedBatchCollator(
        processor=fake_processor,
        sampler=FixedTransformSampler("jpeg_q50"),
        seed=17,
    )
    batch = collator(source_items)
    assert batch.clean_pixels.shape == (2, 3, 224, 224)
    assert batch.transformed_pixels.shape == (2, 3, 224, 224)
    assert torch.equal(batch.labels, torch.tensor([0.0, 1.0]))
    assert batch.condition_ids == ("jpeg_q50", "jpeg_q50")
```

- [ ] **Step 3: Implement shared preprocessing and paired collation**

Use `AutoImageProcessor.from_pretrained("facebook/dinov2-base")` in production. Keep the processor injectable so unit tests do not require network access. Derive per-item transformation seeds from the run seed, epoch, and stable sample ID hash.

Compute sampling strata as `(label, dataset_name)` for authentic images and `(label, dataset_name, generator_family)` for synthetic images. Give each label total sampling mass 0.5, divide that mass equally among its strata, and divide each stratum's mass equally among its rows. Feed these weights to a seeded `WeightedRandomSampler`. This prevents SID-Set or one prolific generator from dominating batches while retaining every approved training example.

```python
def compute_sampling_weights(frame: pd.DataFrame) -> np.ndarray:
    strata = frame.apply(
        lambda row: (
            f"real:{row.dataset_name}"
            if row.label == 0
            else f"fake:{row.dataset_name}:{row.generator_family}"
        ),
        axis=1,
    )
    weights = np.zeros(len(frame), dtype=np.float64)
    for label in (0, 1):
        label_mask = frame.label.to_numpy() == label
        label_strata = strata[label_mask]
        names = sorted(label_strata.unique())
        for name in names:
            mask = label_mask & (strata.to_numpy() == name)
            weights[mask] = 0.5 / (len(names) * int(mask.sum()))
    return weights
```

```python
@dataclass(frozen=True)
class SourceItem:
    image: Image.Image
    label: int
    sample_id: str
    generator_family: str


@dataclass(frozen=True)
class PairedBatch:
    clean_pixels: Tensor
    transformed_pixels: Tensor
    labels: Tensor
    sample_ids: tuple[str, ...]
    condition_ids: tuple[str, ...]


class PairedBatchCollator:
    def __init__(self, processor, sampler, seed: int) -> None:
        self.processor = processor
        self.sampler = sampler
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __call__(self, items: Sequence[SourceItem]) -> PairedBatch:
        specs = [self.sampler.sample(item.sample_id, self.epoch) for item in items]
        seeds = [stable_seed(self.seed, self.epoch, item.sample_id) for item in items]
        transformed = [
            apply_transform(item.image, spec, item_seed)
            for item, spec, item_seed in zip(items, specs, seeds, strict=True)
        ]
        clean_pixels = self.processor(
            images=[item.image for item in items], return_tensors="pt"
        )["pixel_values"]
        transformed_pixels = self.processor(images=transformed, return_tensors="pt")["pixel_values"]
        return PairedBatch(
            clean_pixels=clean_pixels,
            transformed_pixels=transformed_pixels,
            labels=torch.tensor([item.label for item in items], dtype=torch.float32),
            sample_ids=tuple(item.sample_id for item in items),
            condition_ids=tuple(spec.condition_id for spec in specs),
        )
```

- [ ] **Step 4: Run tests and commit**

Run: `python -m pytest tests/unit/data/test_dataset.py tests/unit/data/test_sampling.py tests/unit/data/test_collate.py -v`

Expected: PASS.

```bash
git add src/prooflens/data/dataset.py src/prooflens/data/sampling.py src/prooflens/data/collate.py src/prooflens/inference/preprocess.py tests/unit/data
git commit -m "feat: add paired clean and transformed batches"
```

