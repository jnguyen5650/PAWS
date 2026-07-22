from .video_sr_dataset import VideoSRDataset
from utils.dist import is_distributed, is_main_process
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
from typing import Iterator, List


class SkipBatchSamplerOnce(Sampler[List[int]]):
    """
    Wrapper around an existing *batch sampler* that skips the first N batches exactly once.

    This exists to support "resume mid-epoch" behavior without incurring dataset work
    (e.g., expensive degradations / augmentation / I/O) for batches that were already
    processed before a checkpoint was saved.

    Behavior:
        - On the *first* call to __iter__ after construction, it consumes (skips) the first
          `skip_batches` batches from the underlying batch sampler, then yields the rest.
        - On subsequent calls to __iter__ (future epochs), it yields the full underlying
          batch stream with no skipping.

    Important notes:
        - This wrapper only affects which indices the DataLoader requests. It does not
          change dataset length, shuffling logic, or distributed partitioning by itself.
          Those remain the responsibility of the underlying sampler/batch_sampler.
        - The "skip once" state is stored in-process via `_did_skip`. If you reconstruct
          the DataLoader (new process run), the skip will happen again, which is what you want
          when resuming from a checkpoint.

    Args:
        batch_sampler (Sampler[List[int]]):
            The underlying batch sampler yielding lists of dataset indices (one list per batch).
            In PyTorch DataLoader terms, this is typically `loader.batch_sampler`.
        skip_batches (int):
            Number of initial batches to discard on the first iterator only. Values <= 0
            are treated as 0.

    Typical usage:
        wrapped = SkipBatchSamplerOnce(train_loader.batch_sampler, step_in_epoch)
        train_loader = _clone_dataloader_with_batch_sampler(train_loader, wrapped)
    """
    def __init__(self, batch_sampler: Sampler[List[int]], skip_batches: int):
        self.batch_sampler = batch_sampler
        self.skip_batches = int(max(0, skip_batches))
        self._did_skip = False

    def __iter__(self) -> Iterator[List[int]]:
        """
        Yield batches of indices from the wrapped batch sampler.

        On the first iteration only, consume and discard `skip_batches` batches from the
        underlying iterator, then yield the remainder. On subsequent iterations, yield
        all batches (no skipping).

        Yields:
            List[int]: A list of dataset indices representing a batch.
        """
        it = iter(self.batch_sampler)

        if (not self._did_skip) and self.skip_batches > 0:
            for _ in range(self.skip_batches):
                try:
                    next(it)
                except StopIteration:
                    self._did_skip = True
                    return
            self._did_skip = True

        for batch in it:
            yield batch

    def __len__(self) -> int:
        """
        Return the number of batches this sampler will yield for *the next* iteration.

        For the first iterator (before skipping has been performed), the effective length
        is `len(base) - skip_batches` (clamped at 0). After skipping has occurred, we
        report the full `len(base)`.

        Returns:
            int: Number of batches expected from the next iterator.
        """
        base = len(self.batch_sampler)
        if (not self._did_skip) and self.skip_batches > 0:
            return max(0, base - self.skip_batches)
        return base


def build_dataloaders(config):
    scale_factor = config["model"].get("scale_factor", 4)
    batch_size = config["training"]["batch_size"]
    total_epochs = config["training"]["epochs"]

    train_dataset_config = config["dataset"]["train"]
    val_dataset_config = config["dataset"]["val"]

    degradation_config = config["degradation"]
    custom_degradation_config = config["custom_degradation"]

    train_dataset = VideoSRDataset(
        lr_dir=train_dataset_config["lr_dir"],
        hr_dir=train_dataset_config["hr_dir"],
        scale_factor=scale_factor,
        patch_size=train_dataset_config["patch_size"],
        num_frames=train_dataset_config["num_frames"],
        use_degradations=degradation_config.get("use_degradations", False),
        real_opt=degradation_config,
        use_custom_degradations=custom_degradation_config.get("use_custom_degradations", False),
        custom_opt=custom_degradation_config
    )

    val_dataset = VideoSRDataset(
        lr_dir=val_dataset_config["lr_dir"],
        hr_dir=val_dataset_config["hr_dir"],
        scale_factor=scale_factor,
        patch_size=val_dataset_config["patch_size"],
        num_frames=val_dataset_config["num_frames"],
        use_degradations=degradation_config.get("use_degradations", False),
        real_opt=degradation_config,
        use_custom_degradations=custom_degradation_config.get("use_custom_degradations", False),
        custom_opt=custom_degradation_config
    )

    if is_distributed():
        train_sampler = DistributedSampler(
            train_dataset, shuffle=config["dataloader"]["train"]["shuffle"]
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=config["dataloader"]["train"]["num_workers"],
            sampler=train_sampler,
            persistent_workers=config["dataloader"]["train"]["persistent_workers"],
            pin_memory=config["dataloader"]["train"]["pin_memory"],
            drop_last=True
        )
    else:
        train_sampler = None
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=config["dataloader"]["train"]["num_workers"],
            shuffle=config["dataloader"]["train"]["shuffle"],
            persistent_workers=config["dataloader"]["train"]["persistent_workers"],
            pin_memory=config["dataloader"]["train"]["pin_memory"],
            drop_last=True
        )
    
    if is_distributed():
        val_sampler = DistributedSampler(
            val_dataset, shuffle=config["dataloader"]["val"]["shuffle"], drop_last=False
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=config["dataloader"]["val"]["num_workers"],
            sampler=val_sampler,
            persistent_workers=config["dataloader"]["val"]["persistent_workers"],
            pin_memory=config["dataloader"]["val"]["pin_memory"],
        )
    else:
        val_sampler = None
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=config["dataloader"]["val"]["num_workers"],
            shuffle=config["dataloader"]["val"]["shuffle"],
            persistent_workers=config["dataloader"]["val"]["persistent_workers"],
            pin_memory=config["dataloader"]["val"]["pin_memory"]
        )

    if is_main_process():
        total_iters = total_epochs * len(train_loader)

        print("\n" + "=" * 72)
        print(" Training Configuration Summary")
        print("=" * 72)

        print(f"  Epochs              : {total_epochs}")
        print(f"  Iterations / Epoch  : {len(train_loader)}")
        print(f"  Total Iterations    : {total_iters}")
        print()

        print("  Data Geometry")
        print(f"    Scale Factor      : x{scale_factor}")
        print(f"    Batch Size        : {batch_size}")
        print(f"    Patch Size        : {train_dataset_config['patch_size']}")
        print(f"    Num Frames        : {train_dataset_config['num_frames']}")
        print()

        print("  Training Dataset")
        print(f"    LR Dir            : {train_dataset_config['lr_dir']}")
        print(f"    HR Dir            : {train_dataset_config['hr_dir']}")
        print(f"    Samples           : {len(train_dataset)}")
        print()

        print("  Validation Dataset")
        print(f"    LR Dir            : {val_dataset_config['lr_dir']}")
        print(f"    HR Dir            : {val_dataset_config['hr_dir']}")
        print(f"    Samples           : {len(val_dataset)}")
        print()

        print("  DataLoader")
        print(f"    Train Workers     : {config['dataloader']['train']['num_workers']}")
        print(f"    Val Workers       : {config['dataloader']['val']['num_workers']}")
        print(f"    Distributed       : {is_distributed()}")
        print("=" * 72 + "\n")

    return train_loader, val_loader, len(train_loader), len(val_loader), train_sampler, val_sampler


def _clone_dataloader_with_batch_sampler(loader, batch_sampler) -> DataLoader:
    """
    Rebuild a DataLoader that is identical to `loader` except for the batch_sampler.

    Newer PyTorch versions (and some configurations) freeze certain DataLoader attributes
    after initialization. In those cases, attempting to assign `loader.batch_sampler = ...`
    raises:
        ValueError: batch_sampler attribute should not be set after DataLoader is initialized

    This helper creates a new DataLoader instance that:
        - shares the same dataset object
        - preserves the majority of runtime settings (workers, collate_fn, pinning, etc.)
        - replaces only the `batch_sampler` (and therefore implicitly batch_size/shuffle/sampler/drop_last)

    Critical constraint:
        When `batch_sampler` is provided, you must NOT also pass:
            - batch_size
            - shuffle
            - sampler
            - drop_last
        because those are mutually exclusive with batch_sampler in DataLoader.

    Args:
        loader (DataLoader):
            Existing DataLoader instance to clone.
        batch_sampler:
            The batch sampler to use in the cloned DataLoader. Must yield lists of indices.

    Returns:
        DataLoader: A new DataLoader instance configured like `loader` but with `batch_sampler`.
    """
    # persistent_workers is only valid if num_workers > 0
    persistent = bool(getattr(loader, "persistent_workers", False)) and loader.num_workers > 0

    # Core arguments we want to preserve.
    # NOTE: We intentionally do NOT pass batch_size/shuffle/sampler/drop_last here.
    kwargs = dict(
        dataset=loader.dataset,
        batch_sampler=batch_sampler,
        num_workers=loader.num_workers,
        collate_fn=loader.collate_fn,
        pin_memory=loader.pin_memory,
        timeout=loader.timeout,
        worker_init_fn=loader.worker_init_fn,
        multiprocessing_context=loader.multiprocessing_context,
        generator=loader.generator,
        persistent_workers=persistent,
    )

    # prefetch_factor only makes sense for multiprocessing (num_workers > 0)
    if loader.num_workers > 0 and hasattr(loader, "prefetch_factor"):
        pf = loader.prefetch_factor
        if pf is not None:
            kwargs["prefetch_factor"] = pf

    if hasattr(loader, "pin_memory_device"):
        kwargs["pin_memory_device"] = loader.pin_memory_device

    if hasattr(loader, "in_order"):
        kwargs["in_order"] = loader.in_order

    return DataLoader(**kwargs)


def apply_resume_skip_to_train_loader(train_loader, step_in_epoch) -> DataLoader:
    """
    Wrap a train DataLoader so it skips the first `step_in_epoch` batches exactly once.

    This supports resuming from a checkpoint saved mid-epoch without:
        - re-running dataset __getitem__ (and any expensive degradations/augmentations)
        - needing to modify the training loop to "continue" past early batches

    How it works:
        - We wrap `train_loader.batch_sampler` with SkipBatchSamplerOnce(..., step_in_epoch).
        - On the first resumed epoch iteration, the wrapper consumes `step_in_epoch` batches
          from the underlying batch sampler (cheap: index generation only) and yields the rest.
        - On subsequent epochs, the wrapper yields full epochs as normal.

    Call site requirements:
        - Must be called BEFORE the first iteration over `train_loader` in the resumed run.
        - Typically called after loading the checkpoint (to know step_in_epoch), but before
          the first call to train_one_epoch.

    Args:
        train_loader (DataLoader):
            The existing training DataLoader.
        step_in_epoch (int):
            Number of already-processed batches within the current epoch (as saved in checkpoint).
            If <= 0, no changes are applied.

    Returns:
        DataLoader: The original loader (unchanged) or a wrapped/cloned loader that skips once.
    """
    step_in_epoch = int(step_in_epoch or 0)
    if step_in_epoch <= 0:
        return train_loader

    if isinstance(train_loader.batch_sampler, SkipBatchSamplerOnce):
        return train_loader # already wrapped

    wrapped = SkipBatchSamplerOnce(train_loader.batch_sampler, step_in_epoch)
    return _clone_dataloader_with_batch_sampler(train_loader, wrapped)
