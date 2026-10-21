import gc
import os
import sys
import time
from typing import List


def allocate_memory(chunk_size_mb: int = 64, total_mb: int = 1024, sleep_s: float = 0.05) -> List[bytearray]:
    """Allocate a set of large bytearrays to stress memory usage.

    Args:
        chunk_size_mb: Size of each buffer block in MB.
        total_mb: Total memory target in MB.
        sleep_s: Delay between allocations to let the OS and GC respond.

    Returns:
        List of allocated buffers.
    """
    chunk = chunk_size_mb * 1024 * 1024
    total = total_mb * 1024 * 1024
    buffers: List[bytearray] = []
    allocated = 0

    while allocated + chunk <= total:
        try:
            buffers.append(bytearray(chunk))
            allocated += chunk
            time.sleep(sleep_s)
        except MemoryError:
            print(f"MemoryError reached at approx {allocated / (1024 * 1024):.1f} MB")
            break

    print(f"Allocated ~{allocated / (1024 * 1024):.1f} MB in {len(buffers)} chunks")
    return buffers


def run_smoke_test() -> None:
    print("=== Memory smoke test ===")
    print(f"Python: {sys.version}")
    print(f"Platform: {sys.platform}")
    print(f"Working directory: {os.getcwd()}")

    # Use a moderate load that is meaningful but safe for most local machines.
    targets = [
        (32, 512, 0.05),
        (64, 768, 0.05),
        (128, 1024, 0.02),
    ]

    for chunk_mb, total_mb, delay in targets:
        print(f"\nAllocating chunk={chunk_mb}MB total={total_mb}MB delay={delay}s")
        allocate_memory(chunk_size_mb=chunk_mb, total_mb=total_mb, sleep_s=delay)
        gc.collect()
        time.sleep(0.5)

    print("\nSmoke test complete.")


if __name__ == "__main__":
    run_smoke_test()
