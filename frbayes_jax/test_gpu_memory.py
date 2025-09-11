#!/usr/bin/env python3
"""
Test GPU memory to understand the OOM issue.
"""
import jax
import jax.numpy as jnp
import numpy as np

print("="*60)
print("GPU MEMORY DIAGNOSTIC")
print("="*60)

# Check device
devices = jax.devices()
print(f"\nNumber of devices: {len(devices)}")
for i, device in enumerate(devices):
    print(f"Device {i}: {device}")
    if hasattr(device, 'device_kind'):
        print(f"  Kind: {device.device_kind}")

# Check memory stats if available
print("\nMemory stats:")
try:
    stats = devices[0].memory_stats()
    for key, value in stats.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value / 1e9:.2f} GB")
        else:
            print(f"  {key}: {value}")
except:
    print("  Memory stats not available")

# Test allocation
print("\nTesting allocations:")
sizes_mb = [100, 500, 1000, 2000, 4000, 8000]

for size_mb in sizes_mb:
    try:
        size_bytes = size_mb * 1024 * 1024
        elements = size_bytes // 8  # 8 bytes per float64
        
        # Try to allocate
        arr = jnp.zeros(elements, dtype=jnp.float32)
        arr.block_until_ready()  # Force allocation
        
        print(f"  ✓ Successfully allocated {size_mb} MB")
        
        # Clear it
        del arr
        jax.clear_caches()
        
    except Exception as e:
        print(f"  ✗ Failed to allocate {size_mb} MB: {e}")
        break

# Check JAX memory allocation settings
print("\nJAX Configuration:")
import os
print(f"  XLA_PYTHON_CLIENT_PREALLOCATE: {os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE', 'not set')}")
print(f"  XLA_PYTHON_CLIENT_MEM_FRACTION: {os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION', 'not set')}")
print(f"  JAX_PLATFORM_NAME: {os.environ.get('JAX_PLATFORM_NAME', 'not set')}")

# Test the specific size that's failing
print("\nTesting the failing allocation size (780MB):")
try:
    test_size = 780448000 // 4  # Divide by 4 for float32
    test_arr = jnp.zeros(test_size, dtype=jnp.float32)
    test_arr.block_until_ready()
    print(f"  ✓ Successfully allocated 780MB")
    del test_arr
except Exception as e:
    print(f"  ✗ Failed to allocate 780MB: {e}")

print("\n" + "="*60)