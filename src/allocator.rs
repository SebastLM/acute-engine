// Raw memory acquisition using Rust's global allocator.
// This module is independent of AcuteArena: it only acquires and releases
// whole regions. AcuteArena is responsible for carving those regions into
// per-tensor slices.

use std::alloc::{alloc, dealloc, Layout};

// Acquires size bytes of memory from Rust's global allocator.
// The memory is not guaranteed to be zeroed.
pub unsafe fn allocate(size: usize) -> Result<*mut u8, String> {
    if size == 0 {
        return Err("cannot allocate zero bytes".into());
    }

    let layout = Layout::from_size_align(size, std::mem::align_of::<usize>())
        .map_err(|_| format!("invalid allocation layout: size={size}"))?;

    let ptr = unsafe { alloc(layout) };
    // std::alloc::alloc_zeroed(layout) if zeroed memory needed

    if ptr.is_null() {
        return Err(format!("allocation failed for {size} bytes"));
    }

    Ok(ptr)
}

// Releases a region previously returned by deallocate().
// ptr and size must correspond to the original allocation.
pub unsafe fn deallocate(ptr: *mut u8, size: usize) {
    if ptr.is_null() || size == 0 {
        return;
    }

    let layout = Layout::from_size_align(
        size,
        std::mem::align_of::<usize>(),
    )
    .expect("allocation layout must be valid");

    unsafe {
        dealloc(ptr, layout);
    }
}