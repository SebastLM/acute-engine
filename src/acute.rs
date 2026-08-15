// bump-pointer arena: one raw mmap'd region (from allocator.rs), carved up
// into per-tensor data slices as tensors get created. mirrors ggml_context,
// no per-tensor free, the whole arena gets reclaimed at once on drop

use crate::allocator;

pub struct AcuteArena {
    base: *mut u8,   // start of the mmap'd region
    size: usize,      // total capacity in bytes
    offset: usize,     // bump pointer, in bytes relative to `base`
    n_objects: usize,   // how many tensors have been carved out so far
}

impl AcuteArena {
    pub fn new(size: usize) -> Result<Self, String> {
        let base = unsafe { allocator::allocate(size)? };
        Ok(Self { base, size, offset: 0, n_objects: 0 })
    }

    // reserves space for n elements of T, returns a pointer to the start of
    // that region. memory comes zeroed from the kernel but is otherwise
    // uninitialized as far as T goes, caller owns writing n valid T's in
    // before reading through the pointer. pointer stays valid for the
    // arena's lifetime, nothing ever moves once placed
    pub fn alloc_tensor_data<T>(&mut self, n: usize) -> Result<*mut T, String> {
        let align = align_of::<T>();
        let size = n * size_of::<T>();

        let aligned_offset = (self.offset + align - 1) & !(align - 1);

        if aligned_offset + size > self.size {
            return Err(format!(
                "arena exhausted: need {size} bytes (align {align}), only {} of {} left",
                self.size - self.offset, self.size
            ));
        }

        // aligned_offset + size <= self.size is checked above, and base is
        // a live mapping of at least self.size bytes
        let ptr = unsafe { self.base.add(aligned_offset) as *mut T };
        self.offset = aligned_offset + size;
        self.n_objects += 1;
        Ok(ptr)
    }

    pub fn used(&self) -> usize { self.offset }
    pub fn capacity(&self) -> usize { self.size }
    pub fn n_objects(&self) -> usize { self.n_objects }
}

impl Drop for AcuteArena {
    fn drop(&mut self) {
        unsafe { allocator::deallocate(self.base, self.size); }
    }
}
