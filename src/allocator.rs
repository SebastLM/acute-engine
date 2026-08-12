// raw memory acquisition from the kernel, bypasses the global Rust/libc
// allocator entirely. only hands out whole regions (mmap/munmap), carving
// those into per-tensor slices is AcuteArena's job (acute.rs)

use core::arch::asm;

const PROT_READ: usize = 0x1;
const PROT_WRITE: usize = 0x2;
const MAP_PRIVATE: usize = 0x02;
const MAP_ANONYMOUS: usize = 0x20;

const SYS_MMAP: usize = 9;
const SYS_MUNMAP: usize = 11;

// requests `size` bytes of fresh, zeroed, anonymous memory from the kernel,
// returns a pointer to the start of the mapping.
// size must be > 0, pointer stays valid until unmap() is called with the same size
pub unsafe fn map_anonymous(size: usize) -> Result<*mut u8, String> {
    let ret: usize;
    unsafe {
        asm!(
            "syscall",
            inlateout("rax") SYS_MMAP => ret,
            in("rdi") 0usize,                      // addr: let the kernel pick one
            in("rsi") size,                         // length
            in("rdx") PROT_READ | PROT_WRITE,        // prot
            in("r10") MAP_PRIVATE | MAP_ANONYMOUS,   // flags
            in("r8") usize::MAX,                     // fd: -1 as usize
            in("r9") 0usize,                         // offset
            lateout("rcx") _,                        // syscall clobbers rcx (return addr)
            lateout("r11") _,                        // and r11 (rflags)
            options(nostack)
        );
    }

    // the kernel returns -errno (as a value in [-4095, -1]) on failure,
    // which as an unsigned usize sits right below the top of the range.
    let signed = ret as isize;
    if signed < 0 && signed > -4096 {
        return Err(format!("mmap failed, errno {}", -signed));
    }
    Ok(ret as *mut u8)
}

// releases a mapping previously returned by map_anonymous.
// ptr/size must exactly match a still-live mapping, and nothing may read or
// write through ptr (or anything derived from it) after this call
pub unsafe fn unmap(ptr: *mut u8, size: usize) {
    unsafe {
        asm!(
            "syscall",
            in("rax") SYS_MUNMAP,
            in("rdi") ptr,
            in("rsi") size,
            lateout("rcx") _,
            lateout("r11") _,
            options(nostack)
        );
    }
}
