use core::slice;

// cool trick, this SHOULD inline to no code
#[inline(always)]
#[optimize(speed)]
pub fn endian_restrict<From, To>(from: From) -> To
where
    From: Sized,
    To: Sized,
{
    unsafe {
        let from_ptr = &from as *const From as *const u8;
        std::ptr::read(from_ptr as *const To)
    }
}

#[inline(always)]
#[optimize(speed)]
pub fn slide_window<T: Sized, const TARGET_SIZE: usize, const FULL_SIZE: usize>(
    full: &[T; FULL_SIZE],
    end_point: usize,
) -> &[T; TARGET_SIZE] {
    unsafe {
        &*((full.as_ptr().add(end_point.unchecked_sub(TARGET_SIZE))) as *const [T; TARGET_SIZE])
    }
}
