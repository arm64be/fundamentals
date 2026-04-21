use std::ops::Index;

pub struct Int16Index(pub u16);

pub struct Int16Array<T, const N: usize>(pub [T; N]);

impl<T, const N: usize> Index<Int16Index> for Int16Array<T, N> {
    type Output = T;

    #[inline(always)]
    fn index(&self, index: Int16Index) -> &Self::Output {
        &self.0[index.0 as usize]
    }
}
