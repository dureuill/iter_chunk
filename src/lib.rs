//! A liberal, stable, reimplementation of [`std::iter::Iterator::array_chunks`].

use std::{mem::MaybeUninit, ops::Range};

/// Split an iterator in chunks of at most N elements.
///
/// The last chunk may contain less than N elements.
pub fn chunk_iter<I: IntoIterator, const N: usize>(it: I) -> ChunkIter<I::IntoIter, N> {
    ChunkIter::new(it.into_iter())
}

/// The result of calling [`chunk_iter`].
///
/// An iterator of [`Chunk`]s.
pub struct ChunkIter<I: Iterator, const N: usize> {
    it: I,
}

/// A chunk of at most N `T`s.
///
/// `Iterator::Item` of [`ChunkIter`].
pub struct Chunk<T, const N: usize> {
    buffer: [MaybeUninit<T>; N],
    alive: Range<usize>,
}

impl<T, const N: usize> Chunk<T, N> {
    const UNINIT: MaybeUninit<T> = MaybeUninit::uninit();

    const fn unsupported_usize_max() {
        if N == usize::MAX {
            panic!("requires N < usize::MAX")
        }
    }

    fn empty() -> Self {
        Self {
            buffer: [Self::UNINIT; N],
            // (P1): alive.end == 0 < N
            alive: 0..0,
        }
    }

    fn fill<I>(&mut self, it: I)
    where
        I: Iterator<Item = T>,
    {
        Self::unsupported_usize_max();
        for (index, t) in (0..N).zip(it) {
            // SAFETY: index in 0..N, buffer length is N
            // (P2): buffer[alive] is initialized
            unsafe { self.buffer.get_unchecked_mut(index).write(t) };
            // (P1) alive.end = index + 1 < N
            self.alive.end = index + 1;
        }
    }

    /// Returns a slice view of this chunk.
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: only taking the alive part.
        unsafe { std::mem::transmute(&self.buffer[self.alive.clone()]) }
    }

    /// Returns a mutable slice view of this chunk.
    pub fn as_slice_mut(&mut self) -> &mut [T] {
        // SAFETY: only taking the alive part.
        unsafe { std::mem::transmute(&mut self.buffer[self.alive.clone()]) }
    }

    /// True if the chunk is empty.
    ///
    /// [`ChunkIter`] nevers returns an empty chunk, but iterating on a chunk will progressively make it empty.
    pub fn is_empty(&self) -> bool {
        self.alive.is_empty()
    }

    /// Size of a chunk. Guaranteed to be <= `N``.
    pub fn len(&self) -> usize {
        self.alive.len()
    }

    /// Whether the size of this chunk is `N`.
    ///
    /// [`ChunkIter`] always returns full chunks, except for the last chunks.
    /// Furthermore, iterating on a chunk will decrease it size.
    pub fn is_full(&self) -> bool {
        self.len() == N
    }
}

impl<T, const N: usize> ExactSizeIterator for Chunk<T, N> {
    fn len(&self) -> usize {
        self.len()
    }
}

impl<T, const N: usize> Iterator for Chunk<T, N> {
    type Item = T;

    /// Returns and consume the next item from the chunk, if it is not empty. This reduces the size of the chunk by 1.
    fn next(&mut self) -> Option<Self::Item> {
        // Maintaining (P1) by checking that alive.start < alive.end < N
        if self.alive.is_empty() {
            None
        } else {
            // SAFETY: alive.start < N because of (P1) and check above
            let t = unsafe { self.buffer.get_unchecked_mut(self.alive.start) };

            // SAFETY: t is initialized because of (P2)
            let t = unsafe { t.assume_init_read() };

            // Maintaining (P2) by removing consumed T from alive range
            self.alive.start += 1;
            Some(t)
        }
    }
}

impl<I: Iterator, const N: usize> ChunkIter<I, N> {
    fn new(it: I) -> Self {
        Self { it }
    }
}

impl<I: Iterator, const N: usize> Iterator for ChunkIter<I, N> {
    type Item = Chunk<I::Item, N>;

    /// Returns the next chunk if there remains one.
    ///
    /// Chunks are full except for the last one, which may or may not be full.
    fn next(&mut self) -> Option<Self::Item> {
        let mut chunk = Chunk::empty();
        chunk.fill(&mut self.it);
        if chunk.is_empty() {
            None
        } else {
            Some(chunk)
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    #[should_panic]
    fn unsupported_usize_max() {
        let it: ChunkIter<_, { usize::MAX }> = chunk_iter(std::iter::repeat(()));

        for chunk in it {
            assert_eq!(chunk.count(), usize::MAX);
        }
    }

    #[test]
    fn zero_chunk() {
        let mut it: ChunkIter<_, 0> = chunk_iter(std::iter::repeat(()));
        assert!(it.next().is_none())
    }

    #[test]
    fn three_chunk() {
        let mut it: ChunkIter<_, 3> = chunk_iter(0..=6);
        let mut chunk_0_3 = it.next().unwrap();
        assert_eq!(chunk_0_3.next(), Some(0));
        assert_eq!(chunk_0_3.next(), Some(1));
        assert_eq!(chunk_0_3.next(), Some(2));
        assert_eq!(chunk_0_3.next(), None);

        let mut chunk_3_6 = it.next().unwrap();
        assert_eq!(chunk_3_6.next(), Some(3));
        assert_eq!(chunk_3_6.next(), Some(4));
        assert_eq!(chunk_3_6.next(), Some(5));
        assert_eq!(chunk_3_6.next(), None);

        let mut chunk_6 = it.next().unwrap();
        assert_eq!(chunk_6.next(), Some(6));
        assert_eq!(chunk_3_6.next(), None);

        assert!(it.next().is_none());
    }

    #[test]
    fn two_chunk() {
        let mut it: ChunkIter<_, 3> = chunk_iter(0i32..6);
        let mut chunk_0_3 = it.next().unwrap();
        assert_eq!(chunk_0_3.next(), Some(0));
        assert_eq!(chunk_0_3.next(), Some(1));
        assert_eq!(chunk_0_3.next(), Some(2));
        assert_eq!(chunk_0_3.next(), None);

        let mut chunk_3_6 = it.next().unwrap();
        assert_eq!(chunk_3_6.next(), Some(3));
        assert_eq!(chunk_3_6.next(), Some(4));
        assert_eq!(chunk_3_6.next(), Some(5));
        assert_eq!(chunk_3_6.next(), None);

        assert!(it.next().is_none());
    }
}
