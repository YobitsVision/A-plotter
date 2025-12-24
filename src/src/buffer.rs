use std::alloc::{alloc, dealloc, Layout};
use std::sync::{Arc, Mutex};
// use std::ptr;

pub struct PageAlignedByteBuffer {
    data: Option<Arc<Mutex<Vec<u8>>>>,
    pointer: *mut u8,
    layout: Layout,
}

impl PageAlignedByteBuffer {
    pub fn new(buffer_size: usize) -> Self {
        let alignment = page_size::get();
        let layout = Layout::from_size_align(buffer_size, alignment)
            .expect("Invalid layout for page-aligned buffer");

        unsafe {
            let pointer = alloc(layout);

            if pointer.is_null() {
                std::alloc::handle_alloc_error(layout);
            }

            let data = Vec::from_raw_parts(pointer, buffer_size, buffer_size);

            PageAlignedByteBuffer {
                data: Some(Arc::new(Mutex::new(data))),
                pointer,
                layout,
            }
        }
    }

    pub fn get_buffer(&self) -> Arc<Mutex<Vec<u8>>> {
        self.data.as_ref().unwrap().clone()
    }
}

impl Drop for PageAlignedByteBuffer {
    fn drop(&mut self) {
        // Prevent the Vec from dropping (and attempting to free) the memory
        std::mem::forget(self.data.take().unwrap());

        unsafe {
            dealloc(self.pointer, self.layout);
        }
    }
}

unsafe impl Send for PageAlignedByteBuffer {}

#[cfg(test)]
mod buffer_tests {
    use super::PageAlignedByteBuffer;

    #[test]
    fn buffer_creation_destruction_test() {
        {
            let _test = PageAlignedByteBuffer::new(1024 * 1024);
        }
        assert!(true);
    }
}