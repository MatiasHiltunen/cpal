//! Helper code for registering audio object property listeners.
use std::ptr::NonNull;
use std::panic::{self, AssertUnwindSafe};

use objc2_core_audio::{
    AudioObjectAddPropertyListener, AudioObjectID, AudioObjectPropertyAddress,
    AudioObjectRemovePropertyListener,
};

use super::OSStatus;
use crate::BuildStreamError;

/// A double-indirection to be able to pass a closure (a fat pointer)
/// via a single c_void.
struct PropertyListenerCallbackWrapper(Box<dyn FnMut() + Send>);

/// Maintain an audio object property listener.
/// The listener will be removed when this type is dropped.
pub struct AudioObjectPropertyListener {
    callback: Box<PropertyListenerCallbackWrapper>,
    property_address: AudioObjectPropertyAddress,
    audio_object_id: AudioObjectID,
    removed: bool,
}

impl AudioObjectPropertyListener {
    /// Attach the provided callback as a audio object property listener.
    pub fn new<F: FnMut() + Send + 'static>(
        audio_object_id: AudioObjectID,
        property_address: AudioObjectPropertyAddress,
        callback: F,
    ) -> Result<Self, BuildStreamError> {
        // Allocate the wrapper but keep ownership until we've successfully registered.
        let mut callback_box = Box::new(PropertyListenerCallbackWrapper(Box::new(callback)));

        // Stable pointer passed to Core Audio.
        let cb_ptr = &mut *callback_box as *mut _ as *mut _;

        unsafe {
            coreaudio::Error::from_os_status(AudioObjectAddPropertyListener(
                audio_object_id,
                NonNull::from(&property_address),
                Some(property_listener_handler_shim),
                cb_ptr,
            ))?;
        };

        Ok(Self {
            callback: callback_box,
            audio_object_id,
            property_address,
            removed: false,
        })
    }

    /// Explicitly remove the property listener.
    /// Use this method if you need to explicitly handle failure to remove
    /// the property listener.
    pub fn remove(mut self) -> Result<(), BuildStreamError> {
        self.remove_inner()
    }

    fn remove_inner(&mut self) -> Result<(), BuildStreamError> {
        unsafe {
            coreaudio::Error::from_os_status(AudioObjectRemovePropertyListener(
                self.audio_object_id,
                NonNull::from(&self.property_address),
                Some(property_listener_handler_shim),
                &*self.callback as *const _ as *mut _,
            ))?;
        }
        self.removed = true;
        Ok(())
    }
}

impl Drop for AudioObjectPropertyListener {
    fn drop(&mut self) {
        if !self.removed {
            let _ = self.remove_inner();
        }
    }
}

/// Callback used to call user-provided closure as a property listener.
unsafe extern "C-unwind" fn property_listener_handler_shim(
    _: AudioObjectID,
    _: u32,
    _: NonNull<AudioObjectPropertyAddress>,
    callback: *mut ::std::os::raw::c_void,
) -> OSStatus {
    let wrapper = &mut *(callback as *mut PropertyListenerCallbackWrapper);

    // Catch panics so they don't unwind across the FFI boundary (which would abort).
    let _ = panic::catch_unwind(AssertUnwindSafe(|| {
        (wrapper.0)();
    }));

    0
}
