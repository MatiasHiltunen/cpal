use super::{Device, OSStatus};
use crate::{BackendSpecificError, DevicesError, SupportedStreamConfigRange};
use objc2_core_audio::{
    kAudioHardwareNoError, kAudioHardwarePropertyDefaultInputDevice,
    kAudioHardwarePropertyDefaultOutputDevice, kAudioHardwarePropertyDevices,
    kAudioObjectPropertyElementMaster, kAudioObjectPropertyScopeGlobal, kAudioObjectSystemObject,
    AudioDeviceID, AudioObjectGetPropertyData, AudioObjectGetPropertyDataSize, AudioObjectID,
    AudioObjectPropertyAddress,
};
use std::mem;
use std::ptr::{null, NonNull};
use std::vec::IntoIter as VecIntoIter;

unsafe fn audio_devices() -> Result<Vec<AudioDeviceID>, OSStatus> {
    let property_address = AudioObjectPropertyAddress {
        mSelector: kAudioHardwarePropertyDevices,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMaster,
    };

    macro_rules! try_status_or_return {
        ($status:expr) => {
            if $status != kAudioHardwareNoError {
                return Err($status);
            }
        };
    }

    // `AudioObjectGetPropertyDataSize` writes the required byte size back into
    // this variable.  It **must** therefore be mutable; passing a pointer
    // derived from an immutable reference triggers undefined behaviour (Core
    // Audio mutates data behind the compiler's back).  See the discussion in
    // <https://github.com/RustAudio/cpal/issues/768> for details.
    let mut data_size: u32 = 0;
    let status = AudioObjectGetPropertyDataSize(
        kAudioObjectSystemObject as AudioObjectID,
        NonNull::from(&property_address),
        0,
        null(),
        NonNull::from(&mut data_size),
    );
    try_status_or_return!(status);

    let device_count = data_size / mem::size_of::<AudioDeviceID>() as u32;

    // A zero-sized query is legal and indicates that no audio devices are
    // currently present (for example, a headless Continuous Integration
    // machine).  If we continued and passed `audio_devices.as_mut_ptr()` to
    // Core Audio, we would hand it a dangling (non-NULL) pointer of length 0 –
    // technically UB and liable to crash if dereferenced.  Instead, return an
    // empty vector early.
    if device_count == 0 {
        return Ok(Vec::new());
    }

    // Pre-allocate the exact capacity we need so that we can write directly
    // into the backing buffer without any reallocations while still keeping a
    // valid pointer for Core Audio.
    let mut audio_devices = Vec::with_capacity(device_count as usize);

    let status = AudioObjectGetPropertyData(
        kAudioObjectSystemObject as AudioObjectID,
        NonNull::from(&property_address),
        0,
        null(),
        NonNull::from(&mut data_size),
        NonNull::new(audio_devices.as_mut_ptr()).unwrap().cast(),
    );
    try_status_or_return!(status);

    audio_devices.set_len(device_count as usize);

    Ok(audio_devices)
}

pub struct Devices(VecIntoIter<AudioDeviceID>);

impl Devices {
    pub fn new() -> Result<Self, DevicesError> {
        let devices = unsafe {
            match audio_devices() {
                Ok(devices) => devices,
                Err(os_status) => {
                    let description = format!("{}", os_status);
                    let err = BackendSpecificError { description };
                    return Err(err.into());
                }
            }
        };
        Ok(Devices(devices.into_iter()))
    }
}

unsafe impl Send for Devices {}
unsafe impl Sync for Devices {}

impl Iterator for Devices {
    type Item = Device;
    fn next(&mut self) -> Option<Device> {
        self.0.next().map(|id| Device {
            audio_device_id: id,
            is_default: false,
        })
    }
}

pub fn default_input_device() -> Option<Device> {
    let property_address = AudioObjectPropertyAddress {
        mSelector: kAudioHardwarePropertyDefaultInputDevice,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMaster,
    };

    let mut audio_device_id: AudioDeviceID = 0;
    // The getter below also writes `data_size` back, so it needs to be mutable
    // for the same reason described earlier.
    let mut data_size = mem::size_of::<AudioDeviceID>() as u32;
    let status = unsafe {
        AudioObjectGetPropertyData(
            kAudioObjectSystemObject as AudioObjectID,
            NonNull::from(&property_address),
            0,
            null(),
            NonNull::from(&mut data_size),
            NonNull::from(&mut audio_device_id).cast(),
        )
    };
    // Fix for clippy warning `unnecessary_cast`.
    // The `kAudioHardwareNoError` constant is of type `OSStatus` (i32), so casting it is unnecessary.
    if status != kAudioHardwareNoError {
        return None;
    }

    let device = Device {
        audio_device_id,
        is_default: true,
    };
    Some(device)
}

pub fn default_output_device() -> Option<Device> {
    let property_address = AudioObjectPropertyAddress {
        mSelector: kAudioHardwarePropertyDefaultOutputDevice,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMaster,
    };

    let mut audio_device_id: AudioDeviceID = 0;
    // Mutable for the same reason – Core Audio may update the value.
    let mut data_size = mem::size_of::<AudioDeviceID>() as u32;
    let status = unsafe {
        AudioObjectGetPropertyData(
            kAudioObjectSystemObject as AudioObjectID,
            NonNull::from(&property_address),
            0,
            null(),
            NonNull::from(&mut data_size),
            NonNull::from(&mut audio_device_id).cast(),
        )
    };
    // Fix for clippy warning `unnecessary_cast`.
    // The `kAudioHardwareNoError` constant is of type `OSStatus` (i32), so casting it is unnecessary.
    if status != kAudioHardwareNoError {
        return None;
    }

    let device = Device {
        audio_device_id,
        is_default: true,
    };
    Some(device)
}

pub type SupportedInputConfigs = VecIntoIter<SupportedStreamConfigRange>;
pub type SupportedOutputConfigs = VecIntoIter<SupportedStreamConfigRange>;
