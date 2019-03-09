pub mod surface {
    use ash::vk;

    pub trait SurfaceExt {
        fn get_physical_device_presentation_support_khr(
            &self,
            physical_device: vk::PhysicalDevice,
            queue_family_indices: u32,
        ) -> bool;
    }
}

pub mod winit_ext {
    use std::os::raw::c_char;
    use winit::Window;

    use ash::{extensions::khr, prelude::VkResult, vk};

    /// Extension trait for Vulkan surface creation.
    pub trait WindowExt {
        /// Test whether presentation is supported on a physical device.
        ///
        /// This function first determines the correct Vulkan WSI extension for this window and then calls one of the
        /// `get_*_presentation_support_*` family of functions on the `PhysicalDevice`.
        fn is_presentation_supported(
            &self,
            physical_device: vk::PhysicalDevice,
            queue_family_indices: u32,
        ) -> VkResult<bool>;

        /// Determine required Vulkan instance extensions.
        ///
        /// This will always include [`VK_KHR_surface`]. One of the platform-dependent WSI extensions,
        /// that corresponds to this window, will also be added.
        ///
        /// Please note, that the device extension [`VK_KHR_swapchain`] is also required for
        /// presentation.
        ///
        /// [`VK_KHR_surface`]: https://www.khronos.org/registry/vulkan/specs/1.0-extensions/html/vkspec.html#VK_KHR_surface
        /// [`VK_KHR_swapchain`]: https://www.khronos.org/registry/vulkan/specs/1.0-extensions/html/vkspec.html#VK_KHR_swapchain
        fn get_required_extensions(&self) -> VkResult<Vec<*const c_char>>;

        /// Create a surface for this window.
        ///
        /// `Instance` must have been created with required extensions, as determined by
        /// `get_required_extensions()`. The `flags` parameter is currently just a place holder. You
        /// should specify `SurfaceCreateFlags::empty()` here.
        fn create_surface(
            &self,
            entry: &ash::Entry,
            instance: &ash::Instance,
            flags: vk::Flags,
            allocator: Option<&vk::AllocationCallbacks>,
        ) -> VkResult<vk::SurfaceKHR>;
    }

    impl WindowExt for Window {
        fn is_presentation_supported(
            &self,
            _physical_device: vk::PhysicalDevice,
            _queue_family_indices: u32,
        ) -> VkResult<bool> {
            match get_backend(self)? {
                Backend::Xlib { .. } => Ok(true), // FIXME: This needs a VisualID, which winit does not expose
                Backend::Wayland { .. } => {
                    // Ok(physical_device.get_wayland_presentation_support_khr(queue_family_indices, display))
                    Ok(true)
                }
                Backend::Win32 { .. } => {
                    // Ok(physical_device.get_win32_presentation_support_khr(queue_family_indices))
                    Ok(true)
                }
            }
        }

        fn get_required_extensions(&self) -> VkResult<Vec<*const c_char>> {
            match get_backend(self)? {
                Backend::Xlib { .. } => Ok(vec![
                    khr::Surface::name().as_ptr(),
                    khr::XlibSurface::name().as_ptr(),
                ]),
                Backend::Wayland { .. } => Ok(vec![
                    khr::Surface::name().as_ptr(),
                    khr::WaylandSurface::name().as_ptr(),
                ]),
                Backend::Win32 { .. } => Ok(vec![
                    khr::Surface::name().as_ptr(),
                    khr::Win32Surface::name().as_ptr(),
                ]),
            }
        }

        fn create_surface(
            &self,
            entry: &ash::Entry,
            instance: &ash::Instance,
            flags: vk::Flags,
            allocator: Option<&vk::AllocationCallbacks>,
        ) -> VkResult<vk::SurfaceKHR> {
            match get_backend(self)? {
                Backend::Xlib { display, window } => {
                    let create_info = vk::XlibSurfaceCreateInfoKHR {
                        flags: vk::XlibSurfaceCreateFlagsKHR::from_raw(flags),
                        dpy: display,
                        window: window,
                        ..Default::default()
                    };

                    unsafe {
                        khr::XlibSurface::new(entry, instance).create_xlib_surface(&create_info, allocator)
                    }
                }

                Backend::Wayland { display, surface } => {
                    let create_info = vk::WaylandSurfaceCreateInfoKHR {
                        flags: vk::WaylandSurfaceCreateFlagsKHR::from_raw(flags),
                        display: display,
                        surface: surface,
                        ..Default::default()
                    };

                    unsafe {
                        khr::WaylandSurface::new(entry, instance)
                            .create_wayland_surface(&create_info, allocator)
                    }
                }

                Backend::Win32 { hinstance, hwnd } => {
                    let create_info = vk::Win32SurfaceCreateInfoKHR {
                        flags: vk::Win32SurfaceCreateFlagsKHR::from_raw(flags),
                        hinstance: hinstance,
                        hwnd: hwnd,
                        ..Default::default()
                    };

                    unsafe {
                        khr::Win32Surface::new(entry, instance).create_win32_surface(&create_info, allocator)
                    }
                }
            }
        }
    }

    #[allow(dead_code)]
    enum Backend {
        Xlib {
            display: *mut vk::Display,
            window: vk::Window,
        },

        Wayland {
            display: *mut vk::wl_display,
            surface: *mut vk::wl_surface,
        },

        Win32 {
            hinstance: vk::HINSTANCE,
            hwnd: vk::HWND,
        },
    }

    #[allow(unused_variables)]
    #[allow(unreachable_code)]
    fn get_backend(window: &Window) -> VkResult<Backend> {
        #[cfg(any(
            target_os = "linux",
            target_os = "dragonfly",
            target_os = "freebsd",
            target_os = "openbsd"
        ))]
        {
            use winit::os::unix::WindowExt;

            if let (Some(display), Some(window)) = (window.get_xlib_display(), window.get_xlib_window()) {
                return Ok(Backend::Xlib {
                    display: display as _,
                    window: window as _,
                });
            }

            if let (Some(display), Some(surface)) =
                (window.get_wayland_display(), window.get_wayland_surface())
            {
                return Ok(Backend::Wayland {
                    display: display as _,
                    surface: surface as _,
                });
            }
        }

        #[cfg(target_os = "windows")]
        {
            use winit::os::windows::WindowExt;

            return Ok(Backend::Win32 {
                hinstance: ::std::ptr::null_mut(), // FIXME: Need HINSTANCE of the correct module
                hwnd: window.get_hwnd() as _,
            });
        }

        Err(vk::Result::ERROR_INITIALIZATION_FAILED)
    }
}
