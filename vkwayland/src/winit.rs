#![warn(rust_2018_idioms)]

use std::{
    cell::RefCell,
    rc::Rc,
    sync::{atomic::AtomicBool, Arc, Mutex},
    time::{Duration, Instant},
};

#[macro_use(define_roles)]
extern crate smithay;
extern crate cgmath;
extern crate vulkan_malloc;

use dacite::{
    core::{
        DeviceExtensions, DeviceExtensionsProperties, Extent2D, Instance, InstanceExtensions, PhysicalDevice,
    },
    khr_surface::SurfaceKhr,
};

use smithay::reexports::winit;
use smithay::{
    backend::{
        input::{InputBackend, Seat as InputSeat, SeatCapabilities as InputSeatCapabilities},
        winit::{WindowSize, WinitInputBackend},
    },
    reexports::{
        calloop::EventLoop,
        wayland_server::{protocol::wl_output, Display as SmithayDisplay},
        winit::{dpi::LogicalSize, EventsLoop},
    },
    wayland::{
        data_device::{default_action_chooser, init_data_device, set_data_device_focus, DataDeviceEvent},
        output::{Mode, Output, PhysicalProperties},
        seat::{CursorImageStatus, Seat, XkbConfig},
        shm::init_shm_global,
    },
};

#[macro_use]
extern crate slog;

#[macro_use]
extern crate glsl_to_spirv_macros;
#[macro_use]
extern crate glsl_to_spirv_macros_impl;

use slog::{Drain, Logger};

mod input_handler;
use input_handler::AnvilInputHandler;
mod shell;
use shell::init_shell;

mod window_map;

mod vulkan_drawer;
use vulkan_drawer::{DeviceSettings, DisplayProber, Screen, VulkanDrawer, VulkanDrawerUtil};

#[macro_use]
extern crate bitflags;

mod dacite_winit {
    use std::{error, fmt};

    use dacite::{
        core, khr_surface, khr_wayland_surface, khr_win32_surface, khr_xlib_surface, wayland_types,
        win32_types, xlib_types,
    };

    use smithay::reexports::winit::Window;

    /// Extension trait for Vulkan surface creation.
    pub trait WindowExt {
        /// Test whether presentation is supported on a physical device.
        ///
        /// This function first determines the correct Vulkan WSI extension for this window and then calls one of the
        /// `get_*_presentation_support_*` family of functions on the `PhysicalDevice`.
        fn is_presentation_supported(
            &self,
            physical_device: &core::PhysicalDevice,
            queue_family_indices: u32,
        ) -> Result<bool, Error>;

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
        fn get_required_extensions(&self) -> Result<core::InstanceExtensionsProperties, Error>;

        /// Create a surface for this window.
        ///
        /// `Instance` must have been created with required extensions, as determined by
        /// `get_required_extensions()`. The `flags` parameter is currently just a place holder. You
        /// should specify `SurfaceCreateFlags::empty()` here.
        fn create_surface(
            &self,
            instance: &core::Instance,
            flags: SurfaceCreateFlags,
            allocator: Option<Box<dyn core::Allocator>>,
        ) -> Result<khr_surface::SurfaceKhr, Error>;
    }

    impl WindowExt for Window {
        fn is_presentation_supported(
            &self,
            physical_device: &core::PhysicalDevice,
            queue_family_indices: u32,
        ) -> Result<bool, Error> {
            let backend = get_backend(self)?;

            match backend {
                Backend::Xlib { .. } => Ok(true), // FIXME: This needs a VisualID, which winit does not expose
                Backend::Wayland { display, .. } => {
                    Ok(physical_device.get_wayland_presentation_support_khr(queue_family_indices, display))
                }
                Backend::Win32 { .. } => {
                    Ok(physical_device.get_win32_presentation_support_khr(queue_family_indices))
                }
            }
        }

        fn get_required_extensions(&self) -> Result<core::InstanceExtensionsProperties, Error> {
            let backend = get_backend(self)?;

            let mut extensions = core::InstanceExtensionsProperties::new();
            extensions.add_khr_surface(25);

            match backend {
                Backend::Xlib { .. } => extensions.add_khr_xlib_surface(6),
                Backend::Wayland { .. } => extensions.add_khr_wayland_surface(5),
                Backend::Win32 { .. } => extensions.add_khr_win32_surface(5),
            };

            Ok(extensions)
        }

        fn create_surface(
            &self,
            instance: &core::Instance,
            _: SurfaceCreateFlags,
            allocator: Option<Box<dyn core::Allocator>>,
        ) -> Result<khr_surface::SurfaceKhr, Error> {
            let backend = get_backend(self)?;

            match backend {
                Backend::Xlib { display, window } => {
                    let create_info = khr_xlib_surface::XlibSurfaceCreateInfoKhr {
                        flags: khr_xlib_surface::XlibSurfaceCreateFlagsKhr::empty(),
                        dpy: display,
                        window: window,
                        chain: None,
                    };

                    Ok(instance.create_xlib_surface_khr(&create_info, allocator)?)
                }

                Backend::Wayland { display, surface } => {
                    let create_info = khr_wayland_surface::WaylandSurfaceCreateInfoKhr {
                        flags: khr_wayland_surface::WaylandSurfaceCreateFlagsKhr::empty(),
                        display: display,
                        surface: surface,
                        chain: None,
                    };

                    Ok(instance.create_wayland_surface_khr(&create_info, allocator)?)
                }

                Backend::Win32 { hinstance, hwnd } => {
                    let create_info = khr_win32_surface::Win32SurfaceCreateInfoKhr {
                        flags: khr_win32_surface::Win32SurfaceCreateFlagsKhr::empty(),
                        hinstance: hinstance,
                        hwnd: hwnd,
                        chain: None,
                    };

                    Ok(instance.create_win32_surface_khr(&create_info, allocator)?)
                }
            }
        }
    }

    /// Error type used throughout this crate.
    #[derive(Debug)]
    pub enum Error {
        /// The windowing system is not supported by either dacite-winit or dacite.
        Unsupported,

        /// A Vulkan error occurred.
        VulkanError(core::Error),
    }

    impl fmt::Display for Error {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            match *self {
                Error::Unsupported => write!(f, "Unsupported"),
                Error::VulkanError(e) => write!(f, "VulkanError({})", e),
            }
        }
    }

    impl error::Error for Error {
        fn description(&self) -> &str {
            match *self {
                Error::Unsupported => "The windowing system is not supported",
                Error::VulkanError(ref e) => e.description(),
            }
        }
    }

    impl From<core::Error> for Error {
        fn from(e: core::Error) -> Self {
            Error::VulkanError(e)
        }
    }

    bitflags! {
        /// Flags used for surface creation.
        ///
        /// This is currently a placeholder, with no valid flags. Use `SurfaceCreateFlags::empty()`.
        pub struct SurfaceCreateFlags: u32 {
            /// Dummy flag
            ///
            /// This flag exists just to satisfy the bitflags! macro, which doesn't support empty
            /// flags. Use `SurfaceCreateFlags::empty()` instead.
            const SURFACE_CREATE_DUMMY = 0;
        }
    }

    #[allow(dead_code)]
    enum Backend {
        Xlib {
            display: *mut xlib_types::Display,
            window: xlib_types::Window,
        },

        Wayland {
            display: *mut wayland_types::wl_display,
            surface: *mut wayland_types::wl_surface,
        },

        Win32 {
            hinstance: win32_types::HINSTANCE,
            hwnd: win32_types::HWND,
        },
    }

    #[allow(unused_variables)]
    #[allow(unreachable_code)]
    fn get_backend(window: &Window) -> Result<Backend, Error> {
        #[cfg(any(
            target_os = "linux",
            target_os = "dragonfly",
            target_os = "freebsd",
            target_os = "openbsd"
        ))]
        {
            use smithay::reexports::winit::os::unix::WindowExt;

            if let (Some(display), Some(window)) = (window.get_xlib_display(), window.get_xlib_window()) {
                return Ok(Backend::Xlib {
                    display: display as _,
                    window: xlib_types::Window(window as _),
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

        Err(Error::Unsupported)
    }
}

use dacite_winit::WindowExt;

fn create_window(events_loop: &EventsLoop, extent: &Extent2D) -> Result<winit::Window, ()> {
    let logical_size = LogicalSize::new(extent.width as f64, extent.height as f64);
    let window = winit::WindowBuilder::new()
        .with_title("VkWayland")
        .with_dimensions(logical_size)
        .with_min_dimensions(logical_size)
        .with_max_dimensions(logical_size)
        .with_visibility(false)
        .build(&events_loop);
    let window = window.map_err(|e| match e {
        winit::CreationError::OsError(e) => println!("Failed to create window ({})", e),
        winit::CreationError::NotSupported => println!("Failed to create window (not supported)"),
    })?;
    Ok(window)
}

fn compute_instance_extensions(window: &winit::Window) -> Result<InstanceExtensions, ()> {
    let available_extensions = Instance::get_instance_extension_properties(None).map_err(|e| {
        println!("Failed to get instance extension properties ({})", e);
    })?;
    let required_extensions = (window as &dyn WindowExt)
        .get_required_extensions()
        .map_err(|e| match e {
            dacite_winit::Error::Unsupported => println!("The windowing system is not supported"),
            dacite_winit::Error::VulkanError(e) => {
                println!("Failed to get required extensions for the window ({})", e)
            }
        })?;
    let missing_extensions = required_extensions.difference(&available_extensions);
    if missing_extensions.is_empty() {
        Ok(required_extensions.to_extensions())
    } else {
        for (name, spec_version) in missing_extensions.properties() {
            println!("Extension {} (revision {}) missing", name, spec_version);
        }
        Err(())
    }
}

#[derive(Clone)]
struct Display {
    surface: Rc<SurfaceKhr>,
    extent: Rc<Extent2D>,
}

impl Screen for Display {
    fn get_surface(&self) -> Result<(Rc<SurfaceKhr>, Rc<Extent2D>, Duration), ()> {
        Ok((
            self.surface.clone(),
            self.extent.clone(),
            Duration::from_millis(8),
        ))
    }
}

struct Prober {
    surfaces: RefCell<Vec<Box<Display>>>,
}

impl DisplayProber for Prober {
    fn get_displays(&self) -> Result<Vec<(u64, Box<dyn Screen>)>, ()> {
        Ok(self
            .surfaces
            .borrow()
            .iter()
            .enumerate()
            .map(|(i, it)| (i as u64, it.clone() as Box<dyn Screen>))
            .collect())
    }
}

fn check_device_extensions(physical_device: Rc<PhysicalDevice>) -> Result<Rc<DeviceExtensions>, ()> {
    let available_extensions = physical_device
        .get_device_extension_properties(None)
        .map_err(|e| {
            println!("Failed to get device extension properties ({})", e);
        })?;
    let mut required_extensions = DeviceExtensionsProperties::new();
    required_extensions.add_khr_swapchain(67);
    let missing_extensions = required_extensions.difference(&available_extensions);
    if missing_extensions.is_empty() {
        Ok(Rc::new(required_extensions.to_extensions()))
    } else {
        for (name, spec_version) in missing_extensions.properties() {
            println!("Extension {} (revision {}) missing", name, spec_version);
        }
        Err(())
    }
}

fn find_suitable_device(
    instance: &Instance,
    display_prober: Box<dyn DisplayProber>,
) -> Result<Rc<DeviceSettings>, ()> {
    let physical_devices = instance.enumerate_physical_devices().map_err(|e| {
        println!("Failed to enumerate physical devices ({})", e);
    })?;
    for physical_device in physical_devices {
        let physical_device = Rc::new(physical_device);
        if let Ok(device_extensions) = check_device_extensions(physical_device.clone()) {
            return Ok(Rc::new(DeviceSettings {
                physical_device,
                device_extensions,
                display_prober,
            }));
        }
    }
    println!("Failed to find a suitable device");
    Err(())
}

pub fn run_winit(
    display: &mut SmithayDisplay,
    mut event_loop: &mut EventLoop<()>,
    log: Logger,
) -> Result<(), ()> {
    let preferred_extent = Extent2D::new(800, 600);

    let events_loop = EventsLoop::new();
    let window = create_window(&events_loop, &preferred_extent)?;

    let instance_extensions = compute_instance_extensions(&window)?;
    let instance = Rc::new(VulkanDrawer::create_instance(
        instance_extensions,
        Some("VkWaylandWinit".to_string()),
    )?);

    let drawer = VulkanDrawer::init(instance.clone(), log.clone());

    let surface = Rc::new(
        window
            .create_surface(&instance, dacite_winit::SurfaceCreateFlags::empty(), None)
            .map_err(|e| match e {
                dacite_winit::Error::Unsupported => println!("The windowing system is not supported"),
                dacite_winit::Error::VulkanError(e) => println!("Failed to create surface ({})", e),
            })?,
    );

    let size = Rc::new(RefCell::new(WindowSize {
        logical_size: window
            .get_inner_size()
            .expect("Winit window was killed during init."),
        dpi_factor: window.get_hidpi_factor(),
    }));

    let extent = Rc::new(Extent2D {
        width: size.borrow().logical_size.width as _,
        height: size.borrow().logical_size.height as _,
    });

    let prober = Box::new(Prober {
        surfaces: RefCell::new(vec![Box::new(Display { surface, extent })]),
    });
    drawer.new_device(find_suitable_device(&instance, prober as Box<dyn DisplayProber>)?)?;

    /* Todo: Call tick() at least once prior to uniding the Window */
    window.show();

    let mut input = WinitInputBackend {
        events_loop,
        events_handler: None,
        time: Instant::now(),
        key_counter: 0,
        seat: InputSeat::new(
            0,
            "winit",
            InputSeatCapabilities {
                pointer: true,
                keyboard: true,
                touch: true,
            },
        ),
        input_config: (),
        handler: None,
        logger: log.new(o!("smithay_winit_component" => "input")),
        size,
    };

    let name = display.add_socket_auto().unwrap().into_string().unwrap();
    info!(log, "Listening on wayland socket"; "name" => name.clone());
    ::std::env::set_var("WAYLAND_DISPLAY", name);

    let running = Arc::new(AtomicBool::new(true));

    /*
     * Initialize the globals
     */

    init_shm_global(display, vec![], log.clone());

    let (compositor_token, _, _, window_map) = init_shell(display, drawer.clone(), log.clone());

    let dnd_icon = Arc::new(Mutex::new(None));

    let dnd_icon2 = dnd_icon.clone();
    init_data_device(
        display,
        move |event| match event {
            DataDeviceEvent::DnDStarted { icon, .. } => {
                *dnd_icon2.lock().unwrap() = icon;
            }
            DataDeviceEvent::DnDDropped => {
                *dnd_icon2.lock().unwrap() = None;
            }
            _ => {}
        },
        default_action_chooser,
        compositor_token.clone(),
        log.clone(),
    );

    let (mut seat, _) = Seat::new(display, "winit".into(), compositor_token.clone(), log.clone());

    let cursor_status = Arc::new(Mutex::new(CursorImageStatus::Default));

    let cursor_status2 = cursor_status.clone();
    let pointer = seat.add_pointer(compositor_token.clone(), move |new_status| {
        // TODO: hide winit system cursor when relevant
        *cursor_status2.lock().unwrap() = new_status
    });

    let keyboard = seat
        .add_keyboard(XkbConfig::default(), 1000, 500, |seat, focus| {
            set_data_device_focus(seat, focus.and_then(|s| s.as_ref().client()))
        })
        .expect("Failed to initialize the keyboard");

    let (output, _) = Output::new(
        display,
        "Winit".into(),
        PhysicalProperties {
            width: 0,
            height: 0,
            subpixel: wl_output::Subpixel::Unknown,
            make: "Smithay".into(),
            model: "Winit".into(),
        },
        log.clone(),
    );

    output.change_current_state(
        Some(Mode {
            width: preferred_extent.width as i32,
            height: preferred_extent.height as i32,
            refresh: 60_000,
        }),
        None,
        None,
    );
    output.set_preferred(Mode {
        width: preferred_extent.width as i32,
        height: preferred_extent.height as i32,
        refresh: 60_000,
    });

    let pointer_location = Rc::new(RefCell::new((0.0, 0.0)));

    input.set_handler(AnvilInputHandler::new(
        log.clone(),
        pointer,
        keyboard,
        window_map.clone(),
        (0, 0),
        running.clone(),
        pointer_location.clone(),
    ));

    info!(log, "Initialization completed, starting the main loop.");

    loop {
        input.dispatch_new_events().unwrap();

        // drawing logic
        #[cfg(target_os = "__Unreachable__")]
        {
            let frame = drawer.draw()?;
            // frame.clear(None, Some((0.8, 0.8, 0.9, 1.0)), false, Some(1.0), None);

            // draw the windows
            drawer.draw_windows(frame, &*window_map.borrow(), compositor_token);

            let (x, y) = *pointer_location.borrow();
            // draw the dnd icon if any
            {
                let guard = dnd_icon.lock().unwrap();
                if let Some(ref surface) = *guard {
                    if surface.as_ref().is_alive() {
                        drawer.draw_dnd_icon(frame, surface, (x as i32, y as i32), compositor_token);
                    }
                }
            }
            // draw the cursor as relevant
            {
                let mut guard = cursor_status.lock().unwrap();
                // reset the cursor if the surface is no longer alive
                let mut reset = false;
                if let CursorImageStatus::Image(ref surface) = *guard {
                    reset = !surface.as_ref().is_alive();
                }
                if reset {
                    *guard = CursorImageStatus::Default;
                }
                // draw as relevant
                if let CursorImageStatus::Image(ref surface) = *guard {
                    drawer.draw_cursor(frame, surface, (x as i32, y as i32), compositor_token);
                }
            }

            /*
              if let Err(err) = frame.finish() {
                   error!(log, "Error during rendering: {:?}", err);
              }
            */
        }

        drawer
            .tick(&mut event_loop, &window_map, compositor_token)
            .unwrap();
        display.flush_clients();

        window_map.borrow_mut().refresh();
    }
}

fn main() {
    // A logger facility, here we use the terminal here
    let log = slog::Logger::root(
        slog_async::Async::default(slog_term::term_full().fuse()).fuse(),
        o!(),
    );

    let mut event_loop = EventLoop::<()>::new().unwrap();
    let mut display = SmithayDisplay::new(event_loop.handle());

    if let Err(()) = run_winit(&mut display, &mut event_loop, log.clone()) {
        crit!(log, "Failed to initialize winit backend.");
    }
}
