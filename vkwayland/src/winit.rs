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

use ash::vk;
use ash_tray::winit_ext::WindowExt;
use std::os::raw::c_char;

fn create_window(events_loop: &EventsLoop, extent: vk::Extent2D) -> Result<winit::Window, ()> {
    let logical_size = LogicalSize::new(extent.width as _, extent.height as _);
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

fn compute_instance_extensions(entry: ash::Entry, window: &winit::Window) -> Result<Vec<*const c_char>, ()> {
    let available_extensions = entry.enumerate_instance_extension_properties().map_err(|e| {
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

    let missing_extensions = required_extensions.iter().filter(|&want| {
        available_extensions
            .into_iter()
            .any(|have| want.0 == have.extension_name && want.1 <= have.spec_version)
            .any(|&have| {
                want.0 == &have.extension_name[0] as *const vk::c_char && want.1 <= have.spec_version
            })
    });
    if missing_extensions.is_empty() {
        Ok(required_extensions
            .into_iter()
            .map(|extention| &extention.0[0] as *const c_char)
            .collect())
    } else {
        for extention in missing_extensions.into_iter() {
            println!(
                "Extension {} (revision {}) missing",
                extention.extension_name, extention.spec_version
            );
        }
        Err(())
    }
}

fn check_device_extensions(
    instance: ash::Instance,
    device: vk::PhysicalDevice,
) -> Result<Vec<*const c_char>, ()> {
    let available_extensions =
        unsafe { instance.enumerate_device_extension_properties(device) }.map_err(|e| {
            println!("Failed to get device extension properties ({})", e);
        })?;

    let required_extensions = [(khr::Swapchain::name(), 67)];

    let missing_extensions = required_extensions.iter().filter(|&want| {
        available_extensions.iter().any(|&have| {
            want.0 == &have.extension_name[0] as *const vk::c_char && want.1 <= have.spec_version
        })
    });
    if missing_extensions.is_empty() {
        Ok(required_extensions
            .into_iter()
            .map(|extention| &extention.0[0] as *const c_char)
            .collect())
    } else {
        for extention in missing_extensions.into_iter() {
            println!(
                "Device Extension {} (revision {}) missing",
                extention.extension_name, extention.spec_version
            );
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
