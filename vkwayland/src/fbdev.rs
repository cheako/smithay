#![warn(rust_2018_idioms)]

#[macro_use]
extern crate slog;
#[macro_use(define_roles)]
extern crate smithay;
extern crate cgmath;

#[macro_use]
extern crate glsl_to_spirv_macros;
#[macro_use]
extern crate glsl_to_spirv_macros_impl;

use slog::Drain;
use slog_syslog::Facility;

use smithay::reexports::{calloop::EventLoop, wayland_server::Display};

mod input_handler;
mod shell;
mod udev;
mod vulkan_drawer;
mod window_map;

fn main() {
    // A logger facility, here we use syslog here
    match slog_syslog::unix_3164(Facility::LOG_USER) {
        Ok(drain) => {
            let log = slog::Logger::root(
                slog_async::Async::default(slog::IgnoreResult::new(drain).fuse()).fuse(),
                o!(),
            );

            let mut event_loop = EventLoop::<()>::new().unwrap();
            let display = Display::new(event_loop.handle());

            if let Err(()) = udev::run_udev(display, &mut event_loop, log.clone()) {
                crit!(log, "Failed to initialize fbdev backend.");
            }
        }
        Err(e) => println!("Failed to start syslog. Error {:?}", e),
    };
}
