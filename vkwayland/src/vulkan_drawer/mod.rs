use std::{
    cell::RefCell,
    collections::{hash_map::Entry, HashMap},
    rc::Rc,
    time::{Duration, SystemTime},
};

use slog::Logger;

use smithay::{
    reexports::wayland_server::{
        calloop::EventLoop,
        protocol::{wl_buffer::WlBuffer, wl_shm::Format as ShmFormat, wl_surface},
    },
    wayland::{
        compositor::{roles::Role, SubsurfaceRole, TraversalAction},
        data_device::DnDIconRole,
        seat::CursorImageRole,
        shm::BufferData,
    },
};

// extern crate dacite;
use dacite::{
    core::{
        Buffer, DescriptorSet, DeviceExtensions, Extent2D, Format, Image, ImageView, Instance,
        InstanceExtensions, MappedMemory, MappedMemoryRange, PhysicalDevice,
    },
    khr_surface::SurfaceKhr,
    VulkanObject,
};

use crate::shell::{MyCompositorToken, MyWindowMap};

mod device;
mod display;
mod init;

use device::Device;

pub struct VulkanDrawer {
    pub instance: Rc<Instance>,
    pub log: Logger,
    devices: RefCell<HashMap<u64, Rc<Device>>>,
}

pub trait VulkanDrawerUtil {
    fn create_instance(
        instance_extensions: InstanceExtensions,
        engine_name: Option<String>,
    ) -> Result<Instance, ()>;
}

pub trait Screen {
    fn get_surface(&self) -> Result<(Rc<SurfaceKhr>, Rc<Extent2D>, Duration), ()>;
}

pub trait DisplayProber {
    fn get_displays(&self) -> Result<Vec<(u64, Box<dyn Screen>)>, ()>;
}

pub struct DeviceSettings {
    pub physical_device: Rc<PhysicalDevice>,
    pub device_extensions: Rc<DeviceExtensions>,
    pub display_prober: Box<dyn DisplayProber>,
}

#[derive(Clone)]
pub struct TextureMetadata {
    pub shm: Option<(RefCell<(Buffer, MappedMemoryRange)>, RefCell<BufferData>)>,
    pub texture: Rc<(Image, MappedMemoryRange)>,
    pub texture_view: Rc<ImageView>,
    pub fragment: usize,
    pub y_inverted: bool,
    pub format: Rc<Format>,
    pub uniform: Rc<(Buffer, MappedMemoryRange)>,
    pub uniform_map: Rc<MappedMemory>,
    pub uniform_matrix: *mut cgmath::Matrix4<f32>,
    pub descriptor_set: Rc<DescriptorSet>,
}

impl VulkanDrawer {
    pub fn init(instance: Rc<Instance>, log: Logger) -> Rc<VulkanDrawer> {
        Rc::new(VulkanDrawer {
            instance,
            log,
            devices: RefCell::new(HashMap::new()),
        })
    }

    pub fn new_device(&self, device_settings: Rc<DeviceSettings>) -> Result<(), ()> {
        match self
            .devices
            .borrow_mut()
            .entry(device_settings.physical_device.id())
        {
            Entry::Vacant(o) => {
                o.insert(Rc::new(Device::init(device_settings)?));
                Ok(())
            }
            _ => Err(()),
        }
    }

    pub fn surface_commit(
        &self,
        buffer: &WlBuffer,
        textures: &mut Option<RefCell<HashMap<u64, RefCell<TextureMetadata>>>>,
    ) -> bool {
        if match textures {
            None => {
                let device_map: RefCell<HashMap<u64, RefCell<TextureMetadata>>> =
                    RefCell::new(HashMap::new());
                for (id, device) in self.devices.borrow().iter() {
                    if let Ok(texture) = device.create_texture(buffer) {
                        device_map.borrow_mut().insert(*id, texture);
                    }
                }
                let ret = !device_map.borrow().is_empty();
                *textures = Some(device_map);
                ret
            }
            Some(textures) => {
                for (id, device) in self.devices.borrow().iter() {
                    match device.surface_commit(buffer, &mut textures.borrow()[id].borrow_mut()) {
                        Ok(()) => {}
                        Err(()) => {
                            textures.borrow_mut().remove(id);
                        }
                    }
                }
                !textures.borrow().is_empty()
            }
        } {
            buffer.release();
            true
        } else {
            false
        }
    }

    pub fn tick(
        &self,
        event_loop: &mut EventLoop<()>,
        window_map: &RefCell<MyWindowMap>,
        compositor_token: MyCompositorToken,
    ) -> Result<(), ()> {
        let counter = SystemTime::now();

        let mut times: Vec<Duration> = vec![];
        for (id, device) in self.devices.borrow().iter() {
            times.push(device.tick(&id, &window_map, compositor_token)?);
        }
        let time = times
            .iter()
            .cloned()
            .fold(Duration::from_millis(17), Duration::min);

        let elapsed = counter.elapsed().map_err(|e| {
            println!("Failed to calculate elapsed ({})", e);
        })?;

        event_loop
            .dispatch(
                Some(if time > elapsed {
                    time - elapsed
                } else {
                    Duration::from_secs(0)
                }),
                &mut (),
            )
            .map_err(|e| {
                println!("Failed to dispatch event loop ({})", e);
            })
    }

    #[cfg(feature = "__Unreachable__")]
    pub fn texture_from_buffer(&self, buffer: wl_buffer::WlBuffer) -> Result<TextureMetadata, ()> {
        // try to retrieve the egl contents of this buffer
        let images = if let Some(display) = &self.egl_display.borrow().as_ref() {
            display.egl_buffer_contents(buffer)
        } else {
            Err(BufferAccessError::NotManaged(buffer))
        };
        match images {
            Ok(images) => {
                // we have an EGL buffer
                let format = match images.format {
                    Format::RGB => UncompressedFloatFormat::U8U8U8,
                    Format::RGBA => UncompressedFloatFormat::U8U8U8U8,
                    _ => {
                        warn!(self.log, "Unsupported EGL buffer format"; "format" => format!("{:?}", images.format));
                        return Err(());
                    }
                };
                let opengl_texture = Texture2d::empty_with_format(
                    &self.display,
                    format,
                    MipmapsOption::NoMipmap,
                    images.width,
                    images.height,
                )
                .unwrap();
                unsafe {
                    images
                        .bind_to_texture(0, opengl_texture.get_id())
                        .expect("Failed to bind to texture");
                }
                Ok(TextureMetadata {
                    texture: opengl_texture,
                    fragment: crate::shaders::BUFFER_RGBA,
                    y_inverted: images.y_inverted,
                    dimensions: (images.width, images.height),
                    images: Some(images), // I guess we need to keep this alive ?
                })
            }
            Err(BufferAccessError::NotManaged(buffer)) => {
                // this is not an EGL buffer, try SHM
                self.texture_from_shm_buffer(buffer)
            }
            Err(err) => {
                error!(self.log, "EGL error"; "err" => format!("{:?}", err));
                Err(())
            }
        }
    }

    #[cfg(feature = "__Unreachable__")]
    fn texture_from_shm_buffer(&self, buffer: wl_buffer::WlBuffer) -> Result<TextureMetadata, ()> {
        match shm_buffer_contents(&buffer, |slice, data| {
            crate::shm_load::load_shm_buffer(data, slice)
                .map(|(image, kind)| (Texture2d::new(&self.display, image).unwrap(), kind, data))
        }) {
            Ok(Ok((texture, kind, data))) => Ok(TextureMetadata {
                texture,
                fragment: kind,
                y_inverted: false,
                dimensions: (data.width as u32, data.height as u32),
                #[cfg(feature = "egl")]
                images: None,
            }),
            Ok(Err(format)) => {
                warn!(self.log, "Unsupported SHM buffer format"; "format" => format!("{:?}", format));
                Err(())
            }
            Err(err) => {
                warn!(self.log, "Unable to load buffer contents"; "err" => format!("{:?}", err));
                Err(())
            }
        }
    }

    #[cfg(feature = "__Unreachable__")]
    pub fn load_shm_buffer(data: BufferData, pool: &[u8]) -> Result<(RawImage2d<'_, u8>, usize), Format> {
        let offset = data.offset as usize;
        let width = data.width as usize;
        let height = data.height as usize;
        let stride = data.stride as usize;

        // number of bytes per pixel
        // TODO: compute from data.format
        let pixelsize = 4;

        // ensure consistency, the SHM handler of smithay should ensure this
        assert!(offset + (height - 1) * stride + width * pixelsize <= pool.len());

        let slice: Cow<'_, [u8]> = if stride == width * pixelsize {
            // the buffer is cleanly continuous, use as-is
            Cow::Borrowed(&pool[offset..(offset + height * width * pixelsize)])
        } else {
            // the buffer is discontinuous or lines overlap
            // we need to make a copy as unfortunately Glium does not
            // expose the OpenGL APIs we would need to load this buffer :/
            let mut data = Vec::with_capacity(height * width * pixelsize);
            for i in 0..height {
                data.extend(&pool[(offset + i * stride)..(offset + i * stride + width * pixelsize)]);
            }
            Cow::Owned(data)
        };

        // sharders format need to be reversed to account for endianness
        let (client_format, fragment) = match data.format {
            Format::Argb8888 => (ClientFormat::U8U8U8U8, crate::shaders::BUFFER_BGRA),
            Format::Xrgb8888 => (ClientFormat::U8U8U8U8, crate::shaders::BUFFER_BGRX),
            Format::Rgba8888 => (ClientFormat::U8U8U8U8, crate::shaders::BUFFER_ABGR),
            Format::Rgbx8888 => (ClientFormat::U8U8U8U8, crate::shaders::BUFFER_XBGR),
            _ => return Err(data.format),
        };
        Ok((
            RawImage2d {
                data: slice,
                width: width as u32,
                height: height as u32,
                format: client_format,
            },
            fragment,
        ))
    }

    #[cfg(feature = "__Unreachable__")]
    pub fn render_texture(
        &self,
        target: &mut vulkan::Frame,
        texture: &Texture2d,
        texture_kind: usize,
        y_inverted: bool,
        surface_dimensions: (u32, u32),
        surface_location: (i32, i32),
        screen_size: (u32, u32),
        blending: vulkan::Blend,
    ) {
        let xscale = 2.0 * (surface_dimensions.0 as f32) / (screen_size.0 as f32);
        let mut yscale = -2.0 * (surface_dimensions.1 as f32) / (screen_size.1 as f32);

        let x = 2.0 * (surface_location.0 as f32) / (screen_size.0 as f32) - 1.0;
        let mut y = 1.0 - 2.0 * (surface_location.1 as f32) / (screen_size.1 as f32);

        if y_inverted {
            yscale = -yscale;
            y -= surface_dimensions.1 as f32;
        }

        let uniforms = uniform! {
            matrix: [
                [xscale,   0.0  , 0.0, 0.0],
                [  0.0 , yscale , 0.0, 0.0],
                [  0.0 ,   0.0  , 1.0, 0.0],
                [   x  ,    y   , 0.0, 1.0]
            ],
            tex: texture,
        };

        target
            .draw(
                &self.vertex_buffer,
                &self.index_buffer,
                &self.programs[texture_kind],
                &uniforms,
                &vulkan::DrawParameters {
                    blend: blending,
                    ..Default::default()
                },
            )
            .unwrap();
    }
}

impl VulkanDrawer {
    fn draw_surface_tree(
        &self,
        root: &wl_surface::WlSurface,
        location: (i32, i32),
        compositor_token: MyCompositorToken,
    ) {
    }

    #[cfg(feature = "__Unreachable__")]
    fn draw_surface_tree(
        &self,
        frame: &mut Frame,
        root: &wl_surface::WlSurface,
        location: (i32, i32),
        compositor_token: MyCompositorToken,
        screen_dimensions: (u32, u32),
    ) {
        compositor_token
            .with_surface_tree_upward(root, location, |_surface, attributes, role, &(mut x, mut y)| {
                // there is actually something to draw !
                if attributes.user_data.texture.is_none() {
                    if let Some(buffer) = attributes.user_data.buffer.take() {
                        if let Ok(m) = self.texture_from_buffer(buffer.clone()) {
                            attributes.user_data.texture = Some(m);
                        }
                        // notify the client that we have finished reading the
                        // buffer
                        buffer.release();
                    }
                }
                if let Some(ref metadata) = attributes.user_data.texture {
                    if let Ok(subdata) = Role::<SubsurfaceRole>::data(role) {
                        x += subdata.location.0;
                        y += subdata.location.1;
                    }
                    self.render_texture(
                        frame,
                        &metadata.texture,
                        metadata.fragment,
                        metadata.y_inverted,
                        metadata.dimensions,
                        (x, y),
                        screen_dimensions,
                        ::vulkan::Blend {
                            color: ::vulkan::BlendingFunction::Addition {
                                source: ::vulkan::LinearBlendingFactor::One,
                                destination: ::vulkan::LinearBlendingFactor::OneMinusSourceAlpha,
                            },
                            alpha: ::vulkan::BlendingFunction::Addition {
                                source: ::vulkan::LinearBlendingFactor::One,
                                destination: ::vulkan::LinearBlendingFactor::OneMinusSourceAlpha,
                            },
                            ..Default::default()
                        },
                    );
                    TraversalAction::DoChildren((x, y))
                } else {
                    // we are not display, so our children are neither
                    TraversalAction::SkipChildren
                }
            })
            .unwrap();
    }

    pub fn draw_windows(&self, window_map: &MyWindowMap, compositor_token: MyCompositorToken) {
        // redraw the frame, in a simple but inneficient way
        {
            window_map.with_windows_from_bottom_to_top(|toplevel_surface, initial_place| {
                if let Some(wl_surface) = toplevel_surface.get_surface() {
                    // this surface is a root of a subsurface tree that needs to be drawn
                    self.draw_surface_tree(&wl_surface, initial_place, compositor_token);
                }
            });
        }
    }

    pub fn draw_cursor(&self, surface: &wl_surface::WlSurface, (x, y): (i32, i32), token: MyCompositorToken) {
        let (dx, dy) = match token.with_role_data::<CursorImageRole, _, _>(surface, |data| data.hotspot) {
            Ok(h) => h,
            Err(_) => {
                warn!(
                    self.log,
                    "Trying to display as a cursor a surface that does not have the CursorImage role."
                );
                (0, 0)
            }
        };
        self.draw_surface_tree(surface, (x - dx, y - dy), token);
    }

    pub fn draw_dnd_icon(
        &self,
        surface: &wl_surface::WlSurface,
        (x, y): (i32, i32),
        token: MyCompositorToken,
    ) {
        if !token.has_role::<DnDIconRole>(surface) {
            warn!(
                self.log,
                "Trying to display as a dnd icon a surface that does not have the DndIcon role."
            );
        }
        self.draw_surface_tree(surface, (x, y), token);
    }
}
