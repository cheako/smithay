use std::{
    cell::{Cell, RefCell},
    rc::Rc,
    time::{Duration, Instant},
};

use smithay::wayland::compositor::TraversalAction;

use crate::shell::{MyCompositorToken, MyWindowMap};
use vulkan_malloc::{Allocator, AllocatorMemoryRequirements, MemoryUsage};

use cgmath::{Matrix4, Point3};
use std::mem::{align_of, size_of};

use super::device::Device;

struct SwapchainSettings {
    swapchain: Rc<SwapchainKhr>,
    extent: Rc<Extent2D>,
    format: Rc<Format>,
}

struct NextImage {
    target: (Rc<Image>, Rc<ImageView>),
    framebuffer: Rc<Framebuffer>,
}

pub struct Display {
    pub image_avalible: Rc<Semaphore>,
    pub swapchain: Rc<SwapchainKhr>,
    pub presentation: u32,
    pub current_framebuffer: Cell<usize>,
    pub marker: Cell<Instant>,
    surface: Rc<SurfaceKhr>,
    extent: Rc<Extent2D>,
    refresh: Duration,
    next_images: Vec<Rc<NextImage>>,
    depth: (Rc<(Image, MappedMemoryRange)>, Rc<ImageView>),
    render_pass: Rc<RenderPass>,
    pipelines: Vec<Rc<Pipeline>>,
    uniform: Rc<(Buffer, MappedMemoryRange)>,
    uniform_map: Rc<MappedMemory>,
    uniform_matrix: *mut cgmath::Matrix4<f32>,
    format: Rc<Format>,
    descriptor_set: Rc<DescriptorSet>,
}

type UniformMatrix = Matrix4<f32>;

fn create_swapchain(
    device: &Device,
    surface: &SurfaceKhr,
    preferred_extent: Rc<Extent2D>,
) -> Result<SwapchainSettings, ()> {
    let capabilities = device
        .device_settings
        .physical_device
        .get_surface_capabilities_khr(surface)
        .map_err(|e| {
            println!("Failed to get surface capabilities ({})", e);
        })?;

    let min_image_count = match capabilities.max_image_count {
        Some(max_image_count) => {
            std::cmp::max(capabilities.min_image_count, std::cmp::min(3, max_image_count))
        }
        None => std::cmp::max(capabilities.min_image_count, 3),
    };

    let surface_formats: Vec<_> = device
        .device_settings
        .physical_device
        .get_surface_formats_khr(surface)
        .map_err(|e| {
            println!("Failed to get surface formats ({})", e);
        })?;

    let mut format = None;
    let mut color_space = None;
    for surface_format in surface_formats {
        if (surface_format.format == Format::B8G8R8A8_UNorm)
            && (surface_format.color_space == ColorSpaceKhr::SRGBNonLinear)
        {
            format = Some(surface_format.format);
            color_space = Some(surface_format.color_space);
            break;
        }
    }

    let format = Rc::new(format.ok_or_else(|| {
        println!("No suitable surface format found");
    })?);

    let (image_sharing_mode, queue_family_indices): (SharingMode, Vec<u32>) =
        if device.presentations.is_empty() {
            (SharingMode::Exclusive, vec![])
        } else {
            (
                SharingMode::Concurrent,
                device.presentations.iter().map(|(index, _)| *index).collect(),
            )
        };

    let extent = match capabilities.current_extent {
        Some(extent) => Rc::new(extent),
        None => preferred_extent.clone(),
    };

    let present_modes: Vec<_> = device
        .device_settings
        .physical_device
        .get_surface_present_modes_khr(surface)
        .map_err(|e| {
            println!("Failed to get surface present modes ({})", e);
        })?;

    let mut present_mode = None;
    for mode in present_modes {
        if mode == PresentModeKhr::Fifo {
            present_mode = Some(PresentModeKhr::Fifo);
            break;
        } else if mode == PresentModeKhr::Immediate {
            present_mode = Some(PresentModeKhr::Immediate);
        }
    }

    if present_mode.is_none() {
        println!("No suitable present mode found");
        return Err(());
    }

    let create_info = SwapchainCreateInfoKhr {
        flags: SwapchainCreateFlagsKhr::empty(),
        surface: (*surface).clone(),
        min_image_count: min_image_count,
        image_format: (*format).clone(),
        image_color_space: color_space.unwrap(),
        image_extent: (*extent).clone(),
        image_array_layers: 1,
        image_usage: ImageUsageFlags::COLOR_ATTACHMENT,
        image_sharing_mode: image_sharing_mode,
        queue_family_indices: queue_family_indices,
        pre_transform: capabilities.current_transform,
        composite_alpha: CompositeAlphaFlagBitsKhr::Opaque,
        present_mode: present_mode.unwrap(),
        clipped: true,
        old_swapchain: None,
        chain: None,
    };

    let swapchain = Rc::new(
        device
            .logical
            .create_swapchain_khr(&create_info, None)
            .map_err(|e| {
                println!("Failed to create swapchain ({})", e);
            })?,
    );

    Ok(SwapchainSettings {
        swapchain,
        extent,
        format,
    })
}

fn create_render_pass(device: &Device, format: &Format) -> Result<Rc<RenderPass>, ()> {
    let create_info = RenderPassCreateInfo {
        flags: RenderPassCreateFlags::empty(),
        attachments: vec![
            AttachmentDescription {
                flags: AttachmentDescriptionFlags::empty(),
                format: (*format).clone(),
                samples: SampleCountFlagBits::SampleCount1,
                load_op: AttachmentLoadOp::Clear,
                store_op: AttachmentStoreOp::Store,
                stencil_load_op: AttachmentLoadOp::DontCare,
                stencil_store_op: AttachmentStoreOp::DontCare,
                initial_layout: ImageLayout::Undefined,
                final_layout: ImageLayout::PresentSrcKhr,
            },
            AttachmentDescription {
                flags: AttachmentDescriptionFlags::empty(),
                format: Format::D32_SFloat_S8_UInt,
                samples: SampleCountFlagBits::SampleCount1,
                load_op: AttachmentLoadOp::Clear,
                store_op: AttachmentStoreOp::DontCare,
                stencil_load_op: AttachmentLoadOp::DontCare,
                stencil_store_op: AttachmentStoreOp::DontCare,
                initial_layout: ImageLayout::Undefined,
                final_layout: ImageLayout::DepthStencilAttachmentOptimal,
            },
        ],
        subpasses: vec![SubpassDescription {
            flags: SubpassDescriptionFlags::empty(),
            pipeline_bind_point: PipelineBindPoint::Graphics,
            input_attachments: vec![],
            color_attachments: vec![AttachmentReference {
                attachment: AttachmentIndex::Index(0),
                layout: ImageLayout::ColorAttachmentOptimal,
            }],
            resolve_attachments: vec![],
            depth_stencil_attachment: Some(AttachmentReference {
                attachment: AttachmentIndex::Index(1),
                layout: ImageLayout::DepthStencilAttachmentOptimal,
            }),
            preserve_attachments: vec![],
        }],
        dependencies: vec![],
        chain: None,
    };

    Ok(Rc::new(
        device
            .logical
            .create_render_pass(&create_info, None)
            .map_err(|e| {
                println!("Failed to create renderpass ({})", e);
            })?,
    ))
}

fn create_depth_image(
    device: &Device,
    extent: &Extent2D,
) -> Result<(Rc<(Image, MappedMemoryRange)>, Rc<ImageView>), ()> {
    let create_info = ImageCreateInfo {
        flags: ImageCreateFlags::empty(),
        image_type: ImageType::Type2D,
        format: Format::D32_SFloat_S8_UInt,
        extent: Extent3D {
            width: extent.width,
            height: extent.height,
            depth: 1,
        },
        mip_levels: 1,
        array_layers: 1,
        samples: SampleCountFlagBits::SampleCount1,
        tiling: ImageTiling::Optimal,
        usage: ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
        sharing_mode: SharingMode::Exclusive,
        queue_family_indices: vec![],
        initial_layout: ImageLayout::DepthStencilAttachmentOptimal,
        chain: None,
    };

    let reqs = AllocatorMemoryRequirements {
        own_memory: false,
        usage: MemoryUsage::GpuOnly,
        required_flags: MemoryPropertyFlags::empty(),
        preferred_flags: None,
        never_allocate: false,
    };

    let one = Rc::new(device.allocator.create_image(&create_info, &reqs).map_err(|e| {
        println!("Failed to create depth image ({})", e);
    })?);

    let create_info = ImageViewCreateInfo {
        flags: ImageViewCreateFlags::empty(),
        image: one.0.clone(),
        view_type: ImageViewType::Type2DArray,
        format: Format::D32_SFloat_S8_UInt,
        components: ComponentMapping::identity(),
        subresource_range: ImageSubresourceRange {
            aspect_mask: ImageAspectFlags::DEPTH | ImageAspectFlags::STENCIL,
            base_mip_level: 0,
            level_count: OptionalMipLevels::Remaining,
            base_array_layer: 0,
            layer_count: OptionalArrayLayers::Remaining,
        },
        chain: None,
    };

    let two = Rc::new(
        device
            .logical
            .create_image_view(&create_info, None)
            .map_err(|e| {
                println!("Failed to create depth image view ({})", e);
            })?,
    );

    Ok((one, two))
}

fn create_framebuffer(
    device: &Device,
    image_view: &ImageView,
    depth_view: &ImageView,
    render_pass: &RenderPass,
    extent: &Extent2D,
) -> Result<Rc<Framebuffer>, ()> {
    let create_info = FramebufferCreateInfo {
        flags: FramebufferCreateFlags::empty(),
        render_pass: (*render_pass).clone(),
        attachments: vec![(*image_view).clone(), (*depth_view).clone()],
        width: extent.width,
        height: extent.height,
        layers: 1,
        chain: None,
    };

    Ok(Rc::new(
        device
            .logical
            .create_framebuffer(&create_info, None)
            .map_err(|e| {
                println!("Failed to create framebuffer ({})", e);
            })?,
    ))
}

fn create_next_images(
    device: &Device,
    swapchain: &SwapchainKhr,
    extent: &Extent2D,
    format: &Format,
    depth_view: &ImageView,
    render_pass: &RenderPass,
) -> Result<Vec<Rc<NextImage>>, ()> {
    let images = swapchain.get_images_khr().map_err(|e| {
        println!("Failed to get swapchain images ({})", e);
    })?;

    images
        .into_iter()
        .map(|image| {
            let create_info = ImageViewCreateInfo {
                flags: ImageViewCreateFlags::empty(),
                image: image.clone(),
                view_type: ImageViewType::Type2D,
                format: format.clone(),
                components: ComponentMapping::identity(),
                subresource_range: ImageSubresourceRange {
                    aspect_mask: ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: OptionalMipLevels::MipLevels(1),
                    base_array_layer: 0,
                    layer_count: OptionalArrayLayers::ArrayLayers(1),
                },
                chain: None,
            };

            let image_view = Rc::new(
                device
                    .logical
                    .create_image_view(&create_info, None)
                    .map_err(|e| {
                        println!("Failed to create swapchain image view ({})", e);
                    })?,
            );

            Ok(Rc::new(NextImage {
                target: (Rc::new(image.clone()), image_view.clone()),
                framebuffer: create_framebuffer(&device, &image_view, &depth_view, &render_pass, &extent)?,
            }))
        })
        .collect()
}

fn create_pipelines(
    device: &Device,
    render_pass: &RenderPass,
    extent: &Extent2D,
) -> Result<Vec<Rc<Pipeline>>, ()> {
    let create_infos: Vec<GraphicsPipelineCreateInfo> = device
        .programs
        .1
        .iter()
        .map(|fragment_shader| GraphicsPipelineCreateInfo {
            flags: PipelineCreateFlags::empty(),
            stages: vec![
                PipelineShaderStageCreateInfo {
                    flags: PipelineShaderStageCreateFlags::empty(),
                    stage: ShaderStageFlagBits::Vertex,
                    module: (*device.programs.0).clone(),
                    name: "main".to_owned(),
                    specialization_info: None,
                    chain: None,
                },
                PipelineShaderStageCreateInfo {
                    flags: PipelineShaderStageCreateFlags::empty(),
                    stage: ShaderStageFlagBits::Fragment,
                    module: (**fragment_shader).clone(),
                    name: "main".to_owned(),
                    specialization_info: None,
                    chain: None,
                },
            ],
            vertex_input_state: PipelineVertexInputStateCreateInfo {
                flags: PipelineVertexInputStateCreateFlags::empty(),
                vertex_binding_descriptions: vec![],
                vertex_attribute_descriptions: vec![],
                chain: None,
            },
            input_assembly_state: PipelineInputAssemblyStateCreateInfo {
                flags: PipelineInputAssemblyStateCreateFlags::empty(),
                topology: PrimitiveTopology::TriangleList,
                primitive_restart_enable: false,
                chain: None,
            },
            tessellation_state: None,
            viewport_state: Some(PipelineViewportStateCreateInfo {
                flags: PipelineViewportStateCreateFlags::empty(),
                viewports: vec![Viewport {
                    x: 0.0,
                    y: 0.0,
                    width: extent.width as f32,
                    height: extent.height as f32,
                    min_depth: 0.0,
                    max_depth: 1.0,
                }],
                scissors: vec![Rect2D::new(Offset2D::zero(), *extent)],
                chain: None,
            }),
            rasterization_state: PipelineRasterizationStateCreateInfo {
                flags: PipelineRasterizationStateCreateFlags::empty(),
                depth_clamp_enable: false,
                rasterizer_discard_enable: false,
                polygon_mode: PolygonMode::Fill,
                cull_mode: CullModeFlags::NONE,
                front_face: FrontFace::Clockwise,
                depth_bias_enable: false,
                depth_bias_constant_factor: 0.0,
                depth_bias_clamp: 0.0,
                depth_bias_slope_factor: 0.0,
                line_width: 1.0,
                chain: None,
            },
            multisample_state: Some(PipelineMultisampleStateCreateInfo {
                flags: PipelineMultisampleStateCreateFlags::empty(),
                rasterization_samples: SampleCountFlagBits::SampleCount1,
                sample_shading_enable: false,
                min_sample_shading: 1.0,
                sample_mask: vec![],
                alpha_to_coverage_enable: false,
                alpha_to_one_enable: false,
                chain: None,
            }),
            depth_stencil_state: None,
            color_blend_state: Some(PipelineColorBlendStateCreateInfo {
                flags: PipelineColorBlendStateCreateFlags::empty(),
                logic_op_enable: false,
                logic_op: LogicOp::Copy,
                attachments: vec![PipelineColorBlendAttachmentState {
                    blend_enable: false,
                    src_color_blend_factor: BlendFactor::One,
                    dst_color_blend_factor: BlendFactor::Zero,
                    color_blend_op: BlendOp::Add,
                    src_alpha_blend_factor: BlendFactor::One,
                    dst_alpha_blend_factor: BlendFactor::Zero,
                    alpha_blend_op: BlendOp::Add,
                    color_write_mask: ColorComponentFlags::R
                        | ColorComponentFlags::G
                        | ColorComponentFlags::B,
                }],
                blend_constants: [0.0, 0.0, 0.0, 0.0],
                chain: None,
            }),
            dynamic_state: None,
            layout: (*device.pipeline.1).clone(),
            render_pass: (*render_pass).clone(),
            subpass: 0,
            base_pipeline: None,
            base_pipeline_index: None,
            chain: None,
        })
        .collect();

    Ok(device
        .logical
        .create_graphics_pipelines(None, &create_infos, None)
        .map_err(|(e, _)| {
            println!("Failed to create pipelines ({})", e);
        })?
        .iter()
        .map(|pipeline| Rc::new(pipeline.clone()))
        .collect())
}

fn create_uniform(device: &Device) -> Result<(Rc<(Buffer, MappedMemoryRange)>), ()> {
    let create_info = BufferCreateInfo {
        flags: BufferCreateFlags::empty(),
        size: size_of::<UniformMatrix>() as _,
        usage: BufferUsageFlags::UNIFORM_BUFFER,
        sharing_mode: SharingMode::Exclusive,
        queue_family_indices: vec![device.graphics.0],
        chain: None,
    };
    let reqs = AllocatorMemoryRequirements {
        own_memory: false,
        usage: MemoryUsage::CpuToGpu,
        required_flags: MemoryPropertyFlags::empty(),
        preferred_flags: None,
        never_allocate: false,
    };
    Ok(Rc::new(
        device.allocator.create_buffer(&create_info, &reqs).map_err(|e| {
            println!("Failed to create display uniform buffer ({})", e);
        })?,
    ))
}

fn create_descriptorset(device: &Device) -> Result<(Rc<DescriptorSet>), ()> {
    let alloc_info = DescriptorSetAllocateInfo {
        descriptor_pool: (*device.descriptors.0).clone(),
        set_layouts: vec![(*device.descriptors.1[1]).clone()],
        chain: None,
    };
    Ok(Rc::new(
        DescriptorPool::allocate_descriptor_sets(&alloc_info).map_err(|e| {
            println!("Failed to allocate display descriptor set ({})", e);
        })?[0]
            .clone(),
    ))
}

impl Display {
    pub fn init(
        device: &Device,
        surface: Rc<SurfaceKhr>,
        extent: Rc<Extent2D>,
        refresh: Duration,
        marker: Instant,
        presentation: u32,
    ) -> Result<Rc<Display>, ()> {
        let SwapchainSettings {
            swapchain,
            extent,
            format,
        } = create_swapchain(&device, &surface, extent)?;
        let image_avalible = device.create_semaphore()?;
        let render_pass = create_render_pass(&device, &format)?;
        let depth = create_depth_image(&device, &extent)?;
        let next_images = create_next_images(&device, &swapchain, &extent, &format, &depth.1, &render_pass)?;
        let pipelines = create_pipelines(&device, &render_pass, &extent)?;
        let uniform = create_uniform(&device)?;
        let uniform_map = Allocator::map_memory(&uniform.1)
            .map(|v| Rc::new(v))
            .map_err(|e| ())?;
        let uniform_matrix = uniform_map.as_ptr() as *mut cgmath::Matrix4<f32>;
        let descriptor_set = create_descriptorset(&device)?;

        /* Start with OpenGL NDC. */
        let proj_matrix = cgmath::ortho(0.0, extent.width as f32, 0.0, extent.height as f32, 1.0, -1.0);
        let view_matrix = cgmath::Matrix4::look_at(
            Point3::new((extent.width >> 1) as f32, (extent.height >> 1) as f32, 1.0),
            Point3::new((extent.width >> 1) as f32, (extent.height >> 1) as f32, 0.0),
            cgmath::Vector3::new(0.0, 1.0, 0.0),
        );

        /* Potentually adjust for Vulkan NDC. */
        let pv_matrix = /* Matrix4::new(
            1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 1.0,
        ) */ proj_matrix
            * view_matrix;
        unsafe { *uniform_matrix = pv_matrix };
        uniform_map.flush(&None);

        let writes = vec![WriteDescriptorSet {
            dst_set: (*descriptor_set).clone(),
            dst_binding: 0,
            dst_array_element: 0,
            descriptor_type: DescriptorType::UniformBuffer,
            elements: WriteDescriptorSetElements::BufferInfo(vec![DescriptorBufferInfo {
                buffer: uniform.0.clone(),
                offset: uniform.1.offset,
                range: uniform.1.size,
            }]),
            chain: None,
        }];

        DescriptorSet::update(Some(&writes), None);

        Ok(Rc::new(Display {
            image_avalible,
            swapchain,
            presentation,
            current_framebuffer: Cell::new(0),
            marker: Cell::new(marker),
            surface,
            extent,
            refresh,
            next_images,
            depth,
            render_pass,
            pipelines,
            uniform,
            uniform_map,
            uniform_matrix,
            format,
            descriptor_set,
        }))
    }
}

fn begin_render_pass(
    command_buffer: &CommandBuffer,
    render_pass: &RenderPass,
    framebuffer: &Framebuffer,
    extent: &Extent2D,
) {
    let clear_values = vec![
        ClearValue::Color(ClearColorValue::Float32([0.0, 0.0, 0.0, 0.0])),
        ClearValue::DepthStencil(ClearDepthStencilValue {
            depth: 1.0,
            stencil: 0,
        }),
    ];

    let render_pass_begin = RenderPassBeginInfo {
        render_pass: render_pass.clone(),
        framebuffer: framebuffer.clone(),
        render_area: Rect2D {
            offset: Offset2D { x: 0, y: 0 },
            extent: extent.clone(),
        },
        clear_values,
        chain: None,
    };

    //        m_commandBuffer.beginRenderPass(renderPassBegin, vk::SubpassContents::eInline);
    command_buffer.begin_render_pass(&render_pass_begin, SubpassContents::Inline);
}

impl Display {
    pub fn tick(
        &self,
        device: &Device,
        command_buffer: &CommandBuffer,
        id: &u64,
        window_map: &RefCell<MyWindowMap>,
        compositor_token: MyCompositorToken,
    ) -> Result<Duration, ()> {
        self.current_framebuffer.set(match self
            .swapchain
            .acquire_next_image_khr(Timeout::Infinite, Some(&self.image_avalible), None)
            .map_err(|e| ())?
        {
            AcquireNextImageResultKhr::Timeout | AcquireNextImageResultKhr::NotReady => Err(()),
            AcquireNextImageResultKhr::Index(index) | AcquireNextImageResultKhr::Suboptimal(index) => {
                Ok(index)
            }
        }?);
        let next_image = &self.next_images[self.current_framebuffer.get()];

        begin_render_pass(
            &command_buffer,
            &self.render_pass,
            &next_image.framebuffer,
            &self.extent,
        );

        command_buffer.bind_descriptor_sets(
            PipelineBindPoint::Graphics,
            &device.pipeline.1,
            1,
            &[(*self.descriptor_set).clone()],
            None,
        );

        let mut current_fragment = 255;
        window_map
            .borrow()
            .with_windows_from_bottom_to_top(|toplevel_surface, initial_place| {
                if let Some(wl_surface) = toplevel_surface.get_surface() {
                    compositor_token
                        .with_surface_tree_upward(
                            wl_surface,
                            initial_place,
                            |_surface, attributes, _role, &(mut _x, mut _y)| {
                                if let Some(textures) = &attributes.user_data.textures {
                                    let textures = textures.borrow();
                                    let metadata = textures[id].borrow();

                                    if metadata.fragment != current_fragment {
                                        command_buffer.bind_pipeline(
                                            PipelineBindPoint::Graphics,
                                            &self.pipelines[metadata.fragment],
                                        );
                                        current_fragment = metadata.fragment;
                                    }

                                    command_buffer.bind_descriptor_sets(
                                        PipelineBindPoint::Graphics,
                                        &device.pipeline.1,
                                        0,
                                        &[(*metadata.descriptor_set).clone()],
                                        None,
                                    );

                                    command_buffer.draw(6, 1, 0, 0);
                                }
                                TraversalAction::SkipChildren
                            },
                        )
                        .expect("Some thing now");
                }
            });

        command_buffer.end_render_pass();

        Ok(self.refresh)
    }
}
