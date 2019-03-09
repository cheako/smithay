use std::{
    cell::{Ref, RefCell},
    collections::{hash_map::Entry, HashMap},
    rc::Rc,
    slice::from_raw_parts_mut,
    time::{Duration, Instant},
};

use dacite::{
    core::{
        AccessFlags, BorderColor, Buffer, BufferCreateFlags, BufferCreateInfo, BufferImageCopy,
        BufferUsageFlags, CommandBuffer, CommandBufferAllocateInfo, CommandBufferBeginInfo,
        CommandBufferResetFlags, CommandBufferUsageFlags, CommandPool, CommandPoolCreateFlags,
        CommandPoolCreateInfo, CompareOp, ComponentMapping, DependencyFlags, DescriptorBufferInfo,
        DescriptorImageInfo, DescriptorPool, DescriptorPoolCreateFlags, DescriptorPoolCreateInfo,
        DescriptorPoolSize, DescriptorSet, DescriptorSetAllocateInfo, DescriptorSetLayout,
        DescriptorSetLayoutBinding, DescriptorSetLayoutCreateFlags, DescriptorSetLayoutCreateInfo,
        DescriptorType, Device as VkDevice, DeviceCreateFlags, DeviceCreateInfo, DeviceExtensions,
        DeviceQueueCreateFlags, DeviceQueueCreateInfo, Extent3D, Fence, FenceCreateFlags, FenceCreateInfo,
        Filter, Format, Image, ImageAspectFlags, ImageCreateFlags, ImageCreateInfo, ImageLayout,
        ImageMemoryBarrier, ImageSubresourceLayers, ImageSubresourceRange, ImageTiling, ImageType,
        ImageUsageFlags, ImageView, ImageViewCreateFlags, ImageViewCreateInfo, ImageViewType,
        MappedMemoryRange, MemoryPropertyFlags, Offset3D, OptionalArrayLayers, OptionalMipLevels,
        PhysicalDevice, Pipeline, PipelineCache, PipelineCacheCreateFlags, PipelineCacheCreateInfo,
        PipelineLayout, PipelineLayoutCreateFlags, PipelineLayoutCreateInfo, PipelineStageFlags, Queue,
        QueueFamilyIndex, QueueFlags, SampleCountFlagBits, Sampler, SamplerAddressMode, SamplerCreateFlags,
        SamplerCreateInfo, SamplerMipmapMode, Semaphore, SemaphoreCreateFlags, SemaphoreCreateInfo,
        ShaderModule, ShaderModuleCreateFlags, ShaderModuleCreateInfo, ShaderStageFlags, SharingMode,
        SubmitInfo, Timeout, WriteDescriptorSet, WriteDescriptorSetElements,
    },
    khr_swapchain::PresentInfoKhr,
};
use smithay::{
    reexports::wayland_server::protocol::wl_buffer::WlBuffer,
    wayland::{
        compositor::TraversalAction,
        shm::{with_buffer_contents as shm_buffer_contents, BufferData},
    },
};

use vulkan_malloc::{Allocator, AllocatorMemoryRequirements, MemoryUsage};

use super::{display::Display, DeviceSettings, DisplayProber, TextureMetadata};
use crate::shell::{MyCompositorToken, MyWindowMap};

const TWOKILO_TEXTURE: u32 = 1;

pub struct Device {
    pub device_settings: Rc<DeviceSettings>,
    pub logical: Rc<VkDevice>,
    pub displays: RefCell<HashMap<u64, Rc<Display>>>,
    pub programs: (Rc<ShaderModule>, Vec<Rc<ShaderModule>>),
    pub descriptors: (Rc<DescriptorPool>, [Rc<DescriptorSetLayout>; 2]),
    pub pipeline: (Rc<PipelineCache>, Rc<PipelineLayout>),
    pub allocator: Rc<Allocator>,
    // family_index: QueueFamilyIndex,
    pub graphics: (u32, Rc<Queue>, Rc<CommandPool>, Rc<CommandBuffer>),
    pub transfer: Option<(Rc<Queue>, Rc<CommandPool>, Rc<CommandBuffer>)>,
    pub presentations: Vec<(u32, Rc<Queue>)>,
    // command_pool: CommandPool,
    // transfer_pool: Option<CommandPool>,
    fence: Rc<Fence>,
    render_done: Rc<Semaphore>,
    sampler: Rc<Sampler>,
    // swapchain: SwapchainKhr,
    // extent: Extent2D,
    // format: Format,
    // render_pass: RenderPass,
    // framebuffers: [Framebuffer; 3],
    // command_pool: CommandPool,
}

struct QueueFamilyIndices {
    graphics: u32,
    transfer: Option<u32>,
    presentations: Vec<u32>,
}

fn find_queue_family_index(physical_device: &PhysicalDevice) -> Result<QueueFamilyIndices, ()> {
    let mut graphics = None;
    let mut transfer = None;
    let mut presentations = vec![];

    let queue_family_properties: Vec<_> = physical_device.get_queue_family_properties();
    for (index, queue_family_properties) in queue_family_properties.into_iter().enumerate() {
        if queue_family_properties.queue_count == 0 {
            continue;
        }

        if graphics.is_none() && queue_family_properties.queue_flags.contains(QueueFlags::GRAPHICS) {
            graphics = Some(index as u32);
        }

        if transfer.is_none()
            && queue_family_properties.queue_flags.contains(QueueFlags::TRANSFER)
            && !queue_family_properties.queue_flags.contains(QueueFlags::GRAPHICS)
            && !queue_family_properties.queue_flags.contains(QueueFlags::COMPUTE)
        {
            transfer = Some(index as u32);
        }

        if queue_family_properties.queue_flags.contains(QueueFlags::TRANSFER) {
            presentations.push(index as u32);
        }
    }

    if let Some(graphics) = graphics {
        // Disregard a single presentations queue family.
        if presentations == vec![graphics] {
            presentations = vec![];
        };

        Ok(QueueFamilyIndices {
            graphics,
            transfer,
            presentations,
        })
    } else {
        Err(())
    }
}

fn create_device(
    physical_device: &PhysicalDevice,
    extensions: &DeviceExtensions,
    indices: &QueueFamilyIndices,
) -> Result<Rc<VkDevice>, ()> {
    let mut device_queue_create_infos = vec![DeviceQueueCreateInfo {
        flags: DeviceQueueCreateFlags::empty(),
        queue_family_index: indices.graphics,
        queue_priorities: vec![1.0],
        chain: None,
    }];

    match indices.transfer {
        Some(index) => device_queue_create_infos.push(DeviceQueueCreateInfo {
            flags: DeviceQueueCreateFlags::empty(),
            queue_family_index: index,
            queue_priorities: vec![1.0],
            chain: None,
        }),
        None => (),
    }

    for index in &indices.presentations {
        device_queue_create_infos.push(DeviceQueueCreateInfo {
            flags: DeviceQueueCreateFlags::empty(),
            queue_family_index: *index,
            queue_priorities: vec![1.0],
            chain: None,
        });
    }

    let create_info = DeviceCreateInfo {
        flags: DeviceCreateFlags::empty(),
        queue_create_infos: device_queue_create_infos,
        enabled_layers: vec![],
        enabled_extensions: (*extensions).clone(),
        enabled_features: None,
        chain: None,
    };
    Ok(Rc::new(
        physical_device.create_device(&create_info, None).map_err(|e| {
            println!("Failed to create device ({})", e);
        })?,
    ))
}

fn create_command_pool(logical: &VkDevice, queue_family_index: u32) -> Result<Rc<CommandPool>, ()> {
    let create_info = CommandPoolCreateInfo {
        flags: CommandPoolCreateFlags::TRANSIENT | CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
        queue_family_index: queue_family_index,
        chain: None,
    };

    Ok(Rc::new(logical.create_command_pool(&create_info, None).map_err(
        |e| {
            println!("Failed to create command pool ({})", e);
        },
    )?))
}

fn create_command_buffer(command_pool: &CommandPool) -> Result<Rc<CommandBuffer>, ()> {
    let allocate_info = CommandBufferAllocateInfo {
        command_pool: (*command_pool).clone(),
        level: dacite::core::CommandBufferLevel::Primary,
        command_buffer_count: 1 as u32,
        chain: None,
    };

    Ok(Rc::new(
        CommandPool::allocate_command_buffers(&allocate_info).map_err(|e| {
            println!("Failed to allocate command buffers ({})", e);
        })?[0]
            .clone(),
    ))
}

fn create_vertex_shader(logical: &VkDevice) -> Result<Rc<ShaderModule>, ()> {
    let vertex_shader_bytes = glsl_vs! { r#"#version 450

out gl_PerVertex {
    vec4 gl_Position;
};
layout( location = 0 ) out vec2 outUV;

layout( set = 0, binding = 0 ) uniform UBO {
    mat4 Model;
} m;
layout( set = 1, binding = 0 ) uniform SUBO {
    mat4 ViewProj;
} vp;

vec2 positions[6] = vec2[](
    vec2(0.0, 1.0),
    vec2(0.0, 0.0),
    vec2(1.0, 0.0),

    vec2(1.0, 0.0),
    vec2(1.0, 1.0),
    vec2(0.0, 1.0)
);
vec2 UV[6] = vec2[](
    vec2(1.0, 1.0),
    vec2(1.0, 0.0),
    vec2(0.0, 0.0),

    vec2(0.0, 0.0),
    vec2(0.0, 1.0),
    vec2(1.0, 1.0)
);

void main() {
    gl_Position = 
          vp.ViewProj
        * m.Model
	* vec4(positions[gl_VertexIndex], 0.0, 1.0);
    outUV = UV[gl_VertexIndex];
}"#
    };

    let create_info = ShaderModuleCreateInfo {
        flags: ShaderModuleCreateFlags::empty(),
        code: vertex_shader_bytes.to_vec(),
        chain: None,
    };

    Ok(Rc::new(
        logical.create_shader_module(&create_info, None).map_err(|e| {
            println!("Failed to create vertex shader module ({})", e);
        })?,
    ))
}

pub mod shader {
    pub const FRAGMENT_COUNT: usize = 7;
    pub const BUFFER_UNKNOWN: usize = 1;
    pub const BUFFER_ARGB: usize = 0;
    pub const BUFFER_XRGB: usize = 1;
    pub const BUFFER_RGBA: usize = 2;
    pub const BUFFER_ABGR: usize = 3;
    pub const BUFFER_XBGR: usize = 4;
    pub const BUFFER_BGRA: usize = 5;
    pub const BUFFER_BGRX: usize = 6;
}
use shader::*;

fn get_format(data: &BufferData) -> (Format, usize, usize) {
    match data.format {
        Argb8888 => (Format::R8G8B8A8_UNorm, BUFFER_ARGB, 1),
        Xrgb8888 => (Format::R8G8B8A8_UNorm, BUFFER_XRGB, 1),
        _ => (Format::R8G8B8A8_UNorm, BUFFER_UNKNOWN, 1),
    }
}

fn create_fragment_shaders(logical: &VkDevice) -> Result<Vec<Rc<ShaderModule>>, ()> {
    let fragment_shaders_bytes: [&[u8]; FRAGMENT_COUNT] = [
        glsl_fs! {r#"#version 450
layout( set = 0, binding = 1 ) uniform sampler2D tex;
layout( location = 0 ) in vec2 v_tex_coords;
layout( location = 0 ) out vec4 outColor; 
void main() {
    vec4 color = texture(tex, v_tex_coords);
    outColor.r = color.y;
    outColor.g = color.z;
    outColor.b = color.w;
    outColor.a = color.x;
}"# },
        glsl_fs! {r#"#version 450
layout( set = 0, binding = 1 ) uniform sampler2D tex;
layout( location = 0 ) in vec2 v_tex_coords;
layout( location = 0 ) out vec4 outColor; 
void main() {
    vec4 color = texture(tex, v_tex_coords);
    outColor.r = color.y;
    outColor.g = color.z;
    outColor.b = color.w;
    outColor.a = 1.0;
}"# },
        glsl_fs! {r#"#version 450
layout( set = 0, binding = 1 ) uniform sampler2D tex;
layout( location = 0 ) in vec2 v_tex_coords;
layout( location = 0 ) out vec4 outColor; 
void main() {
    vec4 color = texture(tex, v_tex_coords);
    outColor.r = color.x;
    outColor.g = color.y;
    outColor.b = color.z;
    outColor.a = color.w;
}"# },
        glsl_fs! {r#"#version 450
layout( set = 0, binding = 1 ) uniform sampler2D tex;
layout( location = 0 ) in vec2 v_tex_coords;
layout( location = 0 ) out vec4 outColor; 
void main() {
    vec4 color = texture(tex, v_tex_coords);
    outColor.r = color.w;
    outColor.g = color.z;
    outColor.b = color.y;
    outColor.a = color.x;
}"# },
        glsl_fs! {r#"#version 450
layout( set = 0, binding = 1 ) uniform sampler2D tex;
layout( location = 0 ) in vec2 v_tex_coords;
layout( location = 0 ) out vec4 outColor; 
void main() {
    vec4 color = texture(tex, v_tex_coords);
    outColor.r = color.w;
    outColor.g = color.z;
    outColor.b = color.y;
    outColor.a = 1.0;
}"# },
        glsl_fs! {r#"#version 450
layout( set = 0, binding = 1 ) uniform sampler2D tex;
layout( location = 0 ) in vec2 v_tex_coords;
layout( location = 0 ) out vec4 outColor; 
void main() {
    vec4 color = texture(tex, v_tex_coords);
    outColor.r = color.z;
    outColor.g = color.y;
    outColor.b = color.x;
    outColor.a = color.w;
}"# },
        glsl_fs! {r#"#version 450
layout( set = 0, binding = 1 ) uniform sampler2D tex;
layout( location = 0 ) in vec2 v_tex_coords;
layout( location = 0 ) out vec4 outColor; 
void main() {
    vec4 color = texture(tex, v_tex_coords);
    outColor.r = color.z;
    outColor.g = color.y;
    outColor.b = color.x;
    outColor.a = 1.0;
}"# },
    ];

    let mut create_info = ShaderModuleCreateInfo {
        flags: ShaderModuleCreateFlags::empty(),
        code: Vec::new(),
        chain: None,
    };

    let mut ret: Vec<Rc<ShaderModule>> = Vec::with_capacity(FRAGMENT_COUNT);
    for (i, bytes) in fragment_shaders_bytes.iter().enumerate() {
        create_info.code = bytes.to_vec();
        ret.push(Rc::new(
            logical.create_shader_module(&create_info, None).map_err(|e| {
                println!("Failed to create fragment shader module #{} ({})", i, e);
            })?,
        ));
    }
    Ok(ret)
}

fn create_descriptor_pool(logical: &VkDevice) -> Result<Rc<DescriptorPool>, ()> {
    let create_info = DescriptorPoolCreateInfo {
        flags: DescriptorPoolCreateFlags::empty(),
        pool_sizes: vec![
            DescriptorPoolSize {
                descriptor_count: 2048 * TWOKILO_TEXTURE,
                descriptor_type: DescriptorType::CombinedImageSampler,
            },
            DescriptorPoolSize {
                descriptor_count: 4096 * TWOKILO_TEXTURE,
                descriptor_type: DescriptorType::UniformBuffer,
            },
        ],
        max_sets: 4096 * TWOKILO_TEXTURE,
        chain: None,
    };

    Ok(Rc::new(
        logical.create_descriptor_pool(&create_info, None).map_err(|e| {
            println!("Failed to create descriptor pool ({})", e);
        })?,
    ))
}

fn create_descriptor_set_layouts(logical: &VkDevice) -> Result<[Rc<DescriptorSetLayout>; 2], ()> {
    let create_info = DescriptorSetLayoutCreateInfo {
        chain: None,
        bindings: vec![
            DescriptorSetLayoutBinding {
                binding: 0,
                descriptor_count: 1,
                descriptor_type: DescriptorType::UniformBuffer,
                stage_flags: ShaderStageFlags::VERTEX,
                immutable_samplers: vec![],
            },
            DescriptorSetLayoutBinding {
                binding: 1,
                descriptor_count: 1,
                descriptor_type: DescriptorType::CombinedImageSampler,
                stage_flags: ShaderStageFlags::FRAGMENT,
                immutable_samplers: vec![],
            },
        ],
        flags: DescriptorSetLayoutCreateFlags::empty(),
    };

    let ubo = Rc::new(
        logical
            .create_descriptor_set_layout(&create_info, None)
            .map_err(|e| {
                println!("Failed to create descriptor set layout ({})", e);
            })?,
    );

    let create_info = DescriptorSetLayoutCreateInfo {
        chain: None,
        bindings: vec![DescriptorSetLayoutBinding {
            binding: 0,
            descriptor_count: 1,
            descriptor_type: DescriptorType::UniformBuffer,
            stage_flags: ShaderStageFlags::VERTEX,
            immutable_samplers: vec![],
        }],
        flags: DescriptorSetLayoutCreateFlags::empty(),
    };

    let subo = Rc::new(
        logical
            .create_descriptor_set_layout(&create_info, None)
            .map_err(|e| {
                println!("Failed to create shared descriptor set layout shared ({})", e);
            })?,
    );

    Ok([ubo, subo])
}

fn create_pipeline_cache(logical: &VkDevice) -> Result<Rc<PipelineCache>, ()> {
    let create_info = PipelineCacheCreateInfo {
        flags: PipelineCacheCreateFlags::empty(),
        initial_data: vec![],
        chain: None,
    };

    Ok(Rc::new(
        logical.create_pipeline_cache(&create_info, None).map_err(|e| {
            println!("Failed to create pipeline cache ({})", e);
        })?,
    ))
}

fn create_pipeline_layout(
    logical: &VkDevice,
    set_layouts: &[Rc<DescriptorSetLayout>; 2],
) -> Result<Rc<PipelineLayout>, ()> {
    let create_info = PipelineLayoutCreateInfo {
        flags: PipelineLayoutCreateFlags::empty(),
        set_layouts: set_layouts
            .iter()
            .map(|set_layout| (**set_layout).clone())
            .collect(),
        push_constant_ranges: vec![],
        chain: None,
    };

    Ok(Rc::new(
        logical.create_pipeline_layout(&create_info, None).map_err(|e| {
            println!("Failed to create pipeline layout ({})", e);
        })?,
    ))
}

fn create_fence(logical: &VkDevice) -> Result<Rc<Fence>, ()> {
    let create_info = FenceCreateInfo {
        flags: FenceCreateFlags::empty(),
        chain: None,
    };

    Ok(Rc::new(logical.create_fence(&create_info, None).map_err(
        |e| {
            println!("Failed to create fence ({})", e);
        },
    )?))
}

fn create_semaphore(logical: &VkDevice) -> Result<Rc<Semaphore>, ()> {
    let create_info = SemaphoreCreateInfo {
        flags: SemaphoreCreateFlags::empty(),
        chain: None,
    };

    Ok(Rc::new(logical.create_semaphore(&create_info, None).map_err(
        |e| {
            println!("Failed to create semaphore ({})", e);
        },
    )?))
}

fn create_sampler(logical: &VkDevice) -> Result<Rc<Sampler>, ()> {
    let create_info = SamplerCreateInfo {
        flags: SamplerCreateFlags::empty(),
        mag_filter: Filter::Nearest,
        min_filter: Filter::Nearest,
        mipmap_mode: SamplerMipmapMode::Nearest,
        address_mode_u: SamplerAddressMode::ClampToEdge,
        address_mode_v: SamplerAddressMode::ClampToEdge,
        address_mode_w: SamplerAddressMode::ClampToEdge,
        mip_lod_bias: 0.0,
        anisotropy_enable: false,
        max_anisotropy: 0.0,
        compare_enable: false,
        compare_op: CompareOp::Always,
        min_lod: 0.0,
        max_lod: 0.0,
        border_color: BorderColor::FloatTransparentBlack,
        unnormalized_coordinates: false,
        chain: None,
    };

    Ok(Rc::new(logical.create_sampler(&create_info, None).map_err(
        |e| {
            println!("Failed to create sampler ({})", e);
        },
    )?))
}

trait BufferSize {
    fn buffer_size(&self) -> u64;
}

impl BufferSize for BufferData {
    fn buffer_size(&self) -> u64 {
        self.height as u64 * self.stride as u64 * get_format(&self).2 as u64
    }
}

impl Device {
    pub fn create_semaphore(&self) -> Result<Rc<Semaphore>, ()> {
        create_semaphore(&self.logical)
    }

    pub fn init(device_settings: Rc<DeviceSettings>) -> Result<Device, ()> {
        let indices = find_queue_family_index(&device_settings.physical_device)?;

        let logical = create_device(
            &device_settings.physical_device,
            &device_settings.device_extensions,
            &indices,
        )?;

        let vertex_shader = create_vertex_shader(&logical)?;
        let fragment_shaders = create_fragment_shaders(&logical)?;
        let programs = (vertex_shader, fragment_shaders);

        let descriptor_pool = create_descriptor_pool(&logical)?;
        let descriptor_set_layouts = create_descriptor_set_layouts(&logical)?;
        let descriptors = (descriptor_pool, descriptor_set_layouts.clone());

        let pipeline_cache = create_pipeline_cache(&logical)?;
        let pipeline_layout = create_pipeline_layout(&logical, &descriptor_set_layouts)?;
        let pipeline = (pipeline_cache, pipeline_layout);

        let allocator = Rc::new(
            Allocator::builder((*logical).clone(), (*device_settings.physical_device).clone()).build(),
        );

        let command_pool = create_command_pool(&logical, indices.graphics)?;
        let graphics = (
            indices.graphics,
            Rc::new(logical.get_queue(indices.graphics, 0)),
            command_pool.clone(),
            create_command_buffer(&command_pool)?,
        );

        let transfer = match indices.transfer {
            Some(index) => {
                let command_pool = create_command_pool(&logical, index)?;
                Some((
                    Rc::new(logical.get_queue(index, 0)),
                    command_pool.clone(),
                    create_command_buffer(&command_pool)?,
                ))
            }
            None => None,
        };

        let presentations = indices
            .presentations
            .iter()
            .map(|index| (index.clone(), Rc::new(logical.get_queue(index.clone(), 0))))
            .collect();

        let fence = create_fence(&logical)?;

        let render_done = create_semaphore(&logical)?;

        let sampler = create_sampler(&logical)?;

        Ok(Device {
            device_settings,
            logical,
            displays: RefCell::new(HashMap::new()),
            programs,
            descriptors,
            pipeline,
            allocator,
            graphics,
            transfer,
            presentations,
            fence,
            render_done,
            sampler,
        })
    }

    fn create_shm_input(&self, size: u64) -> Result<(Buffer, MappedMemoryRange), ()> {
        let create_info = BufferCreateInfo {
            flags: BufferCreateFlags::empty(),
            size,
            usage: BufferUsageFlags::TRANSFER_SRC,
            sharing_mode: SharingMode::Exclusive,
            queue_family_indices: vec![],
            chain: None,
        };

        let reqs = AllocatorMemoryRequirements {
            own_memory: false,
            usage: MemoryUsage::CpuToGpu,
            required_flags: MemoryPropertyFlags::empty(),
            preferred_flags: None,
            never_allocate: false,
        };

        self.allocator.create_buffer(&create_info, &reqs).map_err(|e| ())
    }

    fn create_texture_image(&self, data: &BufferData) -> Result<Rc<(Image, MappedMemoryRange)>, ()> {
        let create_info = ImageCreateInfo {
            flags: ImageCreateFlags::empty(),
            image_type: ImageType::Type2D,
            format: get_format(&data).0,
            extent: Extent3D {
                width: data.width as _,
                height: data.height as _,
                depth: 1,
            },
            mip_levels: 1,
            array_layers: 1,
            samples: SampleCountFlagBits::SampleCount1,
            tiling: ImageTiling::Optimal,
            usage: ImageUsageFlags::TRANSFER_DST | ImageUsageFlags::SAMPLED,
            sharing_mode: SharingMode::Exclusive,
            queue_family_indices: vec![],
            initial_layout: ImageLayout::TransferDstOptimal,
            chain: None,
        };

        let reqs = AllocatorMemoryRequirements {
            own_memory: false,
            usage: MemoryUsage::GpuOnly,
            required_flags: MemoryPropertyFlags::empty(),
            preferred_flags: None,
            never_allocate: false,
        };

        self.allocator
            .create_image(&create_info, &reqs)
            .map(|v| Rc::new(v))
            .map_err(|e| ())
    }

    fn create_texture_view(&self, data: &BufferData, image: &Image) -> Result<Rc<ImageView>, ()> {
        let create_info = ImageViewCreateInfo {
            flags: ImageViewCreateFlags::empty(),
            image: image.clone(),
            view_type: ImageViewType::Type2D,
            format: get_format(&data).0,
            components: ComponentMapping::identity(),
            subresource_range: ImageSubresourceRange {
                aspect_mask: ImageAspectFlags::empty(),
                base_mip_level: 1,
                level_count: OptionalMipLevels::Remaining,
                base_array_layer: 1,
                layer_count: OptionalArrayLayers::Remaining,
            },
            chain: None,
        };

        self.logical
            .create_image_view(&create_info, None)
            .map(|v| Rc::new(v))
            .map_err(|e| ())
    }

    fn create_uniform(&self) -> Result<Rc<(Buffer, MappedMemoryRange)>, ()> {
        let create_info = BufferCreateInfo {
            flags: BufferCreateFlags::empty(),
            size: std::mem::size_of::<cgmath::Matrix4<f32>>() as _,
            usage: BufferUsageFlags::UNIFORM_BUFFER,
            sharing_mode: SharingMode::Exclusive,
            queue_family_indices: vec![],
            chain: None,
        };
        let reqs = AllocatorMemoryRequirements {
            own_memory: false,
            usage: MemoryUsage::CpuOnly,
            required_flags: MemoryPropertyFlags::HOST_COHERENT,
            preferred_flags: None,
            never_allocate: false,
        };

        self.allocator
            .create_buffer(&create_info, &reqs)
            .map(|v| Rc::new(v))
            .map_err(|e| ())
    }

    fn create_descriptor_set(&self) -> Result<Rc<DescriptorSet>, ()> {
        let allocate_info = DescriptorSetAllocateInfo {
            descriptor_pool: (*self.descriptors.0).clone(),
            set_layouts: vec![(*self.descriptors.1[0]).clone()],
            chain: None,
        };

        DescriptorPool::allocate_descriptor_sets(&allocate_info)
            .map(|v| Rc::new(v[0].clone()))
            .map_err(|e| ())
    }

    fn write_descriptor_set(
        &self,
        dst_set: &DescriptorSet,
        uniform: Option<&(Buffer, MappedMemoryRange)>,
        view: &ImageView,
    ) {
        let mut writes = Vec::<WriteDescriptorSet>::with_capacity(2);

        match uniform {
            Some(uniform) => writes.push(WriteDescriptorSet {
                dst_set: dst_set.clone(),
                dst_binding: 0,
                dst_array_element: 0,
                descriptor_type: DescriptorType::UniformBuffer,
                elements: WriteDescriptorSetElements::BufferInfo(vec![DescriptorBufferInfo {
                    buffer: uniform.0.clone(),
                    offset: uniform.1.offset,
                    range: uniform.1.size,
                }]),
                chain: None,
            }),
            None => {}
        };

        writes.push(WriteDescriptorSet {
            dst_set: dst_set.clone(),
            dst_binding: 1,
            dst_array_element: 0,
            descriptor_type: DescriptorType::CombinedImageSampler,
            elements: WriteDescriptorSetElements::ImageInfo(vec![DescriptorImageInfo {
                sampler: Some((*self.sampler).clone()),
                image_view: Some(view.clone()),
                image_layout: ImageLayout::ShaderReadOnlyOptimal,
            }]),
            chain: None,
        });

        DescriptorSet::update(Some(writes.as_slice()), None)
    }

    pub fn create_texture(&self, buffer: &WlBuffer) -> Result<RefCell<TextureMetadata>, ()> {
        match shm_buffer_contents(buffer, |slice, data| {
            assert_eq!(slice.len() as u64, data.buffer_size());
            let shm_input = RefCell::new(self.create_shm_input(data.buffer_size())?);
            let shm = Some((shm_input.clone(), RefCell::new(data)));
            let texture = self.create_texture_image(&data)?;
            let texture_view = self.create_texture_view(&data, &texture.0)?;
            let fragment = get_format(&data).1;
            let y_inverted = false;
            let format = Rc::new(get_format(&data).0);
            let uniform = self.create_uniform()?;
            let uniform_map = Allocator::map_memory(&uniform.1)
                .map(|v| Rc::new(v))
                .map_err(|e| ())?;
            let uniform_matrix = uniform_map.as_ptr() as *mut cgmath::Matrix4<f32>;
            let descriptor_set = self.create_descriptor_set()?;
            self.write_descriptor_set(&descriptor_set, Some(&uniform), &texture_view);

            unsafe {
                *uniform_matrix = cgmath::Matrix4 {
                    x: cgmath::Vector4 {
                        x: data.width as f32,
                        y: 0.0,
                        z: 0.0,
                        w: 0.0,
                    },
                    y: cgmath::Vector4 {
                        x: 0.0,
                        y: data.height as f32,
                        z: 0.0,
                        w: 0.0,
                    },
                    z: cgmath::Vector4 {
                        x: 0.0,
                        y: 0.0,
                        z: 0.0,
                        w: 0.0,
                    },
                    w: cgmath::Vector4 {
                        x: 0.0,
                        y: 0.0,
                        z: 0.0,
                        w: 0.0,
                    },
                };
            }
            uniform_map.flush(&None);

            let ret = TextureMetadata {
                shm,
                texture,
                texture_view,
                fragment,
                y_inverted,
                format,
                uniform,
                uniform_map,
                uniform_matrix,
                descriptor_set,
            };

            self.texture_copy_buffer(&slice, &data, &shm_input.borrow())?;

            Ok(RefCell::new(ret))
        }) {
            Ok(Ok(rc)) => Ok(rc),
            Ok(Err(())) => {
                println!("Unsupported SHM buffer");
                Err(())
            }
            Err(err) => {
                println!("Unable to load buffer contents: {:?}", err);
                Err(())
            }
        }
    }

    pub fn surface_commit(&self, buffer: &WlBuffer, texture: &mut TextureMetadata) -> Result<(), ()> {
        match shm_buffer_contents(buffer, |slice, data| match &texture.shm {
            Some((shm_input, previous_data)) => {
                if data.buffer_size() > previous_data.borrow().buffer_size() {
                    shm_input.replace(self.create_shm_input(data.buffer_size())?);
                }
                if data.height != previous_data.borrow().height
                    || data.width != previous_data.borrow().width
                    || data.format != data.format
                {
                    texture.texture = self.create_texture_image(&data)?;
                    texture.texture_view = self.create_texture_view(&data, &texture.texture.0)?;
                    self.write_descriptor_set(&texture.descriptor_set, None, &texture.texture_view);
                }
                if data.format != previous_data.borrow().format {
                    texture.fragment = get_format(&data).1;
                    texture.y_inverted = false;
                    texture.format = Rc::new(get_format(&data).0);
                }
                previous_data.replace(data);

                unsafe {
                    *texture.uniform_matrix = cgmath::Matrix4 {
                        x: cgmath::Vector4 {
                            x: data.width as f32,
                            y: 0.0,
                            z: 0.0,
                            w: 0.0,
                        },
                        y: cgmath::Vector4 {
                            x: 0.0,
                            y: data.height as f32,
                            z: 0.0,
                            w: 0.0,
                        },
                        z: cgmath::Vector4 {
                            x: 0.0,
                            y: 0.0,
                            z: 0.0,
                            w: 0.0,
                        },
                        w: cgmath::Vector4 {
                            x: 0.0,
                            y: 0.0,
                            z: 0.0,
                            w: 0.0,
                        },
                    };
                }
                texture.uniform_map.flush(&None);

                self.texture_copy_buffer(&slice, &previous_data.borrow(), &shm_input.borrow())
                    .map_err(|_| {
                        "Couldn't copy buffers.";
                        ()
                    })
            }
            None => Err({
                "No SHM Option.";
                ()
            }),
        }) {
            Ok(Ok(())) => Ok(()),
            Ok(Err(format)) => {
                println!("Unsupported SHM buffer format: {:?}", format);
                Err(())
            }
            Err(err) => {
                println!("Unable to load buffer contents: {:?}", err);
                Err(())
            }
        }
    }

    fn texture_copy_buffer(
        &self,
        slice: &[u8],
        data: &BufferData,
        shm_input: &(Buffer, MappedMemoryRange),
    ) -> Result<(), ()> {
        let buffer_map = Allocator::map_memory(&shm_input.1).map_err(|e| {
            println!("Unable to map buffer memory: {:?}", e);
            ()
        })?;
        let buffer_slice =
            unsafe { from_raw_parts_mut(buffer_map.as_ptr() as *mut u8, data.buffer_size() as _) };

        buffer_slice.clone_from_slice(slice);

        buffer_map.flush(&None).map_err(|err| {
            println!("Unable to flush buffer contents: {:?}", err);
            ()
        })
    }
}

fn begin_commands(command_buffer: &CommandBuffer) -> Result<(), ()> {
    let begin_info = CommandBufferBeginInfo {
        flags: CommandBufferUsageFlags::ONE_TIME_SUBMIT,
        inheritance_info: None,
        chain: None,
    };

    command_buffer.begin(&begin_info).map_err(|e| ())
}

fn copy_buffers(
    command_buffer: &CommandBuffer,
    id: &u64,
    window_map: &RefCell<MyWindowMap>,
    compositor_token: MyCompositorToken,
) -> Result<(), ()> {
    let mut precopy_barriers: Vec<ImageMemoryBarrier> = Vec::new();
    let mut postcopy_barriers: Vec<ImageMemoryBarrier> = Vec::new();
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
                                if !metadata.shm.is_none() {
                                    precopy_barriers.push(ImageMemoryBarrier {
                                        src_access_mask: AccessFlags::empty(),
                                        dst_access_mask: AccessFlags::TRANSFER_WRITE,
                                        old_layout: ImageLayout::Undefined,
                                        new_layout: ImageLayout::TransferDstOptimal,
                                        src_queue_family_index: QueueFamilyIndex::Ignored,
                                        dst_queue_family_index: QueueFamilyIndex::Ignored,
                                        image: metadata.texture.0.clone(),
                                        subresource_range: ImageSubresourceRange {
                                            aspect_mask: ImageAspectFlags::COLOR,
                                            base_mip_level: 0,
                                            level_count: OptionalMipLevels::Remaining,
                                            base_array_layer: 0,
                                            layer_count: OptionalArrayLayers::Remaining,
                                        },
                                        chain: None,
                                    });
                                    postcopy_barriers.push(ImageMemoryBarrier {
                                        src_access_mask: AccessFlags::TRANSFER_WRITE,
                                        dst_access_mask: AccessFlags::SHADER_READ,
                                        old_layout: ImageLayout::TransferDstOptimal,
                                        new_layout: ImageLayout::ShaderReadOnlyOptimal,
                                        src_queue_family_index: QueueFamilyIndex::Ignored,
                                        dst_queue_family_index: QueueFamilyIndex::Ignored,
                                        image: metadata.texture.0.clone(),
                                        subresource_range: ImageSubresourceRange {
                                            aspect_mask: ImageAspectFlags::COLOR,
                                            base_mip_level: 0,
                                            level_count: OptionalMipLevels::Remaining,
                                            base_array_layer: 0,
                                            layer_count: OptionalArrayLayers::Remaining,
                                        },
                                        chain: None,
                                    });
                                }
                            }
                            TraversalAction::SkipChildren
                        },
                    )
                    .expect("Some stuff barriers");
            }
        });

    command_buffer.pipeline_barrier(
        PipelineStageFlags::TOP_OF_PIPE,
        PipelineStageFlags::TRANSFER,
        DependencyFlags::empty(),
        None,
        None,
        Some(&precopy_barriers[..]),
    );

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
                                if let Some(shm) = &metadata.shm {
                                    command_buffer.copy_buffer_to_image(
                                        &shm.0.borrow().0,
                                        &metadata.texture.0,
                                        ImageLayout::TransferDstOptimal,
                                        &[BufferImageCopy {
                                            buffer_offset: shm.0.borrow().1.offset,
                                            buffer_row_length: shm.1.borrow().stride as _,
                                            buffer_image_height: shm.1.borrow().height as _,
                                            image_subresource: ImageSubresourceLayers {
                                                aspect_mask: ImageAspectFlags::COLOR,
                                                base_array_layer: 0,
                                                layer_count: 1,
                                                mip_level: 1,
                                            },
                                            image_offset: Offset3D { x: 0, y: 0, z: 0 },
                                            image_extent: Extent3D {
                                                width: shm.1.borrow().width as _,
                                                height: shm.1.borrow().height as _,
                                                depth: 1,
                                            },
                                        }],
                                    );
                                }
                            }
                            TraversalAction::SkipChildren
                        },
                    )
                    .expect("Some stuff copy");
            }
        });

    command_buffer.pipeline_barrier(
        PipelineStageFlags::TRANSFER,
        PipelineStageFlags::FRAGMENT_SHADER,
        DependencyFlags::empty(),
        None,
        None,
        Some(&postcopy_barriers[..]),
    );

    Ok(())
}

impl Device {
    pub fn tick(
        &self,
        device_id: &u64,
        window_map: &RefCell<MyWindowMap>,
        compositor_token: MyCompositorToken,
    ) -> Result<Duration, ()> {
        self.logical.wait_idle().map_err(|e| ())?;

        let command_buffer = match &self.transfer {
            Some((_, _, buffer)) => buffer.clone(),
            None => self.graphics.3.clone(),
        };

        command_buffer
            .reset(CommandBufferResetFlags::empty())
            .map_err(|e| ())?;

        begin_commands(&command_buffer)?;

        if !window_map.borrow().is_empty() {
            copy_buffers(&command_buffer, device_id, &window_map, compositor_token)?
        }

        let displays = self.device_settings.display_prober.get_displays()?;

        let now = Instant::now();
        let mut times: Vec<Duration> = vec![];
        for (id, screen) in displays {
            times.push(
                match self.displays.borrow_mut().entry(id) {
                    Entry::Occupied(e) => {
                        let display = e.get();
                        display.marker.set(now);
                        display.clone()
                    }
                    Entry::Vacant(e) => {
                        let (surface, extent, duration) = screen.get_surface()?;
                        let display = Display::init(
                            &self,
                            surface.clone(),
                            extent,
                            duration,
                            now,
                            if self.presentations.is_empty() {
                                assert!(self
                                    .device_settings
                                    .physical_device
                                    .get_surface_support_khr(self.graphics.0, &surface)
                                    .map_err(|e| ())?);
                                self.graphics.0
                            } else {
                                self.presentations
                                    .iter()
                                    .find(|i| {
                                        self.device_settings
                                            .physical_device
                                            .get_surface_support_khr(i.0, &surface)
                                            .map_err(|e| ())
                                            .unwrap_or(false)
                                    })
                                    .unwrap()
                                    .0
                            },
                        )?;
                        e.insert(display.clone());
                        display
                    }
                }
                .tick(&self, &command_buffer, device_id, &window_map, compositor_token)?,
            );
        }

        command_buffer.end().map_err(|e| ())?;

        let submit_info = SubmitInfo {
            wait_semaphores: self
                .displays
                .borrow()
                .iter()
                .map(|screen| (*screen.1.image_avalible).clone())
                .collect(),
            wait_dst_stage_mask: vec![PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT],
            command_buffers: vec![(*command_buffer).clone()],
            signal_semaphores: vec![(*self.render_done).clone()],
            chain: None,
        };

        self.graphics
            .1
            .submit(Some(&[submit_info]), Some(&self.fence))
            .map_err(|e| ())?;

        while !self.fence.wait_for(Timeout::Infinite).map_err(|e| ())? {}
        self.fence.reset().map_err(|e| ())?;

        self.graphics
            .1
            .queue_present_khr(&mut PresentInfoKhr {
                wait_semaphores: vec![(*self.render_done).clone()],
                swapchains: self
                    .displays
                    .borrow()
                    .iter()
                    .map(|screen| (*screen.1.swapchain).clone())
                    .collect(),
                image_indices: self
                    .displays
                    .borrow()
                    .iter()
                    .map(|screen| screen.1.current_framebuffer.get() as u32)
                    .collect(),
                results: None,
                chain: None,
            })
            .map_err(|e| ())?;

        let mut ids: Vec<u64> = vec![];
        for (id, display) in self.displays.borrow().iter() {
            if (*display).marker.get() != now {
                ids.push(id.clone());
            }
        }
        let mut displays = self.displays.borrow_mut();
        for id in ids {
            displays.remove(&id);
        }
        Ok(times
            .iter()
            .cloned()
            .fold(Duration::from_millis(17), Duration::min))
    }
}
