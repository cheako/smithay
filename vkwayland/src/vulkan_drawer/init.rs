use std::collections::HashMap;

use dacite::{
    core::{
        ApplicationInfo, Extent2D, Format, ImageView, Instance, InstanceCreateFlags, InstanceCreateInfo,
        InstanceExtensions, Version,
    },
    khr_swapchain::SwapchainKhr,
};

use super::{VulkanDrawer, VulkanDrawerUtil};

struct SwapchainSettings {
    swapchain: SwapchainKhr,
    extent: Extent2D,
    image_views: Vec<ImageView>,
    format: Format,
}

impl VulkanDrawerUtil for VulkanDrawer {
    fn create_instance(
        instance_extensions: InstanceExtensions,
        engine_name: Option<String>,
    ) -> Result<Instance, ()> {
        let application_info = ApplicationInfo {
            application_name: Some("VkWayland".to_owned()),
            application_version: (Version {
                major: 0,
                minor: 0,
                patch: 2,
            })
            .as_api_version(),
            engine_name: engine_name,
            engine_version: (Version {
                major: 0,
                minor: 0,
                patch: 2,
            })
            .as_api_version(),
            api_version: Some(Version {
                major: 1,
                minor: 1,
                patch: 0,
            }),
            chain: None,
        };
        let create_info = InstanceCreateInfo {
            flags: InstanceCreateFlags::empty(),
            application_info: Some(application_info),
            enabled_layers: vec![],
            enabled_extensions: instance_extensions,
            chain: None,
        };
        Instance::create(&create_info, None).map_err(|e| {
            println!("Failed to create instance ({})", e);
        })
    }
}

/*
impl VulkanDrawerInit for VulkanDrawer {
        fn blah() {
                let graphics_queue = device.get_queue(queue_family_indices.graphics, 0);
                let present_queue = device.get_queue(queue_family_indices.present, 0);

                let SwapchainSettings {
                        swapchain,
                        extent,
                        image_views: swapchain_image_views,
                        format,
                } = create_swapchain(
                        &physical_device,
                        &device,
                        &surface,
                        &preferred_extent,
                        &queue_family_indices,
                )?;

                let render_pass = create_render_pass(&device, format)?;
                let framebuffers =
                        create_framebuffers(&device, &swapchain_image_views, &render_pass, &extent)?;
                let pipelines = create_pipeline(&device, &render_pass, &extent)?;
                let command_pool = create_command_pool(&device, queue_family_indices.graphics)?;

                Ok(VulkanDrawer {
                        log: log,
                        device: device,
                        graphics_queue: graphics_queue,
                        present_queue: present_queue,
                        swapchain: swapchain,
                        extent: extent,
                        format: format,
                        render_pass: render_pass,
                        framebuffers: [
                                framebuffers[0].clone(),
                                framebuffers[1].clone(),
                                framebuffers[2].clone(),
                        ],
                        pipelines: [
                                pipelines[0].clone(),
                                pipelines[1].clone(),
                                pipelines[2].clone(),
                                pipelines[3].clone(),
                                pipelines[4].clone(),
                        ],
                        command_pool: command_pool,
                })
        }
}
        */
