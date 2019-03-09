use std::collections::HashMap;
use std::os::raw::c_char;

use ash::{version::EntryV1_0, vk, Entry as VkEntry};

use super::{VulkanDrawer, VulkanDrawerUtil};

struct SwapchainSettings {
    swapchain: vk::SwapchainKHR,
    extent: vk::Extent2D,
    image_views: Vec<vk::ImageView>,
    format: vk::Format,
}

impl VulkanDrawerUtil for VulkanDrawer {
    fn create_instance(
        entry: VkEntry,
        instance_extensions: Vec<*const c_char>,
        engine_name: Option<String>,
    ) -> Result<ash::instance::Instance, ()> {
        let app_info = vk::ApplicationInfo {
            p_application_name: "VkWayland".as_ptr() as _,
            api_version: ash::vk_make_version!(1, 1, 0),
            application_version: ash::vk_make_version!(0, 0, 2),
            ..Default::default()
        };
        let create_info = vk::InstanceCreateInfo {
            p_application_info: &app_info,
            enabled_extension_count: instance_extensions.len() as _,
            pp_enabled_extension_names: instance_extensions.as_ptr(),
            ..Default::default()
        };

        unsafe { entry.create_instance(&create_info, None) }.map_err(|e| ())
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
