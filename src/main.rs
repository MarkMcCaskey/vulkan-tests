#[macro_use]
extern crate vulkano;
#[macro_use]
extern crate vulkano_shader_derive;
extern crate winit;
extern crate vulkano_win;

use vulkano_win::VkSurfaceBuild;

use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, DynamicState};
use vulkano::device::Device;
use vulkano::image::AttachmentImage;
use vulkano::framebuffer::{Framebuffer, Subpass};
use vulkano::instance::Instance;
use vulkano::pipeline::GraphicsPipeline;
use vulkano::pipeline::viewport::Viewport;
use vulkano::swapchain;
use vulkano::swapchain::{PresentMode, SurfaceTransform, Swapchain, AcquireError,
                         SwapchainCreationError};
use vulkano::sync::{now, GpuFuture};
use std::mem;
use std::sync::Arc;
use std::collections::VecDeque;

fn compute_new_center(dimensions: [u32; 2], last_mouse_location: [usize; 2], old_center: [f32; 2], zoom: f32) -> [f32; 2] {
    let new_zero_val = old_center[0] + (((((last_mouse_location[0] as f32) / (dimensions[0] as f32)) - 0.5) * 2.0) * zoom) ;
    let new_one_val = old_center[1] + (((((last_mouse_location[1] as f32) / (dimensions[1] as f32)) - 0.5) * 2.0) * zoom);
    let new_zero = if new_zero_val > 2.0 {2.0} else if new_zero_val < -2.0 {-2.0} else {new_zero_val};
    let new_one = if new_one_val > 2.0 {2.0} else if new_one_val < -2.0 {-2.0} else {new_one_val};

    [new_zero, new_one]
}

fn weighted_sum_for_smoothing(center: [f32; 2], adjust_by_vals: [f32; 2]) -> [f32; 2] {
    [(center[0] * 0.95) + (adjust_by_vals[0] * 0.05),
     (center[1] * 0.95) + (adjust_by_vals[1] * 0.05)]
}

fn weighted_sum_for_smoothing_zoom(zoom: f32, new_zoom: f32) -> f32 {
    (zoom * 0.9) + (new_zoom * 0.1)
}

fn main() {
    let instance = {
        let extensions = vulkano_win::required_extensions();
        Instance::new(None, &extensions, None).expect("failed to create Vulkan instance")
    };
    let physical = vulkano::instance::PhysicalDevice::enumerate(&instance)
        .next()
        .expect("no device available");
    println!("Using device: {} (type: {:?})",
             physical.name(),
             physical.ty());

    let mut events_loop = winit::EventsLoop::new();
    let window = winit::WindowBuilder::new()
        .build_vk_surface(&events_loop, instance.clone())
        .unwrap();

    let mut dimensions = {
        let (width, height) = window.window().get_inner_size_pixels().unwrap();
        [width, height]
    };

    let queue = physical
        .queue_families()
        .find(|&q| q.supports_graphics() && window.surface().is_supported(q).unwrap_or(false))
        .expect("couldn't find a graphical queue family");

    let (device, mut queues) = {
        let device_ext = vulkano::device::DeviceExtensions {
            khr_swapchain: true,
            ..vulkano::device::DeviceExtensions::none()
        };

        Device::new(physical,
                    physical.supported_features(),
                    &device_ext,
                    [(queue, 0.5)].iter().cloned())
                .expect("failed to create device")
    };

    let queue = queues.next().unwrap();

    let (mut swapchain, mut images) = {
        let caps = window
            .surface()
            .capabilities(physical)
            .expect("failed to get surface capabilities");

        let alpha = caps.supported_composite_alpha.iter().next().unwrap();
        let format = caps.supported_formats[0].0;

        Swapchain::new(device.clone(),
                       window.surface().clone(),
                       caps.min_image_count,
                       format,
                       dimensions,
                       1,
                       caps.supported_usage_flags,
                       &queue,
                       SurfaceTransform::Identity,
                       alpha,
                       PresentMode::Fifo,
                       true,
                       None)
                .expect("failed to create swapchain")
    };

    let vertex_buffer = {
        #[derive(Debug, Clone)]
        struct Vertex {
            position: [f32; 2],
        }
        impl_vertex!(Vertex, position);

        CpuAccessibleBuffer::from_iter(device.clone(),
                                       BufferUsage::all(),
                                       [Vertex { position: [1.0, -1.0] },
                                        Vertex { position: [-1.0, -1.0] },
                                        Vertex { position: [1.0, 1.0] },
                                        Vertex { position: [-1.0, 1.0] }]
                                               .iter()
                                               .cloned())
                .expect("failed to create buffer")
    };

    let uniform_buffer = vulkano::buffer::cpu_pool::CpuBufferPool::<fs::ty::Data>
        ::new(device.clone(), vulkano::buffer::BufferUsage::all());

    let vs = vs::Shader::load(device.clone()).expect("failed to create shader module");
    let fs = fs::Shader::load(device.clone()).expect("failed to create shader module");

    let mut intermediary = AttachmentImage::transient_multisampled(device.clone(), dimensions, 4, swapchain.format()).unwrap();

    let render_pass = Arc::new(single_pass_renderpass!(device.clone(),
        attachments: {
            intermediary: {
                load: Clear,
                store: DontCare,
                format: swapchain.format(),
                samples: 4,
            },
            color: {
                load: Clear,
                store: Store,
                format: swapchain.format(),
                samples: 1,
            }
        },
        pass: {
            color: [intermediary],
            depth_stencil: {},
            resolve: [color]
        }
    ).unwrap());

    let pipeline = Arc::new(GraphicsPipeline::start()
                                .vertex_input_single_buffer()
                                .vertex_shader(vs.main_entry_point(), ())
                                .triangle_strip()
                                .viewports_dynamic_scissors_irrelevant(1)
                                .fragment_shader(fs.main_entry_point(), ())
                                .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
                                .build(device.clone())
                                .unwrap());

    let mut framebuffers: Option<Vec<Arc<vulkano::framebuffer::Framebuffer<_, _>>>> = None;
    let mut recreate_swapchain = false;
    let mut previous_frame_end = Box::new(now(device.clone())) as Box<GpuFuture>;
    let mut zoom: f32 = 2.0;
    let mut center: [f32; 2] = [-1.5, -1.0];
    let mut last_mouse_location: [usize; 2] = [0, 0];
    let mut adjust_center_by_vals = VecDeque::new();
    let mut adjust_zoom_by_vals = VecDeque::new();

    loop {
        previous_frame_end.cleanup_finished();

        if recreate_swapchain {
            dimensions = {
                let (new_width, new_height) = window.window().get_inner_size_pixels().unwrap();
                [new_width, new_height]
            };

            let (new_swapchain, new_images) = match swapchain.recreate_with_dimension(dimensions) {
                Ok(r) => r,
                Err(SwapchainCreationError::UnsupportedDimensions) => {
                    continue;
                }
                Err(err) => panic!("{:?}", err),
            };

            let new_intermediary = AttachmentImage::transient_multisampled(device.clone(), dimensions, 4, swapchain.format()).unwrap();

            mem::replace(&mut swapchain, new_swapchain);
            mem::replace(&mut images, new_images);
            mem::replace(&mut intermediary, new_intermediary);

            framebuffers = None;

            recreate_swapchain = false;
        }

        if framebuffers.is_none() {
            let new_framebuffers = Some(images
                                            .iter()
                                            .map(|image| {
                Arc::new(Framebuffer::start(render_pass.clone())
                         .add(intermediary.clone())
                         .unwrap()
                         .add(image.clone())
                         .unwrap()
                         .build()
                         .unwrap())
            })
                                        .collect::<Vec<_>>());
            mem::replace(&mut framebuffers, new_framebuffers);
        }

        //TODO: refactor
        let mut new_queue = VecDeque::new();
        while let Some((val_to_adj_by, count)) = adjust_center_by_vals.pop_back() {
            if count <= 0 {
                continue;
            }

            center = weighted_sum_for_smoothing(center, val_to_adj_by);
            new_queue.push_front((val_to_adj_by, count - 1));
        }
        mem::swap(&mut adjust_center_by_vals, &mut new_queue);

        let mut new_queue = VecDeque::new();
        while let Some((val_to_adj_by, count)) = adjust_zoom_by_vals.pop_back() {
            if count <= 0 {
                continue;
            }

            zoom = weighted_sum_for_smoothing_zoom(zoom, val_to_adj_by);
            new_queue.push_front((val_to_adj_by, count - 1));
        }
        mem::swap(&mut adjust_zoom_by_vals, &mut new_queue);

        let uniform_buffer_subbuffer = {
            let uniform_data = fs::ty::Data {
                zoom: zoom.into(),
                center: center.into(),
                dimensions: [dimensions[0] as f32, dimensions[1] as f32].into(),
                _dummy0: [0, 0, 0, 0],
            };

            uniform_buffer.next(uniform_data).unwrap()
        };

        let set = Arc::new(vulkano::descriptor::descriptor_set::PersistentDescriptorSet::start(pipeline.clone(), 0)
            .add_buffer(uniform_buffer_subbuffer).unwrap()
            .build().unwrap()
        );

        let (image_num, acquire_future) = match swapchain::acquire_next_image(swapchain.clone(),
                                                                              None) {
            Ok(r) => r,
            Err(AcquireError::OutOfDate) => {
                recreate_swapchain = true;
                continue;
            }
            Err(err) => panic!("{:?}", err),
        };

        let command_buffer = AutoCommandBufferBuilder::primary_one_time_submit(device.clone(),
                                                                               queue.family())
                .unwrap()
                .begin_render_pass(framebuffers.as_ref().unwrap()[image_num].clone(),
                                   false,
                                   vec![[0.0, 0.0, 1.0, 1.0].into()])
                .unwrap()
                .draw(pipeline.clone(),
                      DynamicState {
                          line_width: None,
                          viewports: Some(vec![Viewport {
                                                   origin: center,
                                                   dimensions: [dimensions[0] as f32,
                                                                dimensions[1] as f32],
                                                   depth_range: 0.0..1.0,
                                               }]),
                          scissors: None,
                      },
                      vertex_buffer.clone(),
                      set.clone(),
                      ())
                .unwrap()

                .end_render_pass()
                .unwrap()

                .build()
                .unwrap();

        let future = previous_frame_end
            .join(acquire_future)
            .then_execute(queue.clone(), command_buffer)
            .unwrap()
            .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
            .then_signal_fence_and_flush()
            .unwrap();
        previous_frame_end = Box::new(future) as Box<_>;

        let mut done = false;
        events_loop.poll_events(|ev| {
            match ev {
                winit::Event::WindowEvent { event: winit::WindowEvent::Closed, .. } => done = true,
                winit::Event::WindowEvent { event: winit::WindowEvent::Resized(_, _), .. } => recreate_swapchain = true,
                winit::Event::WindowEvent { event: winit::WindowEvent::MouseWheel {delta, ..}, ..} => {
                    let new_zoom_amount = 
                        match delta {
                            winit::MouseScrollDelta::LineDelta(f1, f2) => zoom + (zoom * ((f1 - f2) / 10.0)),
                            winit::MouseScrollDelta::PixelDelta(f1, f2) => zoom + (zoom * ((f1 - f2) / 10.0)),
                        };
                    zoom = weighted_sum_for_smoothing_zoom(zoom, new_zoom_amount);
                    adjust_zoom_by_vals.push_front((new_zoom_amount, 9));
                },
                winit::Event::WindowEvent { event: winit::WindowEvent::MouseInput {button: winit::MouseButton::Left,
                                                                                   state: winit::ElementState::Pressed, ..}, ..} => {
                    let new_center = compute_new_center(dimensions, last_mouse_location, center, zoom);
                    center = weighted_sum_for_smoothing(center, new_center);
                    adjust_center_by_vals.push_front((new_center, 19));
                },
                winit::Event::WindowEvent { event: winit::WindowEvent::MouseMoved { position: (x, y), .. }, ..} => {
                    last_mouse_location = [x as usize, y as usize];
                }
                _ => ()
            }
        });
        if done {
            return;
        }
    }
}

mod vs {
    #[allow(dead_code)]
    #[derive(VulkanoShader)]
    #[ty = "vertex"]
    #[src = "
#version 450

layout(location = 0) in vec2 position;

void main() {
    gl_Position = vec4(position, 0.0, 1.0);
}
"]
    struct Dummy;
}

mod fs {
    #[allow(dead_code)]
    #[derive(VulkanoShader)]
    #[ty = "fragment"]
    #[src = "
#version 450

precision highp float;

layout(location = 0) out vec4 f_color;
layout(set = 0, binding = 0) uniform Data {
    float zoom;
    vec2  center;
    vec2  dimensions;
} uniforms;

vec3 hsv2rgb(vec3 c) {
  // from https://github.com/hughsk/glsl-hsv2rgb
  vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
  vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
  return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main() {
    vec2 c = ((gl_FragCoord.xy * uniforms.zoom) / uniforms.dimensions) + uniforms.center ;

    vec2 z = vec2(0.0, 0.0);
    float i;
    for (i = 0.0; i < 1.0; i += 0.005) {
        z = vec2(
            z.x * z.x - z.y * z.y + c.x,
            z.y * z.x + z.x * z.y + c.y
        );

        if (length(z) > 4.0) {
            break;
        }
    }

    f_color = vec4(hsv2rgb(vec3(i + 0.6,1.0,i)), 1.0);
}
"]
    struct Dummy;
}
