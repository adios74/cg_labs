#include "application.hpp"
#include "math.hpp"

#include <imgui.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>

namespace {

using graphics::internal::Context;
using graphics::internal::ModelUniform;
using graphics::internal::SpotLightGpu;
using graphics::internal::PointLightGpu;

constexpr float PI = 3.14159265358979323846f;
constexpr int OBJECT_COUNT = graphics::internal::MAIN_OBJECT_COUNT;

struct PointLight {
    float position[3];
    float color[3];
    float intensity;
};

struct SpotLight {
    float position[3];
    float direction[3];
    float color[3];
    float intensity;
    float inner_angle;
    float outer_angle;
};

struct ObjectState {
    float position[3];
    float rotation[3];
    float scale[3];
    float color[3];
    float orbit_radius;
    float orbit_height;
};

ObjectState objects[OBJECT_COUNT] = {
    {{-0.9f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, {0.75f, 0.75f, 0.75f},
     {1.0f, 1.0f, 1.0f}, 0.7f, 0.25f},
    {{1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, {0.55f, 0.55f, 0.55f},
     {0.8f, 1.0f, 0.85f}, 0.0f, 0.0f}
};

bool use_perspective = true;
bool animate = true;
float animation_speed = 1.0f;
float animation_time = 0.0f;
double previous_time = 0.0;

float camera_position[3] = {0.0f, 0.0f, 5.0f};
float camera_yaw = -90.0f;
float camera_pitch = 0.0f;
float camera_fov = 60.0f;

float ambient_color[3] = {0.12f, 0.12f, 0.15f};
float ambient_intensity = 1.0f;
float directional_direction[3] = {-0.4f, -1.0f, -0.3f};
float directional_color[3] = {1.0f, 0.95f, 0.85f};
float directional_intensity = 1.0f;

float albedo[3] = {0.7f, 0.75f, 0.9f};
float specular_color[3] = {1.0f, 1.0f, 1.0f};
float shininess = 48.0f;

PointLight point_lights[graphics::internal::MAX_POINT_LIGHTS] = {
    {{-2.0f, 1.0f, 1.0f}, {1.0f, 0.25f, 0.15f}, 8.0f},
    {{ 2.0f, 1.0f, 1.0f}, {0.2f, 0.45f, 1.0f}, 8.0f}
};

SpotLight spot_lights[graphics::internal::MAX_SPOT_LIGHTS] = {
    {{0.0f, 2.5f, 2.0f}, {0.0f, -1.0f, -0.6f}, {0.3f, 1.0f, 0.4f},
     12.0f, 18.0f, 30.0f}
};

uint32_t point_count = 2;
uint32_t spot_count = 1;
bool show_light_visualizers = true;

void normalize3(float vector[3]) {
    const float length = std::sqrt(
        vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]);

    if (length > 1.0e-6f) {
        vector[0] /= length;
        vector[1] /= length;
        vector[2] /= length;
    }
}

void lookAt(const float eye[3], const float direction_in[3], math::Mat4 out) {
    float forward[3] = {direction_in[0], direction_in[1], direction_in[2]};
    normalize3(forward);

    float up[3] = {0.0f, 1.0f, 0.0f};

    if (std::fabs(forward[1]) > 0.98f) {
        up[0] = 0.0f;
        up[1] = 0.0f;
        up[2] = 1.0f;
    }

    float side[3] = {
        forward[1] * up[2] - forward[2] * up[1],
        forward[2] * up[0] - forward[0] * up[2],
        forward[0] * up[1] - forward[1] * up[0]
    };
    
    normalize3(side);

    const float corrected_up[3] = {
        side[1] * forward[2] - side[2] * forward[1],
        side[2] * forward[0] - side[0] * forward[2],
        side[0] * forward[1] - side[1] * forward[0]
    };

    math::identity(out);
    out[0][0] = side[0];
    out[1][0] = side[1];
    out[2][0] = side[2];
    out[3][0] = -(side[0] * eye[0] + side[1] * eye[1] + side[2] * eye[2]);

    out[0][1] = corrected_up[0];
    out[1][1] = corrected_up[1];
    out[2][1] = corrected_up[2];
    out[3][1] = -(corrected_up[0] * eye[0] + corrected_up[1] * eye[1] + corrected_up[2] * eye[2]);

    out[0][2] = -forward[0];
    out[1][2] = -forward[1];
    out[2][2] = -forward[2];
    out[3][2] = forward[0] * eye[0] + forward[1] * eye[1] + forward[2] * eye[2];
}

void makeObjectMatrix(const ObjectState& object, float time, math::Mat4 out) {
    const float x = object.position[0] + std::cos(time) * object.orbit_radius;
    const float y = object.position[1] + std::sin(time * 1.7f) * object.orbit_height;
    const float z = object.position[2] + std::sin(time) * object.orbit_radius;

    math::Mat4 translation{};
    math::Mat4 rotation_x{};
    math::Mat4 rotation_y{};
    math::Mat4 rotation_z{};
    math::Mat4 scaling{};
    math::Mat4 temporary{};

    math::translate(x, y, z, translation);
    math::rotateX(object.rotation[0] * PI / 180.0f + time, rotation_x);
    math::rotateY(object.rotation[1] * PI / 180.0f + time, rotation_y);
    math::rotateZ(object.rotation[2] * PI / 180.0f, rotation_z);
    math::scale(object.scale[0], object.scale[1], object.scale[2], scaling);

    math::multiply(translation, rotation_z, temporary);
    math::multiply(temporary, rotation_y, temporary);
    math::multiply(temporary, rotation_x, temporary);
    math::multiply(temporary, scaling, out);
}

void setMarkerMaterial(ModelUniform* uniform, const float color[3]) {
    for (int i = 0; i < 3; ++i) {
        uniform->albedo[i] = color[i];
        uniform->specular_color[i] = 1.0f;
    }
    uniform->albedo[3] = 1.0f;
    uniform->specular_color[3] = 1.0f;
    uniform->material_params[0] = 32.0f;
    uniform->material_params[1] = 1.0f;
    uniform->material_params[2] = 0.0f;
    uniform->material_params[3] = 0.0f;
}

void writeMarker(ModelUniform* uniform, float x, float y, float z,
                 float sx, float sy, float sz, const float color[3]) {
    math::Mat4 translation{};
    math::Mat4 scaling{};
    math::Mat4 model{};
    math::translate(x, y, z, translation);
    math::scale(sx, sy, sz, scaling);
    math::multiply(translation, scaling, model);
    std::memcpy(uniform->model, model, sizeof(model));
    setMarkerMaterial(uniform, color);
}

void writeRod(ModelUniform* uniform, const float start[3], const float end[3],
              const float color[3]) {
    float direction[3] = {
        end[0] - start[0], end[1] - start[1], end[2] - start[2]
    };
    float length = std::sqrt(direction[0] * direction[0] +
                             direction[1] * direction[1] +
                             direction[2] * direction[2]);
    if (length < 1.0e-5f) {
        length = 1.0e-5f;
    }
    for (float& component : direction) {
        component /= length;
    }

    float reference[3] = {0.0f, 1.0f, 0.0f};
    if (std::fabs(direction[1]) > 0.95f) {
        reference[0] = 1.0f;
        reference[1] = 0.0f;
        reference[2] = 0.0f;
    }

    float x_axis[3] = {
        reference[1] * direction[2] - reference[2] * direction[1],
        reference[2] * direction[0] - reference[0] * direction[2],
        reference[0] * direction[1] - reference[1] * direction[0]
    };
    normalize3(x_axis);

    const float y_axis[3] = {
        direction[1] * x_axis[2] - direction[2] * x_axis[1],
        direction[2] * x_axis[0] - direction[0] * x_axis[2],
        direction[0] * x_axis[1] - direction[1] * x_axis[0]
    };

    math::identity(uniform->model);
    for (int i = 0; i < 3; ++i) {
        uniform->model[0][i] = x_axis[i] * 0.008f;
        uniform->model[1][i] = y_axis[i] * 0.008f;
        uniform->model[2][i] = direction[i] * (length * 0.5f);
        uniform->model[3][i] = (start[i] + end[i]) * 0.5f;
    }
    setMarkerMaterial(uniform, color);
}

void updateLightVisualizers(graphics::internal::Context& context) {
    const uint32_t point_base = OBJECT_COUNT;
    const uint32_t spot_base = point_base + graphics::internal::MAX_POINT_LIGHTS;
    const uint32_t cone_base = spot_base + graphics::internal::MAX_SPOT_LIGHTS;
    const float white[3] = {1.0f, 1.0f, 1.0f};

    for (uint32_t i = 0; i < graphics::internal::MAX_POINT_LIGHTS; ++i) {
        ModelUniform* uniform = context.model_uniform_buffer_mapped[point_base + i];
        if (i < point_count) {
            const PointLight& light = point_lights[i];
            writeMarker(uniform, light.position[0], light.position[1], light.position[2],
                        0.09f, 0.09f, 0.09f, light.color);
        } else {
            writeMarker(uniform, 0.0f, -100.0f, 0.0f,
                        0.001f, 0.001f, 0.001f, white);
        }
    }

    for (uint32_t i = 0; i < graphics::internal::MAX_SPOT_LIGHTS; ++i) {
        ModelUniform* uniform = context.model_uniform_buffer_mapped[spot_base + i];
        if (i < spot_count) {
            const SpotLight& light = spot_lights[i];
            writeMarker(uniform, light.position[0], light.position[1], light.position[2],
                        0.11f, 0.11f, 0.11f, light.color);
        } else {
            writeMarker(uniform, 0.0f, -100.0f, 0.0f,
                        0.001f, 0.001f, 0.001f, white);
        }
    }

    for (uint32_t i = 0; i < graphics::internal::MAX_SPOT_LIGHTS; ++i) {
        for (uint32_t segment = 0; segment < graphics::internal::SPOT_CONE_LINES_PER_LIGHT; ++segment) {
            ModelUniform* uniform = context.model_uniform_buffer_mapped[
                cone_base + i * graphics::internal::SPOT_CONE_LINES_PER_LIGHT + segment];

            if (i >= spot_count) {
                writeMarker(uniform, 0.0f, -100.0f, 0.0f,
                            0.001f, 0.001f, 0.001f, white);
                continue;
            }

            const SpotLight& light = spot_lights[i];
            float direction[3] = {light.direction[0], light.direction[1], light.direction[2]};
            normalize3(direction);

            float reference[3] = {0.0f, 1.0f, 0.0f};
            if (std::fabs(direction[1]) > 0.95f) {
                reference[0] = 1.0f;
                reference[1] = 0.0f;
                reference[2] = 0.0f;
            }

            float axis_x[3] = {
                direction[1] * reference[2] - direction[2] * reference[1],
                direction[2] * reference[0] - direction[0] * reference[2],
                direction[0] * reference[1] - direction[1] * reference[0]
            };
            normalize3(axis_x);
            const float axis_y[3] = {
                direction[1] * axis_x[2] - direction[2] * axis_x[1],
                direction[2] * axis_x[0] - direction[0] * axis_x[2],
                direction[0] * axis_x[1] - direction[1] * axis_x[0]
            };

            const float radius = std::tan(light.outer_angle * PI / 180.0f);
            const uint32_t ring_segment = segment % graphics::internal::SPOT_CONE_SEGMENTS;
            const float angle0 = 2.0f * PI * ring_segment / graphics::internal::SPOT_CONE_SEGMENTS;
            const float angle1 = 2.0f * PI * ((ring_segment + 1) % graphics::internal::SPOT_CONE_SEGMENTS) /
                                 graphics::internal::SPOT_CONE_SEGMENTS;

            float origin[3] = {light.position[0], light.position[1], light.position[2]};
            float point0[3];
            float point1[3];
            for (int axis = 0; axis < 3; ++axis) {
                point0[axis] = light.position[axis] + direction[axis] + radius *
                    (std::cos(angle0) * axis_x[axis] + std::sin(angle0) * axis_y[axis]);
                point1[axis] = light.position[axis] + direction[axis] + radius *
                    (std::cos(angle1) * axis_x[axis] + std::sin(angle1) * axis_y[axis]);
            }

            if (segment < graphics::internal::SPOT_CONE_SEGMENTS) {
                writeRod(uniform, origin, point0, white);
            } else {
                writeRod(uniform, point0, point1, white);
            }
        }
    }
}

} // namespace

namespace application {

bool initialize() {
    return true;
}

void shutdown() {
    auto& context = graphics::internal::context;
    vkQueueWaitIdle(context.graphics_queue);
}

void update(double time) {
    auto& context = graphics::internal::context;

    float delta_time = static_cast<float>(time - previous_time);
    previous_time = time;
    delta_time = std::clamp(delta_time, 0.0f, 0.1f);
    if (animate) {
        animation_time += delta_time * animation_speed;
    }

    ImGui::Begin("Lab 1 + Lab 2");

    if (ImGui::CollapsingHeader("Projection", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (ImGui::RadioButton("Perspective", use_perspective)) {
            use_perspective = true;
        }
        ImGui::SameLine();
        if (ImGui::RadioButton("Orthographic", !use_perspective)) {
            use_perspective = false;
        }
    }

    if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::SliderFloat3("Position", camera_position, -10.0f, 10.0f);
        ImGui::SliderFloat("Yaw", &camera_yaw, -180.0f, 180.0f);
        ImGui::SliderFloat("Pitch", &camera_pitch, -89.0f, 89.0f);
        ImGui::SliderFloat("Field of view", &camera_fov, 25.0f, 110.0f);
    }

    if (ImGui::CollapsingHeader("Animation", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Checkbox("Play animation", &animate);
        ImGui::SliderFloat("Animation speed", &animation_speed, 0.0f, 5.0f);
    }

    if (ImGui::CollapsingHeader("Material and ambient light", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::ColorEdit3("Albedo", albedo);
        ImGui::ColorEdit3("Specular color", specular_color);
        ImGui::SliderFloat("Shininess", &shininess, 1.0f, 256.0f);
        ImGui::ColorEdit3("Ambient color", ambient_color);
        ImGui::SliderFloat("Ambient intensity", &ambient_intensity, 0.0f, 2.0f);
    }

    for (int i = 0; i < OBJECT_COUNT; ++i) {
        ObjectState& object = objects[i];
        ImGui::PushID(i);
        const char* label = (i == 0) ? "Sphere 1" : "Sphere 2";
        if (ImGui::CollapsingHeader(label, i == 0 ? ImGuiTreeNodeFlags_DefaultOpen : 0)) {
            ImGui::SliderFloat3("Position", object.position, -3.0f, 3.0f);
            ImGui::SliderFloat3("Rotation", object.rotation, -180.0f, 180.0f);
            ImGui::SliderFloat3("Scale", object.scale, 0.1f, 3.0f);
            ImGui::ColorEdit3("Vertex color multiplier", object.color);
            ImGui::SliderFloat("Orbit radius", &object.orbit_radius, 0.0f, 3.0f);
            ImGui::SliderFloat("Orbit height", &object.orbit_height, -1.0f, 1.0f);
        }
        ImGui::PopID();
    }

    if (ImGui::CollapsingHeader("Directional light", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::SliderFloat3("Direction", directional_direction, -1.0f, 1.0f);
        ImGui::ColorEdit3("Color##directional", directional_color);
        ImGui::SliderFloat("Intensity##directional", &directional_intensity, 0.0f, 5.0f);
    }

    if (ImGui::CollapsingHeader("Point lights", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (point_count < graphics::internal::MAX_POINT_LIGHTS && ImGui::Button("+ Add point light")) {
            point_lights[point_count] = {{0.0f, 2.0f, 2.0f}, {1.0f, 1.0f, 1.0f}, 5.0f};
            ++point_count;
        }
        ImGui::SameLine();
        ImGui::Text("%u / %u", point_count, graphics::internal::MAX_POINT_LIGHTS);

        for (uint32_t i = 0; i < point_count; ++i) {
            ImGui::PushID(1000 + static_cast<int>(i));
            if (ImGui::TreeNode("Point light")) {
                PointLight& light = point_lights[i];
                ImGui::SliderFloat3("Position", light.position, -6.0f, 6.0f);
                ImGui::ColorEdit3("Color", light.color);
                ImGui::SliderFloat("Intensity", &light.intensity, 0.0f, 50.0f);
                if (ImGui::Button("Remove point light")) {
                    for (uint32_t j = i + 1; j < point_count; ++j) {
                        point_lights[j - 1] = point_lights[j];
                    }
                    --point_count;
                    ImGui::TreePop();
                    ImGui::PopID();
                    break;
                }
                ImGui::TreePop();
            }
            ImGui::PopID();
        }
    }

    if (ImGui::CollapsingHeader("Spot lights", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (spot_count < graphics::internal::MAX_SPOT_LIGHTS && ImGui::Button("+ Add spotlight")) {
            spot_lights[spot_count] = {{0.0f, 3.0f, 2.0f}, {0.0f, -1.0f, -1.0f},
                                       {1.0f, 1.0f, 1.0f}, 10.0f, 15.0f, 30.0f};
            ++spot_count;
        }
        ImGui::SameLine();
        ImGui::Text("%u / %u", spot_count, graphics::internal::MAX_SPOT_LIGHTS);

        for (uint32_t i = 0; i < spot_count; ++i) {
            ImGui::PushID(2000 + static_cast<int>(i));
            if (ImGui::TreeNode("Spotlight")) {
                SpotLight& light = spot_lights[i];
                ImGui::SliderFloat3("Position", light.position, -6.0f, 6.0f);
                ImGui::SliderFloat3("Direction", light.direction, -1.0f, 1.0f);
                ImGui::ColorEdit3("Color", light.color);
                ImGui::SliderFloat("Intensity", &light.intensity, 0.0f, 50.0f);
                ImGui::SliderFloat("Inner angle", &light.inner_angle, 1.0f, 60.0f);
                ImGui::SliderFloat("Outer angle", &light.outer_angle, 2.0f, 80.0f);
                light.outer_angle = std::max(light.outer_angle, light.inner_angle + 0.1f);

                if (ImGui::Button("Remove spotlight")) {
                    for (uint32_t j = i + 1; j < spot_count; ++j) {
                        spot_lights[j - 1] = spot_lights[j];
                    }
                    --spot_count;
                    ImGui::TreePop();
                    ImGui::PopID();
                    break;
                }
                ImGui::TreePop();
            }
            ImGui::PopID();
        }
    }

    ImGui::Checkbox("Show light visualizers", &show_light_visualizers);
    ImGui::End();

    const float aspect = context.swapchain_extent.height > 0
        ? static_cast<float>(context.swapchain_extent.width) /
              static_cast<float>(context.swapchain_extent.height)
        : 1.0f;

    math::Mat4 projection{};
    math::Mat4 view{};
    if (use_perspective) {
        math::perspective(camera_fov * PI / 180.0f, aspect, 0.1f, 100.0f, projection);
    } else {
        const float half_height = 3.0f;
        math::ortho(-half_height * aspect, half_height * aspect,
                    -half_height, half_height, 0.1f, 100.0f, projection);
    }
    projection[1][1] = -projection[1][1];

    const float yaw = camera_yaw * PI / 180.0f;
    const float pitch = camera_pitch * PI / 180.0f;
    const float camera_direction[3] = {
        std::cos(pitch) * std::cos(yaw),
        std::sin(pitch),
        std::cos(pitch) * std::sin(yaw)
    };
    lookAt(camera_position, camera_direction, view);

    auto* scene = context.scene_uniform_buffer_mapped;
    std::memcpy(scene->projection, projection, sizeof(projection));
    std::memcpy(scene->view, view, sizeof(view));

    for (int i = 0; i < 3; ++i) {
        scene->camera_position[i] = camera_position[i];
        scene->ambient_color_intensity[i] = ambient_color[i];
        scene->directional_direction[i] = directional_direction[i];
        scene->directional_color_intensity[i] = directional_color[i];
    }
    scene->camera_position[3] = 1.0f;
    scene->ambient_color_intensity[3] = ambient_intensity;
    scene->directional_direction[3] = 0.0f;
    scene->directional_color_intensity[3] = directional_intensity;
    scene->light_counts[0] = static_cast<float>(point_count);
    scene->light_counts[1] = static_cast<float>(spot_count);
    scene->light_counts[2] = 0.0f;
    scene->light_counts[3] = 0.0f;

    for (uint32_t i = 0; i < point_count; ++i) {
        PointLightGpu& gpu_light = context.point_lights_mapped[i];
        for (int axis = 0; axis < 3; ++axis) {
            gpu_light.position[axis] = point_lights[i].position[axis];
            gpu_light.color_intensity[axis] = point_lights[i].color[axis];
        }
        gpu_light.position[3] = 1.0f;
        gpu_light.color_intensity[3] = point_lights[i].intensity;
    }

    for (uint32_t i = 0; i < spot_count; ++i) {
        SpotLightGpu& gpu_light = context.spot_lights_mapped[i];
        SpotLight& light = spot_lights[i];
        float direction[3] = {light.direction[0], light.direction[1], light.direction[2]};
        normalize3(direction);

        for (int axis = 0; axis < 3; ++axis) {
            gpu_light.position[axis] = light.position[axis];
            gpu_light.direction[axis] = direction[axis];
            gpu_light.color_intensity[axis] = light.color[axis];
        }
        gpu_light.position[3] = 1.0f;
        gpu_light.direction[3] = 0.0f;
        gpu_light.color_intensity[3] = light.intensity;
        gpu_light.cone[0] = std::cos(light.inner_angle * PI / 180.0f);
        gpu_light.cone[1] = std::cos(light.outer_angle * PI / 180.0f);
        gpu_light.cone[2] = 0.0f;
        gpu_light.cone[3] = 0.0f;
    }

    vmaFlushAllocation(context.allocator, context.scene_uniform_buffer_allocation, 0, VK_WHOLE_SIZE);
    vmaFlushAllocation(context.allocator, context.point_lights_allocation, 0, VK_WHOLE_SIZE);
    vmaFlushAllocation(context.allocator, context.spot_lights_allocation, 0, VK_WHOLE_SIZE);

    for (int i = 0; i < OBJECT_COUNT; ++i) {
        ModelUniform* uniform = context.model_uniform_buffer_mapped[i];
        math::Mat4 model{};
        makeObjectMatrix(objects[i], animation_time, model);
        std::memcpy(uniform->model, model, sizeof(model));

        for (int channel = 0; channel < 3; ++channel) {
            uniform->albedo[channel] = albedo[channel] * objects[i].color[channel];
            uniform->specular_color[channel] = specular_color[channel];
        }
        uniform->albedo[3] = 1.0f;
        uniform->specular_color[3] = 1.0f;
        uniform->material_params[0] = shininess;
        uniform->material_params[1] = 0.0f;
        uniform->material_params[2] = 0.0f;
        uniform->material_params[3] = 0.0f;
    }

    updateLightVisualizers(context);

    for (int i = 0; i < Context::OBJECT_COUNT; ++i) {
        vmaFlushAllocation(context.allocator, context.model_uniform_buffer_allocations[i],
                           0, VK_WHOLE_SIZE);
    }
}

void render(const graphics::internal::FrameData& frame) {
    auto& context = graphics::internal::context;

    const VkCommandBufferBeginInfo begin_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
    };
    vkBeginCommandBuffer(frame.command_buffer, &begin_info);

    const VkClearValue clear_values[] = {
        {.color = {.float32 = {0.025f, 0.03f, 0.045f, 1.0f}}},
        {.depthStencil = {1.0f, 0}}
    };
    const VkRenderPassBeginInfo render_pass = {
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .renderPass = context.render_pass,
        .framebuffer = frame.framebuffer,
        .renderArea = {.extent = context.swapchain_extent},
        .clearValueCount = 2,
        .pClearValues = clear_values
    };

    vkCmdBeginRenderPass(frame.command_buffer, &render_pass, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(frame.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, context.graphics_pipeline);

    const VkViewport viewport = {
        .x = 0.0f,
        .y = 0.0f,
        .width = static_cast<float>(context.swapchain_extent.width),
        .height = static_cast<float>(context.swapchain_extent.height),
        .minDepth = 0.0f,
        .maxDepth = 1.0f
    };
    const VkRect2D scissor = {.extent = context.swapchain_extent};
    vkCmdSetViewport(frame.command_buffer, 0, 1, &viewport);
    vkCmdSetScissor(frame.command_buffer, 0, 1, &scissor);

    const VkDeviceSize offsets[] = {0};
    vkCmdBindVertexBuffers(frame.command_buffer, 0, 1, &context.vertex_buffer, offsets);
    vkCmdBindIndexBuffer(frame.command_buffer, context.index_buffer, 0, VK_INDEX_TYPE_UINT32);

    const auto draw_model = [&](int model_index) {
        const VkDescriptorSet descriptor_sets[] = {
            context.scene_descriptor_set,
            context.model_descriptor_sets[model_index]
        };
        vkCmdBindDescriptorSets(frame.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                context.pipeline_layout, 0, 2, descriptor_sets, 0, nullptr);
        vkCmdDrawIndexed(frame.command_buffer, context.index_count, 1, 0, 0, 0);
    };

    for (int i = 0; i < OBJECT_COUNT; ++i) {
        draw_model(i);
    }

    if (show_light_visualizers) {
        const int point_base = OBJECT_COUNT;
        const int spot_base = point_base + static_cast<int>(graphics::internal::MAX_POINT_LIGHTS);
        const int cone_base = spot_base + static_cast<int>(graphics::internal::MAX_SPOT_LIGHTS);

        for (uint32_t i = 0; i < point_count; ++i) {
            draw_model(point_base + static_cast<int>(i));
        }
        for (uint32_t i = 0; i < spot_count; ++i) {
            draw_model(spot_base + static_cast<int>(i));
        }
        for (uint32_t i = 0; i < spot_count; ++i) {
            for (uint32_t segment = 0; segment < graphics::internal::SPOT_CONE_LINES_PER_LIGHT; ++segment) {
                draw_model(cone_base + static_cast<int>(i * graphics::internal::SPOT_CONE_LINES_PER_LIGHT + segment));
            }
        }
    }

    vkCmdEndRenderPass(frame.command_buffer);
    vkEndCommandBuffer(frame.command_buffer);
}

} // namespace application
