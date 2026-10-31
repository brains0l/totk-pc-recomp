// AMD FidelityFX Super Resolution 1 on the present path (gfx_window.c present_image): when the window is larger than
// the game's output, EASU scales the game's crop rectangle to the swapchain size and RCAS sharpens it, in place of the
// bilinear vkCmdBlitImage. TOTK_UPSCALER=fsr|bilinear picks it, TOTK_SHARPNESS (0..100, 0 = no RCAS pass) sets RCAS;
// both apply at the next frame.
//
// Per frame: EASU reads the source through a view of its own format (the same colour conversion the blit does) and
// writes an RGBA16F image of the picture's size on the swapchain (all of it, or the part between black bars,
// gfx_window.c fit_rect); RCAS reads that and writes a second one; a 1:1 blit converts the result into its place in
// the swapchain image, which the caller left in TRANSFER_DST. The shaders (runtime/shaders/fsr_*.comp)
// compute AMD's constants from the push constants. Cost: two compute passes over the swapchain size and one copy per
// present; nothing on the game's draws.
//
// The source view of slot s is destroyed when slot s records again: present_image has waited for that slot's fence.
// The two intermediate images are shared by all slots; each frame's first barrier orders it after the previous
// frame's reads (one queue, so the barrier covers earlier submissions).
//
// Colour grading (TOTK_COLOR_PRESET, TOTK_BRIGHTNESS, TOTK_CONTRAST, TOTK_SATURATION, TOTK_GAMMA; live) is one more
// compute pass, runtime/shaders/color_grade.comp, before the menu overlay draws, so the menu keeps its colours. With FSR
// it grades the FSR result into the other intermediate; without FSR (vk_grade_record) it reads the source over the crop
// bilinearly, as the blit would, and takes the blit's place. Neutral values (the vanilla preset) record nothing.
// Presets and slider ranges as proposed in valentderah/totk-pc-recomp issue 136 (Live Color Profiles & OLED Display
// Tuning).
#include "vk.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "color_grade_spv.h"
#include "fsr_easu_spv.h"
#include "fsr_rcas_spv.h"
#include "host.h"

#define SLOTS 8

static struct {
    int tried, ok;
    PFN_vkCmdPushDescriptorSetKHR push;
    VkDescriptorSetLayout dsl;
    VkPipelineLayout layout;
    VkPipeline easu, rcas, grade;
    VkSampler sampler;
    VkImage img[2];                // EASU output, RCAS output
    VmaAllocation alloc[2];
    VkImageView view[2];
    uint32_t w, h;                 // their size
    VkImageView src_view[SLOTS];   // the source view each slot's last frame read
    int logged, grade_logged;
} F;

static VkPipeline make_pipeline(VkCtx *c, const uint32_t *code, size_t size) {
    VkShaderModuleCreateInfo mi = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = size, .pCode = code};
    VkShaderModule m = VK_NULL_HANDLE;
    if (vkCreateShaderModule(c->device, &mi, NULL, &m) != VK_SUCCESS) return VK_NULL_HANDLE;
    VkComputePipelineCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                      .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = m, .pName = "main"},
                                      .layout = F.layout};
    VkPipeline p = VK_NULL_HANDLE;
    if (vkCreateComputePipelines(c->device, VK_NULL_HANDLE, 1, &ci, NULL, &p) != VK_SUCCESS) p = VK_NULL_HANDLE;
    vkDestroyShaderModule(c->device, m, NULL);
    return p;
}

static int init(VkCtx *c) {
    F.tried = 1;
    F.push = (PFN_vkCmdPushDescriptorSetKHR)vkGetDeviceProcAddr(c->device, "vkCmdPushDescriptorSetKHR");
    if (!F.push) {
        rt_log("vk: FSR off: no VK_KHR_push_descriptor; the present scales bilinearly");
        return 0;
    }
    VkDescriptorSetLayoutBinding b[2] = {
        {.binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
        {.binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT}};
    VkDescriptorSetLayoutCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                           .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR,
                                           .bindingCount = 2, .pBindings = b};
    VkPushConstantRange pr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 48};   // color_grade.comp: 48, FSR: 32
    VkPipelineLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1,
                                     .pSetLayouts = &F.dsl, .pushConstantRangeCount = 1, .pPushConstantRanges = &pr};
    VkSamplerCreateInfo si = {.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, .magFilter = VK_FILTER_LINEAR,
                              .minFilter = VK_FILTER_LINEAR, .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
                              .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                              .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                              .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, .maxLod = 0};
    if (vkCreateDescriptorSetLayout(c->device, &dci, NULL, &F.dsl) != VK_SUCCESS ||
        vkCreatePipelineLayout(c->device, &li, NULL, &F.layout) != VK_SUCCESS ||
        vkCreateSampler(c->device, &si, NULL, &F.sampler) != VK_SUCCESS ||
        !(F.easu = make_pipeline(c, fsr_easu_spv, sizeof fsr_easu_spv)) ||
        !(F.rcas = make_pipeline(c, fsr_rcas_spv, sizeof fsr_rcas_spv)) ||
        !(F.grade = make_pipeline(c, color_grade_spv, sizeof color_grade_spv))) {
        rt_log("vk: FSR off: its pipelines could not be created; the present scales bilinearly");
        return 0;
    }
    rt_log("vk: FSR 1 (EASU + RCAS) ready for the present");
    return F.ok = 1;
}

static void free_images(VkCtx *c) {
    for (int i = 0; i < 2; i++) {
        if (F.view[i]) vkDestroyImageView(c->device, F.view[i], NULL);
        if (F.img[i]) vmaDestroyImage(c->vma, F.img[i], F.alloc[i]);
        F.view[i] = VK_NULL_HANDLE;
        F.img[i] = VK_NULL_HANDLE;
    }
    F.w = F.h = 0;
}

static int ensure_images(VkCtx *c, uint32_t w, uint32_t h) {
    if (F.img[0] && F.w == w && F.h == h) return 1;
    // A swapchain size change: earlier presents may still read the old images. vkDeviceWaitIdle needs every queue
    // externally synchronized, so it holds the queue lock like a submission.
    vk_queue_lock();
    vkDeviceWaitIdle(c->device);
    vk_queue_unlock();
    free_images(c);
    for (int i = 0; i < 2; i++) {
        VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D,
                                .format = VK_FORMAT_R16G16B16A16_SFLOAT, .extent = {w, h, 1}, .mipLevels = 1,
                                .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
                                .usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                                         VK_IMAGE_USAGE_TRANSFER_SRC_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
        if (vmaCreateImage(c->vma, &ci, &ai, &F.img[i], &F.alloc[i], NULL) != VK_SUCCESS) {
            F.img[i] = VK_NULL_HANDLE;
            free_images(c);
            return 0;
        }
        VkImageViewCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = F.img[i],
                                    .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = VK_FORMAT_R16G16B16A16_SFLOAT,
                                    .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        if (vkCreateImageView(c->device, &vi, NULL, &F.view[i]) != VK_SUCCESS) {
            F.view[i] = VK_NULL_HANDLE;
            free_images(c);
            return 0;
        }
    }
    F.w = w;
    F.h = h;
    return 1;
}

static void barrier(VkCommandBuffer cb, VkImage img, VkImageLayout from, VkImageLayout to, VkPipelineStageFlags2 ss,
                    VkAccessFlags2 sa, VkPipelineStageFlags2 ds, VkAccessFlags2 da) {
    VkImageMemoryBarrier2 b = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, .srcStageMask = ss, .srcAccessMask = sa,
                               .dstStageMask = ds, .dstAccessMask = da, .oldLayout = from, .newLayout = to,
                               .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                               .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = img,
                               .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    VkDependencyInfo d = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1,
                          .pImageMemoryBarriers = &b};
    vkCmdPipelineBarrier2(cb, &d);
}

static void bind_pass(VkCommandBuffer cb, VkPipeline p, VkImageView in, VkImageView out, const float *pc, uint32_t pc_size) {
    VkDescriptorImageInfo ii = {F.sampler, in, VK_IMAGE_LAYOUT_GENERAL}, oi = {VK_NULL_HANDLE, out, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet w[2] = {
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 0, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &ii},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 1, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .pImageInfo = &oi}};
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p);
    rec_push(F.push, cb, VK_PIPELINE_BIND_POINT_COMPUTE, F.layout, 0, 2, w);
    vkCmdPushConstants(cb, F.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pc_size, pc);
}

// ---- colour grading
// Presets (TOTK_COLOR_PRESET): {brightness, contrast, saturation, gamma}. custom takes the four sliders.
static const struct { const char *name; float g[4]; } presets[] = {
    {"vanilla", {0.00f, 1.00f, 1.00f, 1.00f}},
    {"vivid", {0.00f, 1.15f, 1.35f, 0.95f}},     // for OLED panels: punchier colours, shadows lifted
    {"clarity", {0.02f, 1.10f, 1.15f, 1.00f}},
};

static float clampf(float v, float lo, float hi) { return v < lo ? lo : v > hi ? hi : v; }

int vk_grade_preset(const char *name, float g[4]) {
    for (size_t i = 0; i < sizeof presets / sizeof *presets; i++)
        if (!strcmp(name, presets[i].name)) {
            memcpy(g, presets[i].g, sizeof presets[i].g);
            return 1;
        }
    return 0;
}

// The grade in effect; 0 when it changes nothing (no pass is recorded then).
static int grade_params(float g[4]) {
    if (!vk_grade_preset(cfg_value("TOTK_COLOR_PRESET"), g)) {   // custom (or an unknown name): the sliders, in the menu's units
        g[0] = clampf((float)cfg_double("TOTK_BRIGHTNESS"), -30.0f, 30.0f) / 100.0f;
        g[1] = clampf((float)cfg_double("TOTK_CONTRAST"), 70.0f, 150.0f) / 100.0f;
        g[2] = clampf((float)cfg_double("TOTK_SATURATION"), 50.0f, 200.0f) / 100.0f;
        g[3] = clampf((float)cfg_double("TOTK_GAMMA"), 0.80f, 1.20f);
    }
    return fabsf(g[0]) > 1e-4f || fabsf(g[1] - 1) > 1e-4f || fabsf(g[2] - 1) > 1e-4f || fabsf(g[3] - 1) > 1e-4f;
}

static void log_grade(const float g[4]) {
    if (F.grade_logged) return;
    F.grade_logged = 1;
    rt_log("vk: colour grade %s: brightness %+.0f%%, contrast %.0f%%, saturation %.0f%%, gamma %.2f",
           cfg_value("TOTK_COLOR_PRESET"), g[0] * 100, g[1] * 100, g[2] * 100, g[3]);
}

// One grading dispatch: in (sampled) to out (storage), both GENERAL, over the output size dw x dh.
static void grade_pass(VkCommandBuffer cb, VkImageView in, VkImageView out, const float map[4], uint32_t dw, uint32_t dh,
                       const float g[4]) {
    float pc[12] = {map[0], map[1], map[2], map[3], (float)dw, (float)dh, 0, 0, g[0], g[1], g[2], g[3]};
    bind_pass(cb, F.grade, in, out, pc, sizeof pc);
    vkCmdDispatch(cb, (dw + 15) / 16, (dh + 15) / 16, 1);
}

// The source view of this slot: the source's own format (the blit's colour conversion), alpha read as 1.
static VkImageView source_view(VkCtx *c, unsigned slot, VkImage src, VkFormat fmt) {
    if (F.src_view[slot]) vkDestroyImageView(c->device, F.src_view[slot], NULL);
    VkImageViewCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = src,
                                .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = fmt,
                                .components = {VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                                               VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_ONE},
                                .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    if (vkCreateImageView(c->device, &vi, NULL, &F.src_view[slot]) != VK_SUCCESS) F.src_view[slot] = VK_NULL_HANDLE;
    return F.src_view[slot];
}

static const VkPipelineStageFlags2 CS = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, XF = VK_PIPELINE_STAGE_2_TRANSFER_BIT;

// The game's writes to the source become visible to the compute read; the earlier frame's reads of the
// intermediates finish before this frame writes them.
static void begin_passes(VkCommandBuffer cb, VkImage src) {
    barrier(cb, src, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            VK_ACCESS_2_MEMORY_WRITE_BIT, CS, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    for (int i = 0; i < 2; i++)
        barrier(cb, F.img[i], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, CS | XF, 0, CS,
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
}

// Intermediate `last` holds the picture: it goes 1:1 into its place on the swapchain image (TRANSFER_DST), and later
// submissions that write the source wait for the compute reads.
static void end_passes(VkCommandBuffer cb, VkImage src, int last, VkImage dst, const int out[4], uint32_t dw, uint32_t dh) {
    barrier(cb, F.img[last], VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL, CS, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            XF, VK_ACCESS_2_TRANSFER_READ_BIT);
    barrier(cb, src, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL, CS, 0, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, 0);
    VkImageBlit region = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                          .srcOffsets = {{0, 0, 0}, {(int)dw, (int)dh, 1}},
                          .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                          .dstOffsets = {{out[0], out[1], 0}, {out[0] + (int)dw, out[1] + (int)dh, 1}}};
    vkCmdBlitImage(cb, F.img[last], VK_IMAGE_LAYOUT_GENERAL, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region,
                   VK_FILTER_NEAREST);
}

int vk_fsr_record(VkCommandBuffer cb, unsigned slot, VkImage src, VkFormat fmt, int sw, int sh, const int crop[4],
                  VkImage dst, const int out[4]) {
    int cw = crop[2], ch = crop[3];
    uint32_t dw = (uint32_t)out[2], dh = (uint32_t)out[3];
    if (strcmp(cfg_value("TOTK_UPSCALER"), "fsr") || (dw <= (uint32_t)cw && dh <= (uint32_t)ch) || !fmt) return 0;
    VkCtx *c = vk_ctx();
    if (!F.tried) init(c);
    if (!F.ok || !ensure_images(c, dw, dh)) return 0;
    slot %= SLOTS;
    VkImageView sv = source_view(c, slot, src, fmt);
    if (!sv) return 0;
    int sharp = cfg_int("TOTK_SHARPNESS");
    if (sharp < 0) sharp = 0;
    if (sharp > 100) sharp = 100;
    if (!F.logged) {
        F.logged = 1;
        rt_log("vk: FSR upscales %dx%d to %ux%u (sharpness %d)", cw, ch, dw, dh, sharp);
    }
    begin_passes(cb, src);
    uint32_t gx = (dw + 15) / 16, gy = (dh + 15) / 16;
    float easu[8] = {(float)cw, (float)ch, (float)sw, (float)sh, (float)dw, (float)dh, (float)crop[0], (float)crop[1]};
    bind_pass(cb, F.easu, sv, F.view[0], easu, sizeof easu);
    vkCmdDispatch(cb, gx, gy, 1);
    int last = 0;
    if (sharp > 0) {
        barrier(cb, F.img[0], VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL, CS, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                CS, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        float rcas[8] = {(float)dw, (float)dh, (float)(100 - sharp) / 50.0f, 0, 0, 0, 0, 0};
        bind_pass(cb, F.rcas, F.view[0], F.view[1], rcas, sizeof rcas);
        vkCmdDispatch(cb, gx, gy, 1);
        last = 1;
    }
    float g[4];
    if (grade_params(g)) {   // the FSR result 1:1 into the other intermediate, which the previous pass may have read
        log_grade(g);
        int next = 1 - last;
        barrier(cb, F.img[last], VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL, CS, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                CS, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        barrier(cb, F.img[next], VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL, CS, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                CS, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        const float map[4] = {0, 0, 1.0f / (float)dw, 1.0f / (float)dh};
        grade_pass(cb, F.view[last], F.view[next], map, dw, dh, g);
        last = next;
    } else {
        F.grade_logged = 0;
    }
    end_passes(cb, src, last, dst, out, dw, dh);
    return 1;
}

int vk_grade_record(VkCommandBuffer cb, unsigned slot, VkImage src, VkFormat fmt, int sw, int sh, const int crop[4],
                    VkImage dst, const int out[4]) {
    float g[4];
    if (!fmt || !grade_params(g)) {
        F.grade_logged = 0;
        return 0;
    }
    uint32_t dw = (uint32_t)out[2], dh = (uint32_t)out[3];
    VkCtx *c = vk_ctx();
    if (!F.tried) init(c);
    if (!F.ok || !ensure_images(c, dw, dh)) return 0;
    slot %= SLOTS;
    VkImageView sv = source_view(c, slot, src, fmt);
    if (!sv) return 0;
    log_grade(g);
    begin_passes(cb, src);
    // the crop rectangle of the source, stretched over the output as the blit would
    const float map[4] = {(float)crop[0] / (float)sw, (float)crop[1] / (float)sh,
                          (float)crop[2] / ((float)dw * (float)sw), (float)crop[3] / ((float)dh * (float)sh)};
    grade_pass(cb, sv, F.view[0], map, dw, dh, g);
    end_passes(cb, src, 0, dst, out, dw, dh);
    return 1;
}

void vk_fsr_release(void) {
    VkCtx *c = vk_ctx();
    if (!c || !F.tried) return;
    free_images(c);
    for (int i = 0; i < SLOTS; i++)
        if (F.src_view[i]) vkDestroyImageView(c->device, F.src_view[i], NULL);
    if (F.easu) vkDestroyPipeline(c->device, F.easu, NULL);
    if (F.rcas) vkDestroyPipeline(c->device, F.rcas, NULL);
    if (F.grade) vkDestroyPipeline(c->device, F.grade, NULL);
    if (F.sampler) vkDestroySampler(c->device, F.sampler, NULL);
    if (F.layout) vkDestroyPipelineLayout(c->device, F.layout, NULL);
    if (F.dsl) vkDestroyDescriptorSetLayout(c->device, F.dsl, NULL);
    memset(&F, 0, sizeof F);
}
