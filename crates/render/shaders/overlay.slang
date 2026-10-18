// Overlay geometry drawn over the reference mesh: the retopology mesh and tool
// previews. Triangles are filled; lines and points are expanded to
// screen-space quads so their width does not depend on the GPU's line support.
//
// Everything is pulled towards the camera by a fraction of its distance, so
// the retopo mesh stays visible over the surface it is snapped to: its faces
// and edges are flat chords that would otherwise sink under convex areas.

// Must match `OverlayPushConstants` in overlay.rs.
struct PushConstants
{
    float4x4 viewProj;
    float4 eye;          // xyz: camera position
    float2 viewportSize; // pixels
    float lineWidth;     // pixels
    float depthPull;     // fraction of the distance to the eye
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

float4 project(float3 p)
{
    float3 toEye = pc.eye.xyz - p;
    return mul(pc.viewProj, float4(p + toEye * pc.depthPull, 1.0));
}

// Colors arrive as sRGB bytes; the framebuffer expects linear values.
float4 linearColor(float4 srgb)
{
    return float4(pow(srgb.rgb, 2.2), srgb.a);
}

struct ColorOutput
{
    float4 position : SV_Position;
    float4 color : COLOR;
};

// ---- Triangles ------------------------------------------------------------

[shader("vertex")]
ColorOutput triangleVertex([[vk::location(0)]] float3 position : POSITION, [[vk::location(1)]] float4 color : COLOR)
{
    ColorOutput output;
    output.position = project(position);
    output.color = linearColor(color);
    return output;
}

[shader("fragment")]
float4 colorFragment(ColorOutput input) : SV_Target
{
    return input.color;
}

// ---- Lines: one instance per segment, six vertices per instance -------------

static const float2 QUAD[6] = { float2(0, -1), float2(1, -1), float2(1, 1), float2(0, -1), float2(1, 1), float2(0, 1) };

[shader("vertex")]
ColorOutput lineVertex(
    [[vk::location(0)]] float3 a : POSITION0,
    [[vk::location(1)]] float3 b : POSITION1,
    [[vk::location(2)]] float4 color : COLOR,
    uint vertex : SV_VulkanVertexID)
{
    float4 ca = project(a);
    float4 cb = project(b);
    // Clip against the near plane (reverse-Z: z <= w) so segments crossing
    // behind the camera do not flip.
    float da = ca.w - ca.z;
    float db = cb.w - cb.z;
    if (da < 0.0 && db >= 0.0)
        ca = lerp(cb, ca, db / (db - da));
    else if (db < 0.0 && da >= 0.0)
        cb = lerp(ca, cb, da / (da - db));

    float2 sa = ca.xy / ca.w * pc.viewportSize;
    float2 sb = cb.xy / cb.w * pc.viewportSize;
    float2 dir = sb - sa;
    float len = length(dir);
    dir = len > 1e-6 ? dir / len : float2(1, 0);
    float2 normal = float2(-dir.y, dir.x);

    float2 corner = QUAD[vertex];
    float4 c = corner.x < 0.5 ? ca : cb;
    // NDC spans 2 units per viewport size; offset by half the width in pixels.
    c.xy += normal * corner.y * pc.lineWidth / pc.viewportSize * c.w;

    ColorOutput output;
    output.position = c;
    output.color = linearColor(color);
    return output;
}

// ---- Points: one instance per point, six vertices per instance --------------

struct PointOutput
{
    float4 position : SV_Position;
    float4 color : COLOR;
    float2 uv : TEXCOORD;
};

static const float2 CORNERS[6] = { float2(-1, -1), float2(1, -1), float2(1, 1), float2(-1, -1), float2(1, 1), float2(-1, 1) };

[shader("vertex")]
PointOutput pointVertex(
    [[vk::location(0)]] float3 position : POSITION,
    [[vk::location(1)]] float4 color : COLOR,
    [[vk::location(2)]] float size : PSIZE,
    uint vertex : SV_VulkanVertexID)
{
    float4 c = project(position);
    float2 corner = CORNERS[vertex];
    c.xy += corner * size / pc.viewportSize * c.w;
    PointOutput output;
    output.position = c;
    output.color = linearColor(color);
    output.uv = corner;
    return output;
}

[shader("fragment")]
float4 pointFragment(PointOutput input) : SV_Target
{
    float r = length(input.uv);
    if (r > 1.0)
        discard;
    // Dark rim so points read on any background.
    float rim = smoothstep(0.6, 0.85, r);
    return float4(lerp(input.color.rgb, input.color.rgb * 0.25, rim), input.color.a);
}
