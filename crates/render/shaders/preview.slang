// Bake preview: the low-poly as baked, shaded with its tangent-space normal
// map the way a MikkTSpace engine does (interpolated normal and tangent left
// unnormalized, bitangent per pixel) and darkened by its occlusion.

// Must match `PreviewPushConstants` in preview.rs.
struct PushConstants
{
    float4x4 viewProj;
    float4 eye;
    uint mode;    // 0 lit, 1 normal map colors, 2 occlusion
    uint directx; // green down in the normal map
    uint2 pad;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]]
Sampler2D normalMap;
[[vk::binding(1, 0)]]
Sampler2D occlusionMap;

struct VertexOutput
{
    float4 position : SV_Position;
    float3 world : POSITION;
    float3 normal : NORMAL;
    float4 tangent : TANGENT;
    float2 uv : TEXCOORD;
};

[shader("vertex")]
VertexOutput previewVertex(
    [[vk::location(0)]] float3 position : POSITION,
    [[vk::location(1)]] float3 normal : NORMAL,
    [[vk::location(2)]] float4 tangent : TANGENT,
    [[vk::location(3)]] float2 uv : TEXCOORD)
{
    VertexOutput output;
    output.position = mul(pc.viewProj, float4(position, 1.0));
    output.world = position;
    output.normal = normal;
    output.tangent = tangent;
    // Maps are stored from the top row down; UV v = 0 is the bottom.
    output.uv = float2(uv.x, 1.0 - uv.y);
    return output;
}

[shader("fragment")]
float4 previewFragment(VertexOutput input) : SV_Target
{
    float3 encoded = normalMap.Sample(input.uv).rgb;
    float occlusion = occlusionMap.Sample(input.uv).r;
    if (pc.mode == 1)
        return float4(encoded, 1.0);
    if (pc.mode == 2)
        return float4(occlusion.xxx, 1.0);

    float3 local = encoded * 2.0 - 1.0;
    if (pc.directx != 0)
        local.y = -local.y;
    float3 N = input.normal;
    float3 T = input.tangent.xyz;
    float sign = input.tangent.w < 0.0 ? -1.0 : 1.0;
    float3 B = sign * cross(N, T);
    float3 n = normalize(local.x * T + local.y * B + local.z * N);

    // A headlight plus a light from above, so detail reads from any angle.
    float3 toEye = normalize(pc.eye.xyz - input.world);
    float key = saturate(dot(n, toEye));
    float top = saturate(dot(n, normalize(float3(0.3, 1.0, 0.4)))) * 0.35;
    float3 albedo = float3(0.62, 0.6, 0.58);
    float3 color = albedo * (0.12 * occlusion + (key * 0.8 + top) * lerp(0.35, 1.0, occlusion));
    return float4(color, 1.0);
}
