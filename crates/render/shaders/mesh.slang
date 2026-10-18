// Shaded rendering of a triangle mesh: headlight diffuse with a soft rim,
// a neutral "clay" look that reads sculpt detail well.

struct PushConstants
{
    float4x4 viewProj;
    float4 eye;   // xyz: camera position
    float4 color; // rgb: base color (linear)
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

struct VertexInput
{
    [[vk::location(0)]] float3 position : POSITION;
    [[vk::location(1)]] float3 normal : NORMAL;
};

struct VertexOutput
{
    float4 position : SV_Position;
    float3 worldPos : POSITION;
    float3 normal : NORMAL;
};

[shader("vertex")]
VertexOutput vertexMain(VertexInput input)
{
    VertexOutput output;
    output.position = mul(pc.viewProj, float4(input.position, 1.0));
    output.worldPos = input.position;
    output.normal = input.normal;
    return output;
}

[shader("fragment")]
float4 fragmentMain(VertexOutput input, bool frontFacing : SV_IsFrontFace) : SV_Target
{
    float3 n = normalize(input.normal);
    float3 v = normalize(pc.eye.xyz - input.worldPos);
    // Back faces of open meshes are shaded as seen from behind, in a darker tint.
    if (!frontFacing)
        n = -n;
    float ndv = saturate(dot(n, v));
    float diffuse = 0.15 + 0.85 * ndv;
    float rim = pow(1.0 - ndv, 4.0) * 0.25;
    float3 color = pc.color.rgb * diffuse + rim;
    if (!frontFacing)
        color *= float3(0.55, 0.35, 0.35);
    return float4(color, 1.0);
}
