// Shaded rendering of the reference mesh: headlight diffuse with a soft rim,
// a neutral "clay" look that reads sculpt detail well.

static const uint FLAG_SHOW_MESHLETS = 1;

// Must match `MeshPushConstants` in mesh.rs.
struct PushConstants
{
    float4x4 viewProj;
    float4 eye;   // xyz: camera position
    float4 color; // rgb: base color (linear)
    uint flags;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

struct VertexInput
{
    [[vk::location(0)]] float3 position : POSITION;
    // Octahedral-encoded unit normal.
    [[vk::location(1)]] float2 normal : NORMAL;
};

struct VertexOutput
{
    float4 position : SV_Position;
    float3 worldPos : POSITION;
    float3 normal : NORMAL;
    nointerpolation uint meshlet : MESHLET;
};

float3 decodeOctahedral(float2 e)
{
    float3 n = float3(e.x, e.y, 1.0 - abs(e.x) - abs(e.y));
    float t = saturate(-n.z);
    n.x += n.x >= 0.0 ? -t : t;
    n.y += n.y >= 0.0 ? -t : t;
    return normalize(n);
}

[shader("vertex")]
VertexOutput vertexMain(VertexInput input, uint instance : SV_VulkanInstanceID)
{
    VertexOutput output;
    output.position = mul(pc.viewProj, float4(input.position, 1.0));
    output.worldPos = input.position;
    output.normal = decodeOctahedral(input.normal);
    // Each meshlet is drawn as one instance whose firstInstance is its index.
    output.meshlet = instance;
    return output;
}

float3 meshletColor(uint id)
{
    uint h = id * 2654435761u;
    return float3((h >> 8) & 255, (h >> 16) & 255, (h >> 24) & 255) / 255.0 * 0.8 + 0.2;
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
    float3 base = (pc.flags & FLAG_SHOW_MESHLETS) != 0 ? meshletColor(input.meshlet) : pc.color.rgb;
    float3 color = base * diffuse + rim;
    if (!frontFacing)
        color *= float3(0.55, 0.35, 0.35);
    return float4(color, 1.0);
}
