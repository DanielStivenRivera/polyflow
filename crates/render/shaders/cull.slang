// Per-meshlet visibility test: frustum culling of the bounding sphere and
// backface culling of the normal cone. Visible meshlets are appended as
// indexed indirect draws, consumed by vkCmdDrawIndexedIndirectCount.

struct Meshlet
{
    float3 center;
    float radius;
    float3 coneAxis;
    float coneCutoff;
    uint firstIndex;
    uint indexCount;
    uint2 pad;
};

// VkDrawIndexedIndirectCommand
struct DrawCommand
{
    uint indexCount;
    uint instanceCount;
    uint firstIndex;
    int vertexOffset;
    uint firstInstance;
};

static const uint FLAG_BACKFACE_CULLING = 1;

// Must match `CullPushConstants` in mesh.rs (128 bytes, the guaranteed minimum).
struct CullConstants
{
    float4 planes[5]; // left, right, bottom, top, near (far is at infinity)
    float4 eye;       // xyz: camera position
    Meshlet *meshlets;
    DrawCommand *draws;
    uint *counters;   // [0]: draw count, [1]: visible triangles
    uint meshletCount;
    uint flags;
};

[[vk::push_constant]]
ConstantBuffer<CullConstants> pc;

bool isVisible(Meshlet m)
{
    for (uint i = 0; i < 5; i++)
    {
        if (dot(pc.planes[i].xyz, m.center) + pc.planes[i].w < -m.radius)
            return false;
    }
    if ((pc.flags & FLAG_BACKFACE_CULLING) != 0)
    {
        // meshoptimizer's cone test, without a singularity at the center.
        float3 offset = m.center - pc.eye.xyz;
        if (dot(offset, m.coneAxis) >= m.coneCutoff * length(offset) + m.radius)
            return false;
    }
    return true;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void cullMain(uint3 id: SV_DispatchThreadID)
{
    uint index = id.x;
    if (index >= pc.meshletCount)
        return;
    Meshlet m = pc.meshlets[index];
    if (!isVisible(m))
        return;

    uint slot;
    InterlockedAdd(pc.counters[0], 1, slot);
    uint triangles;
    InterlockedAdd(pc.counters[1], m.indexCount / 3, triangles);

    DrawCommand draw;
    draw.indexCount = m.indexCount;
    draw.instanceCount = 1;
    draw.firstIndex = m.firstIndex;
    draw.vertexOffset = 0;
    draw.firstInstance = index; // lets the vertex shader know its meshlet
    pc.draws[slot] = draw;
}
