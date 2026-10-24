// GPU baker, tracing rays through the reference mesh's BVH in plain compute:
// the fallback for GPUs without hardware ray queries, and a port of
// `polyflow_spatial::Bvh` (same traversal, same triangle test).

import bake_common;

// ---- Reference mesh queries (polyflow_spatial) ----------------------------------

void triangleVertices(Params *p, uint tri, out float3 a, out float3 b, out float3 c)
{
    a = p.refPositions[p.refIndices[tri * 3]].xyz;
    b = p.refPositions[p.refIndices[tri * 3 + 1]].xyz;
    c = p.refPositions[p.refIndices[tri * 3 + 2]].xyz;
}

// Möller–Trumbore, two-sided; false for no hit. `triangle::intersect_ray`.
bool intersectRay(float3 origin, float3 dir, float3 a, float3 b, float3 c, out float t, out float u, out float v)
{
    t = 0;
    u = 0;
    v = 0;
    float3 e1 = b - a;
    float3 e2 = c - a;
    float3 p = cross(dir, e2);
    float det = dot(e1, p);
    if (abs(det) <= F32_EPSILON * max(dot(e1, e1), dot(e2, e2)) * length(dir))
        return false;
    float invDet = 1.0 / det;
    float3 s = origin - a;
    u = dot(s, p) * invDet;
    if (!(u >= 0.0 && u <= 1.0))
        return false;
    float3 q = cross(s, e1);
    v = dot(dir, q) * invDet;
    if (v < 0.0 || u + v > 1.0)
        return false;
    t = dot(e2, q) * invDet;
    return true;
}

// Entry distance into the node's box, or -1 if the ray misses it within tMax.
float rayBox(float3 origin, float3 invDir, Node node, float tMax)
{
    float3 t0 = (node.min - origin) * invDir;
    float3 t1 = (node.max - origin) * invDir;
    float near = nanMax(nanMax(nanMin(t0.x, t1.x), nanMin(t0.y, t1.y)), nanMin(t0.z, t1.z));
    near = nanMax(near, 0.0);
    float far = nanMin(nanMin(nanMax(t0.x, t1.x), nanMax(t0.y, t1.y)), nanMax(t0.z, t1.z));
    far = nanMin(far, tMax);
    return near <= far ? near : -1.0;
}

// Nearest hit within [0, tMax]; triangle NO_TRIANGLE if none. `Bvh::raycast`.
Hit raycastBvh(Params *p, float3 origin, float3 dir, float tMax)
{
    Hit best;
    best.t = tMax;
    best.triangle = NO_TRIANGLE;
    best.u = 0;
    best.v = 0;
    float3 invDir = 1.0 / dir;
    uint stack[STACK_SIZE];
    stack[0] = 0;
    uint len = rayBox(origin, invDir, p.nodes[0], best.t) >= 0.0 ? 1 : 0;
    while (len > 0)
    {
        len -= 1;
        uint index = stack[len];
        Node node = p.nodes[index];
        if (node.count > 0)
        {
            for (uint k = 0; k < node.count; k++)
            {
                uint tri = p.nodeTriangles[node.first + k];
                float3 a, b, c;
                triangleVertices(p, tri, a, b, c);
                float t, u, v;
                if (intersectRay(origin, dir, a, b, c, t, u, v) && t >= 0.0 && t <= best.t)
                {
                    best.t = t;
                    best.triangle = tri;
                    best.u = u;
                    best.v = v;
                }
            }
            continue;
        }
        uint left = index + 1;
        uint right = index + node.first;
        float l = rayBox(origin, invDir, p.nodes[left], best.t);
        float r = rayBox(origin, invDir, p.nodes[right], best.t);
        // Push the farther child first so the nearer one is visited first.
        if (l >= 0.0 && r >= 0.0)
        {
            uint near = l <= r ? left : right;
            uint far = l <= r ? right : left;
            stack[len] = far;
            stack[len + 1] = near;
            len += 2;
        }
        else if (l >= 0.0)
        {
            stack[len] = left;
            len += 1;
        }
        else if (r >= 0.0)
        {
            stack[len] = right;
            len += 1;
        }
    }
    return best;
}

// Whether anything lies on the segment. `Bvh::occluded`.
bool occludedBvh(Params *p, float3 origin, float3 dir, float tMax)
{
    float3 invDir = 1.0 / dir;
    uint stack[STACK_SIZE];
    stack[0] = 0;
    uint len = rayBox(origin, invDir, p.nodes[0], tMax) >= 0.0 ? 1 : 0;
    while (len > 0)
    {
        len -= 1;
        uint index = stack[len];
        Node node = p.nodes[index];
        if (node.count > 0)
        {
            for (uint k = 0; k < node.count; k++)
            {
                float3 a, b, c;
                triangleVertices(p, p.nodeTriangles[node.first + k], a, b, c);
                float t, u, v;
                if (intersectRay(origin, dir, a, b, c, t, u, v) && t >= 0.0 && t <= tMax)
                    return true;
            }
            continue;
        }
        uint children[2] = { index + 1, index + node.first };
        for (uint k = 0; k < 2; k++)
        {
            if (rayBox(origin, invDir, p.nodes[children[k]], tMax) >= 0.0)
            {
                stack[len] = children[k];
                len += 1;
            }
        }
    }
    return false;
}

struct BvhTracer : ITracer
{
    Params *p;

    Hit closest(float3 origin, float3 dir, float tMax)
    {
        return raycastBvh(p, origin, dir, tMax);
    }

    bool occluded(float3 origin, float3 dir, float tMax)
    {
        return occludedBvh(p, origin, dir, tMax);
    }
};

[shader("compute")]
[numthreads(64, 1, 1)]
void bakeMain(uint3 id: SV_DispatchThreadID)
{
    BvhTracer tracer;
    tracer.p = pc.params;
    bakeTexel(tracer, id);
}
