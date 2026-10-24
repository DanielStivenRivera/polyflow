// GPU baker, tracing rays with hardware ray queries against an acceleration
// structure built from the reference mesh (one geometry, triangles in mesh
// order, opaque, two-sided).

import bake_common;

[[vk::binding(0, 0)]]
RaytracingAccelerationStructure scene;

struct RtTracer : ITracer
{
    Hit closest(float3 origin, float3 dir, float tMax)
    {
        RayDesc ray;
        ray.Origin = origin;
        ray.Direction = dir;
        ray.TMin = 0.0;
        ray.TMax = tMax;
        RayQuery<RAY_FLAG_FORCE_OPAQUE> query;
        query.TraceRayInline(scene, RAY_FLAG_NONE, 0xff, ray);
        query.Proceed();
        Hit hit;
        hit.triangle = NO_TRIANGLE;
        hit.t = tMax;
        hit.u = 0;
        hit.v = 0;
        if (query.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
        {
            float2 bary = query.CommittedTriangleBarycentrics();
            hit.t = query.CommittedRayT();
            hit.triangle = query.CommittedPrimitiveIndex();
            hit.u = bary.x;
            hit.v = bary.y;
        }
        return hit;
    }

    bool occluded(float3 origin, float3 dir, float tMax)
    {
        RayDesc ray;
        ray.Origin = origin;
        ray.Direction = dir;
        ray.TMin = 0.0;
        ray.TMax = tMax;
        RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> query;
        query.TraceRayInline(scene, RAY_FLAG_NONE, 0xff, ray);
        query.Proceed();
        return query.CommittedStatus() == COMMITTED_TRIANGLE_HIT;
    }
};

[shader("compute")]
[numthreads(64, 1, 1)]
void bakeRtMain(uint3 id: SV_DispatchThreadID)
{
    RtTracer tracer;
    bakeTexel(tracer, id);
}
