// The GPU baker's per-texel work, shared by its two entry points:
// `bake.slang` traces rays through the BVH in plain compute, `bake_rt.slang`
// with hardware ray queries. A line-by-line port of `polyflow_bake::cpu`
// (projection, tangent-space encoding, occlusion), so the GPU and CPU bakers
// agree to within float rounding. The CPU rasterizes the UVs and turns the
// sums into maps.

struct Corner
{
    float4 position; // xyz
    float4 normal;   // xyz, unit length
    float4 tangent;  // xyz unit tangent, w bitangent sign
    float4 cage;     // xyz
};

struct Node
{
    float3 min;
    uint first; // leaf: first triangle in `nodeTriangles`; interior: offset of the right child
    float3 max;
    uint count; // triangles in a leaf, 0 for interior nodes
};

static const uint MAP_TANGENT = 1;
static const uint MAP_OBJECT = 2;
static const uint MAP_AO = 4;
static const uint FLAG_DIRECTX = 8;
static const uint FLAG_CAGE = 16;

static const uint NO_TRIANGLE = 0xffffffff;
// `polyflow_spatial::STACK_SIZE`.
static const uint STACK_SIZE = 88;
static const float F32_EPSILON = 1.1920929e-7;
static const float TAU = 6.28318530717958647692;

// Must match `BakeParams` in bake.rs.
struct Params
{
    Corner *corners;
    float4 *refPositions;
    float4 *refNormals;
    uint *refIndices;
    Node *nodes;
    uint *nodeTriangles;
    float2 *aoPoints;
    uint *rasterTriangles; // per texel of the band
    float2 *rasterBary;
    float *sumTangent;     // 3 per texel of the band
    float *sumObject;      // 3 per texel of the band
    float *sumAo;
    uint *hits;
    uint *covered;
    uint width;
    uint bandRows;
    uint flags;
    uint aoCount;
    float frontal;
    float rear;
    float aoDistance;
    float aoBias;
};

// Must match `BakePushConstants` in bake.rs.
struct PushConstants
{
    Params *params;
    uint pass;
    uint firstRow;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

// ---- glam equivalents -----------------------------------------------------------

float3 normalizeOr(float3 v, float3 fallback)
{
    float rcp = 1.0 / length(v);
    return (!isinf(rcp) && !isnan(rcp) && rcp > 0.0) ? v * rcp : fallback;
}

float signum(float x)
{
    // f32::signum: +1 for +0.0, -1 for -0.0.
    return (asuint(x) & 0x80000000) != 0 ? -1.0 : 1.0;
}

float3 anyOrthonormalVector(float3 n)
{
    float sign = signum(n.z);
    float a = -1.0 / (sign + n.z);
    float b = n.x * n.y * a;
    return float3(b, sign + n.y * n.y * a, -n.y);
}

void anyOrthonormalPair(float3 n, out float3 t, out float3 b)
{
    float sign = signum(n.z);
    float a = -1.0 / (sign + n.z);
    float c = n.x * n.y * a;
    t = float3(1.0 + sign * n.x * n.x * a, sign * c, -sign * n.x);
    b = float3(c, sign + n.y * n.y * a, -n.y);
}

// min/max that drop NaN, like f32::min/max.
float nanMin(float a, float b) { return isnan(a) ? b : (isnan(b) ? a : min(a, b)); }
float nanMax(float a, float b) { return isnan(a) ? b : (isnan(b) ? a : max(a, b)); }


// ---- Ray tracing --------------------------------------------------------------

struct Hit
{
    float t;
    uint triangle; // NO_TRIANGLE if nothing was hit
    float u;       // hit point = (1 - u - v) * a + u * b + v * c
    float v;
};

interface ITracer
{
    // Nearest hit with t in [0, tMax].
    Hit closest(float3 origin, float3 dir, float tMax);
    // Whether anything lies on the segment t in [0, tMax].
    bool occluded(float3 origin, float3 dir, float tMax);
};

// ---- Baking (polyflow_bake::cpu) ------------------------------------------------

// `TangentFrame::encode`.
float3 encodeTangent(float3 N, float3 T, float sign, float3 world)
{
    float3 B = sign * cross(N, T);
    float scale = dot(N, N) * dot(T, T);
    // Columns T, B, N; determinant = N . (T x B).
    float3 tmp0 = cross(B, N);
    float3 tmp1 = cross(N, T);
    float3 tmp2 = cross(T, B);
    float det = dot(N, tmp2);
    float3 local;
    if (abs(det) > 1e-6 * scale)
    {
        float invDet = 1.0 / det;
        local = float3(dot(tmp0 * invDet, world), dot(tmp1 * invDet, world), dot(tmp2 * invDet, world));
    }
    else
    {
        float3 n = normalizeOr(N, float3(0, 0, 1));
        float3 rejected = T - n * (dot(T, n) / dot(n, n));
        float3 t = normalizeOr(rejected, anyOrthonormalVector(n));
        float3 b = sign * cross(n, t);
        local = float3(dot(world, t), dot(world, b), dot(world, n));
    }
    return normalizeOr(local, float3(0, 0, 1));
}

// `texel_random`.
uint pcgHash(uint v)
{
    v = v * 747796405u + 2891336453u;
    uint w = ((v >> ((v >> 28) + 4)) ^ v) * 277803737u;
    return (w >> 22) ^ w;
}

float2 texelRandom(uint x, uint y, uint pass)
{
    uint a = pcgHash(x ^ pcgHash(y ^ pcgHash(pass)));
    uint b = pcgHash(a);
    return float2(float(a), float(b)) * (1.0 / 4294967296.0);
}

float visibility<T : ITracer>(T tracer, Params *p, float3 position, float3 normal, float3 faceNormal, uint x, uint y, uint pass)
{
    float3 t, b;
    anyOrthonormalPair(normal, t, b);
    float3 origin = position + faceNormal * p.aoBias;
    float2 shift = texelRandom(x, y, pass);
    uint blocked = 0;
    for (uint i = 0; i < p.aoCount; i++)
    {
        float2 s = p.aoPoints[i] + shift;
        s = s - trunc(s);
        float r = sqrt(s.x);
        float angle = TAU * s.y;
        float3 dir = t * (r * cos(angle)) + b * (r * sin(angle)) + normal * sqrt(max(1.0 - s.x, 0.0));
        float below = dot(dir, faceNormal);
        if (below < 0.0)
            dir -= faceNormal * (2.0 * below);
        if (tracer.occluded(origin, dir, p.aoDistance))
            blocked += 1;
    }
    return 1.0 - float(blocked) / float(p.aoCount);
}

// Adds sample `pc.pass` of texel `id.x` of the band to the sums.
void bakeTexel<T : ITracer>(T tracer, uint3 id)
{
    Params *p = pc.params;
    uint i = id.x;
    if (i >= p.width * p.bandRows)
        return;
    uint tri = p.rasterTriangles[i];
    if (tri == NO_TRIANGLE)
        return;
    uint x = i % p.width;
    uint y = pc.firstRow + i / p.width;
    p.covered[i] += 1;

    // `BakeMesh::sample`.
    float2 bary = p.rasterBary[i];
    float3 w = float3(1.0 - bary.x - bary.y, bary.x, bary.y);
    Corner c0 = p.corners[tri * 3];
    Corner c1 = p.corners[tri * 3 + 1];
    Corner c2 = p.corners[tri * 3 + 2];
    float3 position = c0.position.xyz * w.x + c1.position.xyz * w.y + c2.position.xyz * w.z;
    float3 frameNormal = c0.normal.xyz * w.x + c1.normal.xyz * w.y + c2.normal.xyz * w.z;
    float4 tangent = c0.tangent * w.x + c1.tangent * w.y + c2.tangent * w.z;
    float3 cagePoint = c0.cage.xyz * w.x + c1.cage.xyz * w.y + c2.cage.xyz * w.z;
    float sign = tangent.w < 0.0 ? -1.0 : 1.0;
    float3 normal = normalizeOr(frameNormal, float3(0, 0, 0));

    // `ray`: from the cage point through the surface, or along the normal.
    bool valid = false;
    float3 origin = float3(0, 0, 0);
    float3 dir = float3(0, 0, 0);
    float reach = 0;
    bool nonzero = any(normal != float3(0, 0, 0));
    if ((p.flags & FLAG_CAGE) != 0)
    {
        float3 toSurface = position - cagePoint;
        float len = length(toSurface);
        float3 ap = abs(position);
        if (len > 1e-6 * (max(max(ap.x, ap.y), ap.z) + 1.0))
        {
            valid = true;
            origin = cagePoint;
            dir = toSurface / len;
            reach = len + p.rear;
        }
        else if (nonzero)
        {
            valid = true;
            origin = position;
            dir = -normal;
            reach = p.rear;
        }
    }
    else if (nonzero)
    {
        valid = true;
        origin = position + normal * p.frontal;
        dir = -normal;
        reach = p.frontal + p.rear;
    }

    Hit hit;
    hit.triangle = NO_TRIANGLE;
    if (valid)
        hit = tracer.closest(origin, dir, reach);
    if (hit.triangle == NO_TRIANGLE)
    {
        // Until something is hit, the object map shows the low-poly.
        if (p.hits[i] == 0 && (p.flags & MAP_OBJECT) != 0)
        {
            p.sumObject[i * 3] += normal.x;
            p.sumObject[i * 3 + 1] += normal.y;
            p.sumObject[i * 3 + 2] += normal.z;
        }
        return;
    }
    bool firstHit = p.hits[i] == 0;
    p.hits[i] += 1;

    // `SurfacePoint::at`.
    uint ia = p.refIndices[hit.triangle * 3];
    uint ib = p.refIndices[hit.triangle * 3 + 1];
    uint ic = p.refIndices[hit.triangle * 3 + 2];
    float3 hw = float3(1.0 - hit.u - hit.v, hit.u, hit.v);
    float3 pa = p.refPositions[ia].xyz;
    float3 pb = p.refPositions[ib].xyz;
    float3 pc3 = p.refPositions[ic].xyz;
    float3 surfacePosition = pa * hw.x + pb * hw.y + pc3 * hw.z;
    float3 face = normalizeOr(cross(pb - pa, pc3 - pa), float3(0, 0, 0));
    float3 smooth = p.refNormals[ia].xyz * hw.x + p.refNormals[ib].xyz * hw.y + p.refNormals[ic].xyz * hw.z;
    float3 surfaceNormal = normalizeOr(smooth, face);
    float3 faceNormal = dot(face, surfaceNormal) < 0.0 ? -face : face;
    faceNormal = normalizeOr(faceNormal, surfaceNormal);

    if ((p.flags & MAP_TANGENT) != 0)
    {
        float3 n = encodeTangent(frameNormal, tangent.xyz, sign, surfaceNormal);
        if ((p.flags & FLAG_DIRECTX) != 0)
            n.y = -n.y;
        p.sumTangent[i * 3] += n.x;
        p.sumTangent[i * 3 + 1] += n.y;
        p.sumTangent[i * 3 + 2] += n.z;
    }
    if ((p.flags & MAP_OBJECT) != 0)
    {
        if (firstHit)
        {
            // Drop the low-poly normals summed by earlier misses.
            p.sumObject[i * 3] = 0;
            p.sumObject[i * 3 + 1] = 0;
            p.sumObject[i * 3 + 2] = 0;
        }
        p.sumObject[i * 3] += surfaceNormal.x;
        p.sumObject[i * 3 + 1] += surfaceNormal.y;
        p.sumObject[i * 3 + 2] += surfaceNormal.z;
    }
    if ((p.flags & MAP_AO) != 0)
        p.sumAo[i] += visibility(tracer, p, surfacePosition, surfaceNormal, faceNormal, x, y, pc.pass);
}
