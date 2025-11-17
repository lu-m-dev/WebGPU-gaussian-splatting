const SH_C0: f32 = 0.28209479177387814;
const SH_C1 = 0.4886025119029199;
const SH_C2 = array<f32,5>(
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
);
const SH_C3 = array<f32,7>(
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
);

override workgroupSize: u32;
override sortKeyPerThread: u32;

struct DispatchIndirect {
    dispatch_x: atomic<u32>,
    dispatch_y: u32,
    dispatch_z: u32,
}

struct SortInfos {
    keys_size: atomic<u32>,  // instance_count in DrawIndirect
    //data below is for info inside radix sort 
    padded_size: u32, 
    passes: u32,
    even_pass: u32,
    odd_pass: u32,
}

//TODO: bind your data here
@group(2) @binding(0)
var<storage, read_write> sort_infos: SortInfos;
@group(2) @binding(1)
var<storage, read_write> sort_depths : array<u32>;
@group(2) @binding(2)
var<storage, read_write> sort_indices : array<u32>;
@group(2) @binding(3)
var<storage, read_write> sort_dispatch: DispatchIndirect;

@group(0) @binding(0)
var<uniform> camera: CameraUniforms;
@group(0) @binding(1)
var<storage, read> gaussians : array<Gaussian>;
@group(0) @binding(2)
var<storage, read> sh_buf : array<u32>;

@group(1) @binding(0)
var<storage, read_write> splats : array<Splat>;
@group(1) @binding(1)
var<uniform> settings: RenderSettings;

/// reads the ith sh coef from the storage buffer 
fn sh_coef(splat_idx: u32, c_idx: u32) -> vec3<f32> {
    //TODO: access your binded sh_coeff, see load.ts for how it is stored
    let resid = c_idx % 2u;
    let ind = splat_idx * 24u + (c_idx >> 1u) * 3u + resid;
    let rg = unpack2x16float(sh_buf[ind]);
    let ba = unpack2x16float(sh_buf[ind + 1u]);
    if (resid == 0u) {
        return vec3<f32>(rg, ba.x);
    }
    return vec3<f32>(rg.y, ba);
}

// spherical harmonics evaluation with Condon–Shortley phase
fn computeColorFromSH(dir: vec3<f32>, v_idx: u32, sh_deg: u32) -> vec3<f32> {
    var result = SH_C0 * sh_coef(v_idx, 0u);

    if sh_deg > 0u {

        let x = dir.x;
        let y = dir.y;
        let z = dir.z;

        result += - SH_C1 * y * sh_coef(v_idx, 1u) + SH_C1 * z * sh_coef(v_idx, 2u) - SH_C1 * x * sh_coef(v_idx, 3u);

        if sh_deg > 1u {

            let xx = dir.x * dir.x;
            let yy = dir.y * dir.y;
            let zz = dir.z * dir.z;
            let xy = dir.x * dir.y;
            let yz = dir.y * dir.z;
            let xz = dir.x * dir.z;

            result += SH_C2[0] * xy * sh_coef(v_idx, 4u) + SH_C2[1] * yz * sh_coef(v_idx, 5u) + SH_C2[2] * (2.0 * zz - xx - yy) * sh_coef(v_idx, 6u) + SH_C2[3] * xz * sh_coef(v_idx, 7u) + SH_C2[4] * (xx - yy) * sh_coef(v_idx, 8u);

            if sh_deg > 2u {
                result += SH_C3[0] * y * (3.0 * xx - yy) * sh_coef(v_idx, 9u) + SH_C3[1] * xy * z * sh_coef(v_idx, 10u) + SH_C3[2] * y * (4.0 * zz - xx - yy) * sh_coef(v_idx, 11u) + SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh_coef(v_idx, 12u) + SH_C3[4] * x * (4.0 * zz - xx - yy) * sh_coef(v_idx, 13u) + SH_C3[5] * z * (xx - yy) * sh_coef(v_idx, 14u) + SH_C3[6] * x * (xx - 3.0 * yy) * sh_coef(v_idx, 15u);
            }
        }
    }
    result += 0.5;

    return  max(vec3<f32>(0.), result);
}

@compute @workgroup_size(workgroupSize,1,1)
fn preprocess(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) wgs: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&gaussians)) {
        return;
    }

    let gaussian = gaussians[idx];
    let xy = unpack2x16float(gaussian.pos_opacity[0]);
    let zw = unpack2x16float(gaussian.pos_opacity[1]);
    let pos_world = vec4<f32>(xy.x, xy.y, zw.x, 1.0f);
    let pos_view = camera.view * pos_world;
    let pos_clip = camera.proj * pos_view;
    let pos_ndc = pos_clip.xyz / pos_clip.w;

    if (abs(pos_ndc.x) > 1.2f || abs(pos_ndc.y) > 1.2f || pos_view.z < 0.0f) {
        return;
    }

    // jacobian
    let x_view = pos_view.x;
    let y_view = pos_view.y;
    let z_view = pos_view.z;
    let fx = camera.focal.x;
    let fy = camera.focal.y;
    let mat_jacobian = mat3x3<f32>(
        fx / z_view, 0.0f, -fx * x_view / (z_view * z_view),
        0.0f, fy / z_view, -fy * y_view / (z_view * z_view),
        0.0f, 0.0f, 0.0f,
    );

    // scaling
    let s01 = unpack2x16float(gaussian.scale[0]);
    let s23 = unpack2x16float(gaussian.scale[1]);
    let scale = exp(vec3<f32>(s01.x, s01.y, s23.x)) * settings.gaussian_scaling;
    let mat_scale = mat3x3<f32>(
        scale.x, 0.0f, 0.0f,
        0.0f, scale.y, 0.0f,
        0.0f, 0.0f, scale.z,
    );

    // rotation
    let quaternion = normalize(
        vec4<f32>(
            unpack2x16float(gaussian.rot[0]),
            unpack2x16float(gaussian.rot[1]),
        )
    );
    let x = quaternion.y;
    let y = quaternion.z;
    let z = quaternion.w;
    let w = quaternion.x;
    let mat_rotation = mat3x3<f32>(
        1.f - 2.f * (y * y + z * z), 2.f * (x * y - w * z), 2.f * (x * z + w * y),
        2.f * (x * y + w * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - w * x),
        2.f * (x * z - w * y), 2.f * (y * z + w * x), 1.f - 2.f * (x * x + y * y),
    );

    // cov3d
    let mat_m = mat_scale * mat_rotation;
    let mat_cov3d = transpose(mat_m) * mat_m;

    // cov2d
    let mat_view = mat3x3<f32>(camera.view[0].xyz, camera.view[1].xyz, camera.view[2].xyz);
    let mat_t = mat_jacobian * mat_view;
    let mat_cov = mat_t * mat_cov3d * transpose(mat_t);
    let cov = vec3<f32>(
        mat_cov[0][0] + 0.3f,
        mat_cov[0][1],
        mat_cov[1][1] + 0.3f,
    );
    
    // radius
	let det = (cov.x * cov.z - cov.y * cov.y);
	if (det == 0.0f) {
        return;
    }
    let mid = 0.5f * (cov.x + cov.z);
	let lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	let lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	let radius = ceil(3.f * sqrt(max(lambda1, lambda2)));

	let det_inv = 1.0f / det;
	let conic = vec3<f32>(cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv);

    let rgb = computeColorFromSH(normalize(pos_view.xyz), idx, u32(settings.sh_deg));
    let opacity = 1.0f / (1.0f + exp(-zw.y));

    let write_idx = atomicAdd(&sort_infos.keys_size, 1u);

    sort_indices[write_idx] = write_idx;
    sort_depths[write_idx] = 0xFFFFFFFFu - bitcast<u32>(-pos_view.z);

    splats[write_idx].center = pack2x16float(pos_ndc.xy);
    splats[write_idx].size = pack2x16float(0.5f * camera.viewport);
    splats[write_idx].rgba[0] = pack2x16float(rgb.xy);
    splats[write_idx].rgba[1] = pack2x16float(vec2<f32>(rgb.z, opacity));
    splats[write_idx].conic[0] = pack2x16float(conic.xy);
    splats[write_idx].conic[1] = pack2x16float(vec2<f32>(conic.z, radius));

    let keys_per_dispatch = workgroupSize * sortKeyPerThread; 
    // increment DispatchIndirect.dispatchx each time you reach limit for one dispatch of keys
    if ((write_idx % keys_per_dispatch) == 0u) {
        _ = atomicAdd(&sort_dispatch.dispatch_x, 1u);
    }
}