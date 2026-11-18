import bevy_hanabi::render::vfx_common::{DrawIndexedIndirectArgs, EffectMetadata, Spawner};

/// Key-value pair for sorting, with optional second sort key.
struct KeyValuePair {
    /// Sorting key.
    key: u32,
    /// Secondary sorting key. Sorts value with the same primary key.
    key2: u32,
    /// Value associated with the sort key(s), generally an index to some other data.
    /// Copied as is and otherwise ignored by the sorting algorithm.
    value: u32,
}

struct SortBuffer {
    count: i32,
    pairs: array<KeyValuePair>,
}

struct IndirectIndexBuffer {
    data: array<u32>,
}

struct MergeParams {
    max_list_size: u32,
    source_buffer: u32,
}

@group(0) @binding(0) var<storage, read_write> indirect_index_buffer : IndirectIndexBuffer;
@group(0) @binding(1) var<storage, read> sort_buffer : SortBuffer;
// Technically read-only, but the type contains atomic<> fields and wasm is strict about it
@group(0) @binding(2) var<storage, read_write> effect_metadatas : array<EffectMetadata>;
@group(0) @binding(3) var<storage, read_write> spawner : Spawner;
@group(0) @binding(4) var<storage, read> merge_params : MergeParams;
@group(0) @binding(5) var<storage, read_write> draw_indirect_buffer : array<DrawIndexedIndirectArgs>;

/// Copy the sorted particle indices back into the effect index buffer.
@compute @workgroup_size(64)
fn sort_copy(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let row_index = global_invocation_id.x;
    let effect_metadata_index = spawner.effect_metadata_index;
    let count = atomicLoad(&effect_metadatas[effect_metadata_index].alive_count); // TODO - atomic not needed
    if (row_index >= count) {
        return;
    }

    let base_particle = spawner.slab_offset;

    // Write the particle index back into the same ping-pong column index we read it from
    // in vfx_sort_fill. Sorting is optional and shouldn't influence that ping-pong logic.
    let write_index = effect_metadatas[effect_metadata_index].indirect_write_index;

    let source_offset = merge_params.source_buffer * u32(sort_buffer.count);
    let particle_index = sort_buffer.pairs[source_offset + row_index].value;
    indirect_index_buffer.data[(base_particle + row_index) * 3u + write_index] = particle_index;

    // Depth slices: rows are sorted farthest first, so each edge splits them
    // once. The row just before that split records it; if every row is
    // nearer than an edge, the first row does.
    //
    // Each band, nearest first, draws from its own args after the effect's,
    // zeroed by the indirect pass: band b's instances run from edge b's split
    // (0 past the last edge) to edge b - 1's (the count before the first).
    // Each split adds to one band's count and takes from the next.
    let edge_count = spawner.depth_slice_count;
    if (edge_count == 0u) {
        return;
    }
    let band_args = spawner.draw_indirect_index + 1u;
    // The back-to-front sort key is the distance's bits, inverted.
    let distance = bitcast<f32>(~sort_buffer.pairs[source_offset + row_index].key);
    var next_distance = -1.0;
    if (row_index + 1u < count) {
        next_distance = bitcast<f32>(~sort_buffer.pairs[source_offset + row_index + 1u].key);
    }
    for (var edge = 0u; edge < edge_count; edge += 1u) {
        let at = spawner.depth_slice_edges[edge];
        if (distance >= at && next_distance < at) {
            let split = row_index + 1u;
            spawner.depth_slice_rows[edge] = split;
            atomicSub(&draw_indirect_buffer[band_args + edge].instance_count, split);
            atomicAdd(&draw_indirect_buffer[band_args + edge + 1u].instance_count, split);
        } else if (row_index == 0u && distance < at) {
            spawner.depth_slice_rows[edge] = 0u;
        }
    }
    if (row_index == 0u) {
        spawner.depth_slice_rows[edge_count] = count;
        atomicAdd(&draw_indirect_buffer[band_args].instance_count, count);
    }
}
