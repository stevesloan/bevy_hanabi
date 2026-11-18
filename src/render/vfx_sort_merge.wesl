/// Key-value pair to sort, with optional secondary key.
///
/// Sorting operates on the key (and the secondary key as discriminant, if present,
/// and if the primary keys are equal). The value is carried over alongside the key(s),
/// unmodified. It generally represents or indexes the payload associated with the key(s).
struct KeyValuePair {
    /// Sorting key.
    key: u32,
    /// Secondary sorting key. Used for values with equal primary key.
    @if(HAS_DUAL_KEY)
    key2: u32,
    /// Value associated with the sort key(s), generally an index to some other data.
    /// Copied as is and otherwise ignored by the sorting algorithm.
    value: u32,
}

/// Buffer of key-value pairs to sort.
struct SortBuffer {
    /// Number of items in the buffer.
    count: i32,
    /// Pairs to sort. On output, contains the pairs in sorted order.
    pairs: array<KeyValuePair>,
}

/// Check whether kv1 > kv2, comparing the key(s) of each pair.
fn compare_greater(kv1: KeyValuePair, kv2: KeyValuePair) -> bool {
    if (kv1.key > kv2.key) {
        return true;
    }
    @if(HAS_DUAL_KEY)
    if (kv1.key == kv2.key) {
        return kv1.key2 > kv2.key2;
    }
    return false;
}

struct MergeParams {
    /// Maximum number of items per list for this pass.
    max_list_size: u32,
    /// Half of the ping-pong buffer to read from: 0 for the first half, 1 for
    /// the second half.
    source_buffer: u32,
}

@group(0) @binding(0) var<storage, read_write> sort_buffer : SortBuffer;
@group(0) @binding(1) var<storage, read> merge_params : MergeParams;

/// Threads a row of a merge pass's dispatch holds: the dispatch is
/// `(min(n, 65535), ceil(n / 65535), 1)` workgroups for `n` of them, so it
/// fits the per-dimension limit at any capacity.
const MAX_WORKGROUPS_PER_ROW: u32 = 65535u;

/// The rank of `kv` among the pairs `[lo, hi)` of `src`: how many sort
/// before it. `after_ties`: pairs equal to it count as before it.
fn rank_in(kv: KeyValuePair, lo_in: u32, hi_in: u32, src: u32, after_ties: bool) -> u32 {
    var lo = lo_in;
    var hi = hi_in;
    while (lo < hi) {
        let mid = (lo + hi) / 2u;
        let other = sort_buffer.pairs[src + mid];
        var before: bool;
        if (after_ties) {
            before = !compare_greater(other, kv);
        } else {
            before = compare_greater(kv, other);
        }
        if (before) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    return lo - lo_in;
}

/// One merge pass: every pair of consecutive sorted lists of up to
/// `max_list_size` items becomes one sorted list in the other ping-pong half.
///
/// One thread per item, each writing it at its rank in the merged list: its
/// index in its own list plus its rank in the other, found by binary search.
/// Equal pairs keep the first list's before the second's, so the merge is
/// stable. An unpaired last list is copied as it is.
@compute @workgroup_size(64)
fn merge_sort(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let index = global_invocation_id.y * MAX_WORKGROUPS_PER_ROW * 64u + global_invocation_id.x;
    let total_num_items = u32(sort_buffer.count);
    if (index >= total_num_items) {
        return;
    }
    let list_size = merge_params.max_list_size;
    let src = merge_params.source_buffer * total_num_items;
    let dst = (1u - merge_params.source_buffer) * total_num_items;

    let start_a = (index / (2u * list_size)) * (2u * list_size);
    let start_b = min(start_a + list_size, total_num_items);
    let end_b = min(start_b + list_size, total_num_items);
    let kv = sort_buffer.pairs[src + index];
    var rank: u32;
    if (index < start_b) {
        rank = (index - start_a) + rank_in(kv, start_b, end_b, src, false);
    } else {
        rank = (index - start_b) + rank_in(kv, start_a, start_b, src, true);
    }
    sort_buffer.pairs[dst + start_a + rank] = kv;
}
