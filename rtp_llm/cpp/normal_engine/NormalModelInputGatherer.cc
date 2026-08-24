#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include "torch/all.h"
#include "rtp_llm/cpp/cache/CacheGroupTagOrder.h"
#include "rtp_llm/cpp/cache/Types.h"
#include "rtp_llm/cpp/models/ModelTypes.h"
#include "rtp_llm/cpp/multimodal_processor/MultimodalInputUtils.h"
#include "rtp_llm/cpp/normal_engine/NormalModelInputGatherer.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/StatusUtil.h"

namespace rtp_llm {

namespace {

bool asyncDebugEnabled() {
    const char* env = std::getenv("RTP_LLM_ASYNC_DEBUG");
    return env != nullptr && std::string(env) == "1";
}

bool deviceInputEnabled() {
    const char* env = std::getenv("RTP_LLM_DEVICE_INPUT");
    return env != nullptr && std::string(env) == "1";
}

struct GatherModelInputContext {
    int                      input_vocab_size;
    bool                     need_cal_position_id;
    size_t                   max_blocks_num;
    int*                     merged_tokens;
    int*                     input_lengths;
    int*                     lm_output_indexes;
    int*                     combo_position_ids;
    GroupOrdinalBlockIdPair* kv_cache_update_mapping;
    int                      batch_idx;
    int*                     sequence_lengths;
    bool                     has_multimodal_input;
    bool                     has_mm_extra_input;
    size_t                   total_decode_batch_size;
    int*                     prefix_lengths;
    int*                     prefix_lengths_host;
    int*                     merged_text_mask;
    int*                     mm_features_locs;
    int                      token_idx;
    int                      cum_output_seq_len;
    int                      mm_feature_index;
};

enum class GatherContextMode {
    DECODE,
    CONTEXT
};

GatherModelInputContext createGatherContext(const NormalModelInputGathererConfig& config,
                                            GptModelInputs&                       model_input,
                                            const StreamGroups&                   stream_groups,
                                            GatherContextMode                     mode) {
    GatherModelInputContext ctx{};
    ctx.input_vocab_size =
        config.input_vocab_size ? static_cast<int>(config.input_vocab_size) : static_cast<int>(config.vocab_size);
    ctx.need_cal_position_id =
        (config.mm_position_ids_style != PositionIdsStyle::DEFAULT) || config.has_positional_encoding;
    ctx.max_blocks_num       = stream_groups.curBlocksNum();
    ctx.merged_tokens        = model_input.combo_tokens.data_ptr<int32_t>();
    ctx.input_lengths        = model_input.input_lengths.data_ptr<int32_t>();
    ctx.sequence_lengths     = model_input.sequence_lengths.data_ptr<int32_t>();
    ctx.combo_position_ids   = ctx.need_cal_position_id ? model_input.combo_position_ids.data_ptr<int32_t>() : nullptr;
    ctx.has_multimodal_input = config.is_multimodal && stream_groups.has_multimodal_input();
    ctx.has_mm_extra_input   = config.is_multimodal && stream_groups.hasMMExtraInput();
    ctx.prefix_lengths       = model_input.prefix_lengths.data_ptr<int32_t>();
    ctx.prefix_lengths_host  = nullptr;
    ctx.merged_text_mask     = ctx.has_multimodal_input ? model_input.text_tokens_mask.data_ptr<int32_t>() : nullptr;
    ctx.mm_features_locs     = ctx.has_multimodal_input ? model_input.mm_features_locs.data_ptr<int32_t>() : nullptr;

    size_t kv_cache_mapping_offset = 0;
    if (mode == GatherContextMode::DECODE) {
        ctx.batch_idx = 0;
    } else {
        ctx.total_decode_batch_size = stream_groups.totalDecodeBatchSize();
        ctx.batch_idx               = static_cast<int>(ctx.total_decode_batch_size);
        ctx.token_idx               = ctx.batch_idx;
        ctx.mm_feature_index        = 0;
        kv_cache_mapping_offset     = stream_groups.decodeBlockUpdateCopyNum();
    }
    ctx.kv_cache_update_mapping =
        model_input.kv_cache_update_mapping.defined() ?
            reinterpret_cast<GroupOrdinalBlockIdPair*>(model_input.kv_cache_update_mapping.data_ptr())
                + kv_cache_mapping_offset :
            nullptr;

    if (ctx.merged_text_mask) {
        size_t current_tokens_size = stream_groups.modelExecuteTokenSize();
        std::fill(ctx.merged_text_mask, ctx.merged_text_mask + current_tokens_size, 1);
    }

    return ctx;
}

BlockIndicesType projectKernelBlocks(const GroupBlockToPoolBlockBinding& binding, const CacheGroup& group) {
    PoolBlockToKernelBlockProjection projection(group.seq_size_per_block, group.kernel_seq_size_per_block);
    const size_t                     kernel_blocks_per_pool_block = group.kernelBlocksPerPoolBlock();
    BlockIndicesType                 projected;
    projected.reserve(projection.projectedSize(binding.size()));
    for (size_t group_block_position = 0; group_block_position < binding.size(); ++group_block_position) {
        const auto pool_block_id = binding.lookup(GroupBlockPosition{group_block_position});
        if (!pool_block_id.has_value()) {
            projected.insert(projected.end(), kernel_blocks_per_pool_block, NULL_BLOCK_IDX);
            continue;
        }
        projection.append(*pool_block_id, projected);
    }
    return projected;
}

void copyKvCacheBlocksToModelInput(GptModelInputs&                                    model_input,
                                   const BatchKVCacheResource&                        kv_cache,
                                   int                                                stream_batch_idx,
                                   int                                                model_batch_idx,
                                   const std::unordered_map<std::string, CacheGroup>& cache_groups) {
    const auto& plan = model_input.kv_cache_block_table_plan;
    if (plan.groupCount() == 0 || plan.poolNumel() == 0) {
        return;
    }
    RTP_LLM_CHECK_WITH_INFO(model_input.kv_cache_kernel_block_id.dim() == 1,
                            "packed kv_cache_kernel_block_id must be flat");
    RTP_LLM_CHECK_WITH_INFO(model_input.kv_cache_block_id.dim() == 1, "packed kv_cache_block_id must be flat");
    RTP_LLM_CHECK_WITH_INFO(static_cast<size_t>(kv_cache.groupNums()) == plan.groupCount(),
                            "request cache resource group count=%d does not match cache tag count=%zu",
                            kv_cache.groupNums(),
                            plan.groupCount());
    RTP_LLM_CHECK_WITH_INFO(model_batch_idx >= 0 && static_cast<size_t>(model_batch_idx) < plan.batchCapacity(),
                            "model batch index %d exceeds packed table capacity %zu",
                            model_batch_idx,
                            plan.batchCapacity());

    int32_t* kernel_dst_base = model_input.kv_cache_kernel_block_id.data_ptr<int32_t>();
    int32_t* store_dst_base  = model_input.kv_cache_block_id.data_ptr<int32_t>();

    for (uint32_t group_ordinal = 0; group_ordinal < plan.groupCount(); ++group_ordinal) {
        const auto& region   = plan.group(group_ordinal);
        const auto& tag      = region.tag;
        const auto  group_it = cache_groups.find(tag);
        RTP_LLM_CHECK_WITH_INFO(group_it != cache_groups.end(), "model input cache config missing tag=%s", tag.c_str());
        const auto& binding       = kv_cache.blockBinding(stream_batch_idx, tag);
        auto        kernel_blocks = projectKernelBlocks(binding, group_it->second);
        RTP_LLM_CHECK_WITH_INFO(kernel_blocks.size() <= region.kernel.row_width,
                                "kernel block table overflow for tag=%s: blocks=%zu capacity=%zu",
                                tag.c_str(),
                                kernel_blocks.size(),
                                region.kernel.row_width);
        RTP_LLM_CHECK_WITH_INFO(binding.size() <= region.pool.row_width,
                                "physical block table overflow for tag=%s: blocks=%zu capacity=%zu",
                                tag.c_str(),
                                binding.size(),
                                region.pool.row_width);

        int32_t* const kernel_dst =
            kernel_dst_base + region.kernel.offset + static_cast<size_t>(model_batch_idx) * region.kernel.row_width;
        std::memcpy(kernel_dst, kernel_blocks.data(), kernel_blocks.size() * sizeof(int32_t));

        int32_t* const store_dst =
            store_dst_base + region.pool.offset + static_cast<size_t>(model_batch_idx) * region.pool.row_width;
        for (size_t group_block_position = 0; group_block_position < binding.size(); ++group_block_position) {
            const auto pool_block_id        = binding.lookup(GroupBlockPosition{group_block_position});
            store_dst[group_block_position] = pool_block_id.has_value() ? pool_block_id->value : NULL_BLOCK_IDX;
        }
        model_input.kv_cache_pool_valid_lengths[group_ordinal].data_ptr<int32_t>()[model_batch_idx] =
            static_cast<int32_t>(binding.size());
        model_input.kv_cache_kernel_valid_lengths[group_ordinal].data_ptr<int32_t>()[model_batch_idx] =
            static_cast<int32_t>(kernel_blocks.size());
    }
}

void gatherMultimodalInputsForContextBatch(const GenerateStreamPtr&    stream,
                                           GatherModelInputContext&    ctx,
                                           std::vector<torch::Tensor>& gathered_mm_features,
                                           std::vector<torch::Tensor>& gathered_mm_extra_input,
                                           TensorHolder&               host_holder) {
    if (!ctx.has_multimodal_input) {
        return;
    }
    std::vector<torch::Tensor> mm_features = stream->multimodalFeatures();
    torch::Tensor              mm_locs     = stream->multimodalLocations();
    if (!mm_locs.defined()) {
        return;
    }
    auto mm_extra_input = stream->multimodalExtraInput();
    RTP_LLM_CHECK_WITH_INFO(mm_locs.numel() == static_cast<int64_t>(mm_features.size()),
                            "mm_locs count %ld != mm_features count %zu for stream %ld",
                            mm_locs.numel(),
                            mm_features.size(),
                            stream->streamId());
    RTP_LLM_CHECK_WITH_INFO(mm_extra_input.empty() || mm_extra_input.size() == mm_features.size(),
                            "mm_extra_input count %zu != mm_features count %zu for stream %ld",
                            mm_extra_input.size(),
                            mm_features.size(),
                            stream->streamId());

    auto*     mm_locs_data = mm_locs.data_ptr<int32_t>();
    const int reuse_length = stream->reuseLength();
    if (mm_locs.numel() > 1) {
        RTP_LLM_CHECK_WITH_INFO(std::is_sorted(mm_locs_data, mm_locs_data + mm_locs.numel()),
                                "mm_locs must be sorted in ascending order for reuse handling");
    }
    for (int i = 0; i < static_cast<int>(mm_features.size()); ++i) {
        const auto&   mm_feature  = mm_features[i];
        const int64_t feature_len = mm_feature.size(0);
        const int64_t feature_loc = mm_locs_data[i];
        const int64_t feature_end = feature_loc + feature_len;
        if (reuse_length >= feature_end) {
            continue;
        }

        // ViT still runs on and caches the complete image. Only the rows already
        // represented by the reused KV prefix are omitted from this model input.
        // This gatherer owns prefix slicing; downstream injectors receive sliced
        // features with non-negative local locations only.
        const int64_t token_offset    = std::max<int64_t>(reuse_length - feature_loc, 0);
        auto          current_feature = mm_feature.slice(0, token_offset, feature_len).contiguous();
        if (!current_feature.is_cuda()) {
            host_holder.hold_host(current_feature);
            gathered_mm_features.emplace_back(current_feature.to(torch::kCUDA, /*non_blocking=*/true));
        } else {
            gathered_mm_features.emplace_back(std::move(current_feature));
        }

        ctx.mm_features_locs[ctx.mm_feature_index] =
            ctx.token_idx + static_cast<int>(std::max<int64_t>(feature_loc - reuse_length, 0));
        ctx.mm_feature_index++;

        if (!mm_extra_input.empty()) {
            auto current_extra_input =
                sliceMultimodalExtraInput(mm_extra_input[i], mm_feature, token_offset, feature_len);
            if (!current_extra_input.is_cuda()) {
                host_holder.hold_host(current_extra_input);
                gathered_mm_extra_input.emplace_back(current_extra_input.to(torch::kCUDA, /*non_blocking=*/true));
            } else {
                gathered_mm_extra_input.emplace_back(std::move(current_extra_input));
            }
        }
    }
    auto text_token_mask = stream->textTokensMask();
    memcpy(ctx.merged_text_mask + ctx.token_idx, text_token_mask.data(), text_token_mask.size() * sizeof(int));
}

// Boundary adapter: the first kv_cache_update_mapping column is an
// adapter-local group_ordinal in canonical sorted-tag order. NormalExecutor
// decodes it back to tags before touching the cache layer.
void addCacheUpdateCopy(GatherModelInputContext&              ctx,
                        const std::vector<TaggedBlockIdPair>& update_mapping,
                        const std::vector<std::string>&       boundary_group_tags) {
    if (!ctx.kv_cache_update_mapping) {
        return;
    }
    for (const auto& mapping : update_mapping) {
        const auto group_ordinal = groupOrdinalForTag(boundary_group_tags, mapping.tag, "cache update mapping");
        *ctx.kv_cache_update_mapping++ =
            GroupOrdinalBlockIdPair{static_cast<int32_t>(group_ordinal), mapping.src, mapping.dst};
    }
}

torch::Tensor buildLmOutputIndexesOnCuda(const GptModelInputs& model_input, const StreamGroups& stream_groups) {
    const auto total_batch_size         = static_cast<int64_t>(stream_groups.totalModelBatchSize());
    const auto total_decode_batch_size  = static_cast<int64_t>(stream_groups.totalDecodeBatchSize());
    const auto total_context_batch_size = total_batch_size - total_decode_batch_size;
    auto       cuda_i32                 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);

    if (total_batch_size == 0) {
        return torch::empty({0}, cuda_i32);
    }

    std::vector<torch::Tensor> parts;
    parts.reserve(2);

    if (total_decode_batch_size > 0) {
        parts.push_back(torch::arange(0, total_decode_batch_size, cuda_i32));
    }

    if (total_context_batch_size > 0) {
        auto context_input_lengths =
            model_input.input_lengths
                .narrow(/*dim=*/0, /*start=*/total_decode_batch_size, /*length=*/total_context_batch_size)
                .to(cuda_i32);
        auto context_indexes = context_input_lengths.cumsum(/*dim=*/0).to(torch::kInt32)
                               + static_cast<int64_t>(total_decode_batch_size - 1);
        parts.push_back(context_indexes);
    }

    if (parts.size() == 1) {
        return parts.front().contiguous();
    }
    return torch::cat(parts, /*dim=*/0).contiguous();
}

torch::Tensor buildLmOutputIndexesOnHost(const GptModelInputs& model_input, const StreamGroups& stream_groups) {
    const auto total_batch_size         = static_cast<int64_t>(stream_groups.totalModelBatchSize());
    const auto total_decode_batch_size  = static_cast<int64_t>(stream_groups.totalDecodeBatchSize());
    const auto total_context_batch_size = total_batch_size - total_decode_batch_size;
    auto       indexes = torch::empty({total_batch_size}, torch::TensorOptions(torch::kInt32).pinned_memory(true));
    auto*      dst     = indexes.data_ptr<int32_t>();
    for (int64_t i = 0; i < total_decode_batch_size; ++i) {
        dst[i] = static_cast<int32_t>(i);
    }
    if (total_context_batch_size > 0) {
        auto        input_lengths = model_input.input_lengths.is_cuda() ? model_input.input_lengths.cpu().contiguous() :
                                                                          model_input.input_lengths.contiguous();
        const auto* lengths       = input_lengths.data_ptr<int32_t>();
        int32_t     offset        = static_cast<int32_t>(total_decode_batch_size);
        for (int64_t i = 0; i < total_context_batch_size; ++i) {
            offset += lengths[total_decode_batch_size + i];
            dst[total_decode_batch_size + i] = offset - 1;
        }
    }
    return indexes;
}

torch::Tensor publishInt32ToCuda(const torch::Tensor& tensor, TensorHolder& host_holder) {
    if (!tensor.defined()) {
        return tensor;
    }
    auto cuda_i32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    if (tensor.is_cuda() && tensor.scalar_type() == torch::kInt32) {
        return tensor;
    }
    if (tensor.numel() == 0) {
        return torch::empty(tensor.sizes(), cuda_i32);
    }
    host_holder.hold_host(tensor);
    return tensor.to(cuda_i32, /*non_blocking=*/true);
}

void publishModelInputCoreTensorsToCuda(GptModelInputs& model_input, TensorHolder& host_holder) {
    // TODO(async): stream state is still gathered through CPU pointers above.
    // Publish only device tensors at the model boundary.
    RTP_LLM_PROFILE_SCOPE("normal_engine.model_input_gatherer.publish_core_tensors_to_cuda");
    model_input.combo_tokens     = publishInt32ToCuda(model_input.combo_tokens, host_holder);
    model_input.input_lengths    = publishInt32ToCuda(model_input.input_lengths, host_holder);
    model_input.sequence_lengths = publishInt32ToCuda(model_input.sequence_lengths, host_holder);
    model_input.prefix_lengths   = publishInt32ToCuda(model_input.prefix_lengths, host_holder);
}

void publishPackedBlockTablesToCuda(GptModelInputs& model_input, TensorHolder& host_holder) {
    const auto publish = [&](const torch::Tensor& host, torch::Tensor& device) {
        RTP_LLM_CHECK_WITH_INFO(host.defined() && host.device().is_cpu(),
                                "packed host block table must be defined on CPU");
        RTP_LLM_CHECK_WITH_INFO(device.defined() && device.is_cuda(),
                                "packed device block table must be defined on CUDA");
        RTP_LLM_CHECK_WITH_INFO(host.numel() == device.numel(), "packed host/device block table size mismatch");
        if (host.numel() == 0) {
            return;
        }
        host_holder.hold_host(host);
        device.copy_(host, /*non_blocking=*/true);
    };
    publish(model_input.kv_cache_block_id, model_input.kv_cache_block_id_device);
    publish(model_input.kv_cache_kernel_block_id, model_input.kv_cache_kernel_block_id_device);
}

}  // anonymous namespace

namespace {

std::vector<std::string> buildBoundaryGroupTags(const std::unordered_map<std::string, CacheGroup>& groups) {
    std::vector<std::string> tags;
    tags.reserve(groups.size());
    for (const auto& [tag, group] : groups) {
        (void)group;
        tags.push_back(tag);
    }
    return sortedCacheGroupTags(tags, "model input cache");
}

}  // namespace

NormalModelInputGatherer::NormalModelInputGatherer(const NormalModelInputGathererConfig& config):
    config_(config), boundary_group_tags_(buildBoundaryGroupTags(config.kv_cache_groups)) {}

CacheBlockTablePackingPlan
NormalModelInputGatherer::makeBlockTablePackingPlan(const StreamGroups& stream_groups) const {
    RTP_LLM_CHECK_WITH_INFO(config_.cache_config.has_value(), "cache block-table packing requires CacheConfig");
    std::vector<size_t> pool_row_widths(boundary_group_tags_.size(), 0);
    const auto          measure_stream = [&](const GenerateStreamPtr& stream) {
        const auto& cache = *stream->kvCachePtr();
        for (int batch = 0; batch < stream->currentBatchSize(); ++batch) {
            for (size_t ordinal = 0; ordinal < boundary_group_tags_.size(); ++ordinal) {
                pool_row_widths[ordinal] =
                    std::max(pool_row_widths[ordinal], cache.blockBinding(batch, boundary_group_tags_[ordinal]).size());
            }
        }
    };
    for (const auto& stream : stream_groups.decodeStreams()) {
        measure_stream(stream);
    }
    for (const auto& stream : stream_groups.contextStreams()) {
        measure_stream(stream);
    }
    return CacheBlockTablePackingPlan::create(
        *config_.cache_config, stream_groups.totalModelBatchSize(), pool_row_widths);
}

CacheBlockTablePackingPlan NormalModelInputGatherer::blockTablePackingPlan(const StreamGroups& stream_groups) const {
    return makeBlockTablePackingPlan(stream_groups);
}

GptModelInputs NormalModelInputGatherer::allocateModelInputBuffers(const StreamGroups& stream_groups) const {
    const size_t current_tokens_size      = stream_groups.modelExecuteTokenSize();
    const size_t total_batch_size         = stream_groups.totalModelBatchSize();
    const size_t total_decode_batch_size  = stream_groups.totalDecodeBatchSize();
    const size_t total_context_batch_size = stream_groups.totalContextBatchSize();
    const size_t total_block_copy_num     = stream_groups.totalBlockUpdateCopyNum();
    const size_t max_blocks_num           = stream_groups.curBlocksNum();
    const size_t max_cache_keys_num       = std::max(max_blocks_num, stream_groups.maxCacheKeysNum());
    const size_t multimodal_features_len  = stream_groups.mmFeaturesLen();
    const bool   has_multimodal_input     = config_.is_multimodal && stream_groups.has_multimodal_input();
    const bool   need_cal_position_id =
        (config_.mm_position_ids_style != PositionIdsStyle::DEFAULT) || config_.has_positional_encoding;

    static const auto pinned_i32  = torch::TensorOptions(torch::kInt32).pinned_memory(true);
    static const auto pinned_i64  = torch::TensorOptions(torch::kInt64).pinned_memory(true);
    static const auto pinned_bool = torch::TensorOptions(torch::kBool).pinned_memory(true);

    GptModelInputs model_input;
    model_input.combo_tokens          = torch::empty({(int64_t)current_tokens_size}, pinned_i32);
    model_input.input_lengths         = torch::empty({(int64_t)total_batch_size}, pinned_i32);
    model_input.sequence_lengths      = torch::empty({(int64_t)total_decode_batch_size}, pinned_i32);
    model_input.prefix_lengths        = torch::empty({(int64_t)total_context_batch_size}, pinned_i32);
    model_input.request_id            = torch::empty({(int64_t)total_context_batch_size}, pinned_i64);
    model_input.request_pd_separation = torch::empty({(int64_t)total_context_batch_size}, pinned_bool);

    const auto cuda_i32                         = torch::TensorOptions(torch::kInt32).device(torch::kCUDA);
    model_input.kv_cache_block_id               = torch::empty({0}, pinned_i32);
    model_input.kv_cache_block_id_device        = torch::empty({0}, cuda_i32);
    model_input.kv_cache_kernel_block_id        = torch::empty({0}, pinned_i32);
    model_input.kv_cache_kernel_block_id_device = torch::empty({0}, cuda_i32);

    if (max_blocks_num) {
        RTP_LLM_CHECK_WITH_INFO(config_.cache_config.has_value(), "cache block-table packing requires CacheConfig");
        const int64_t group_num = (int64_t)boundary_group_tags_.size();
        // Every group dimension below is ordered by the packing plan. The plan
        // carries each execution ordinal's tag together with its packed region.
        model_input.kv_cache_block_table_plan = makeBlockTablePackingPlan(stream_groups);
        RTP_LLM_CHECK_WITH_INFO(model_input.kv_cache_block_table_plan.groupCount() == static_cast<size_t>(group_num),
                                "packing plan group count mismatch");
        model_input.kv_cache_block_id = torch::full(
            {static_cast<int64_t>(model_input.kv_cache_block_table_plan.poolNumel())}, NULL_BLOCK_IDX, pinned_i32);
        model_input.kv_cache_kernel_block_id = torch::full(
            {static_cast<int64_t>(model_input.kv_cache_block_table_plan.kernelNumel())}, NULL_BLOCK_IDX, pinned_i32);
        model_input.kv_cache_block_id_device =
            torch::empty({static_cast<int64_t>(model_input.kv_cache_block_table_plan.poolNumel())}, cuda_i32);
        model_input.kv_cache_kernel_block_id_device =
            torch::empty({static_cast<int64_t>(model_input.kv_cache_block_table_plan.kernelNumel())}, cuda_i32);
        model_input.kv_cache_pool_valid_lengths.reserve(group_num);
        model_input.kv_cache_kernel_valid_lengths.reserve(group_num);
        for (int64_t ordinal = 0; ordinal < group_num; ++ordinal) {
            model_input.kv_cache_pool_valid_lengths.push_back(
                torch::zeros({static_cast<int64_t>(total_batch_size)}, pinned_i32));
            model_input.kv_cache_kernel_valid_lengths.push_back(
                torch::zeros({static_cast<int64_t>(total_batch_size)}, pinned_i32));
        }
        model_input.kv_cache_group_types    = torch::empty({group_num}, pinned_i32);
        model_input.kv_cache_update_mapping = torch::empty({(int64_t)total_block_copy_num, 3}, pinned_i32);
        // CP-sharded group block tables can be narrower than the global cache-key
        // namespace. Keep cache_keys independently sized so PD writer and reader
        // derive identical keys from the complete token sequence.
        model_input.cache_keys =
            torch::zeros({(int64_t)total_context_batch_size, (int64_t)max_cache_keys_num}, pinned_i64);
    }

    if (need_cal_position_id) {
        model_input.combo_position_ids =
            torch::empty({(int64_t)(current_tokens_size * config_.position_id_len_factor)}, pinned_i32);
    }
    if (has_multimodal_input) {
        model_input.text_tokens_mask = torch::empty({(int64_t)current_tokens_size}, pinned_i32);
        model_input.mm_features_locs = torch::empty({(int64_t)multimodal_features_len}, pinned_i32);
    }

    model_input.kv_block_stride_bytes     = config_.block_stride_bytes;
    model_input.kv_scale_stride_bytes     = config_.scale_stride_bytes;
    model_input.seq_size_per_block        = config_.seq_size_per_block;
    model_input.kernel_seq_size_per_block = config_.kernel_seq_size_per_block;
    model_input.pd_separation             = config_.role_type == RoleType::PREFILL;
    model_input.warmup                    = config_.warm_up;
    model_input.decode_entrance           = config_.decode_entrance;
    model_input.use_opaque_kv_cache_store = config_.use_opaque_kv_cache_store;
    model_input.is_fake_stream            = stream_groups.isFakeStream();

    return model_input;
}

void NormalModelInputGatherer::initializeKvCacheMetadata(GptModelInputs& model_input) const {
    if (!model_input.kv_cache_group_types.defined()) {
        return;
    }
    // Boundary adapter: kv_cache_group_types is the payload parallel to the block
    // tables, so it is permuted into the same canonical sorted-tag order here.
    auto* dst = model_input.kv_cache_group_types.data_ptr<int32_t>();
    for (size_t idx = 0; idx < boundary_group_tags_.size(); ++idx) {
        const auto& group = config_.kv_cache_groups.at(boundary_group_tags_[idx]);
        dst[idx]          = static_cast<int32_t>(group.policy.group_type);
    }
}

absl::Status NormalModelInputGatherer::processDecodeStreams(GptModelInputs&     model_input,
                                                            const StreamGroups& stream_groups) const {
    RTP_LLM_PROFILE_SCOPE("normal_engine.model_input_gatherer.process_decode_streams");
    auto ctx = createGatherContext(config_, model_input, stream_groups, GatherContextMode::DECODE);

    const char* device_input_env        = std::getenv("RTP_LLM_DEVICE_INPUT");
    bool        use_normal_device_state = device_input_env != nullptr && std::string(device_input_env) == "1"
                                   && stream_groups.totalContextBatchSize() == 0
                                   && stream_groups.totalDecodeBatchSize() > 0 && !ctx.need_cal_position_id;
    if (use_normal_device_state) {
        for (const auto& stream : stream_groups.decodeStreams()) {
            const auto& state = stream->getNormalAsyncDeviceState();
            if (stream->currentBatchSize() != 1 || !state.last_sample_token_gpu.defined()
                || !state.last_sample_token_gpu.is_cuda() || !state.next_seq_len_gpu.defined()
                || !state.next_seq_len_gpu.is_cuda()) {
                use_normal_device_state = false;
                break;
            }
        }
    }
    std::vector<torch::Tensor> normal_combo_tokens_gpu;
    std::vector<torch::Tensor> normal_sequence_lengths_gpu;
    if (use_normal_device_state) {
        normal_combo_tokens_gpu.reserve(stream_groups.totalDecodeBatchSize());
        normal_sequence_lengths_gpu.reserve(stream_groups.totalDecodeBatchSize());
    }

    for (const auto& stream : stream_groups.decodeStreams()) {
        model_input.need_all_logits        = model_input.need_all_logits || stream->calculateLoss();
        model_input.need_all_hidden_states = model_input.need_all_hidden_states || stream->needReturnHiddenStates();
        auto  current_batch_size           = stream->currentBatchSize();
        auto& kv_cache                     = *stream->kvCachePtr();
        RTP_LLM_LOG_DEBUG("decode kv_cache: %s", kv_cache.debugString().c_str());
        RTP_LLM_LOG_DEBUG("decode stream: %s", stream->debugString().c_str());

        for (auto i = 0; i < current_batch_size; ++i) {
            model_input.trace_ids.push_back(stream->traceId());
            if (use_normal_device_state) {
                const auto&             state = stream->getNormalAsyncDeviceState();
                static std::atomic<int> debug_log_budget{200};
                if (asyncDebugEnabled() && stream->hasPendingAsyncBookkeeping()
                    && debug_log_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
                    RTP_LLM_LOG_WARNING("[async-debug] gather decode with pending bookkeeping: stream=%ld pd_sep=%d "
                                        "status=%s cpu_seq=%d state_next_real=%d cur_blocks=%zu batch_idx=%d",
                                        stream->streamId(),
                                        stream->queryPdSep(),
                                        StreamStateToString(stream->getStatus()).c_str(),
                                        stream->seqLength(),
                                        state.next_real_seq_len,
                                        stream->curBlocksNum(),
                                        ctx.batch_idx);
                }
                normal_combo_tokens_gpu.push_back(state.last_sample_token_gpu.reshape({1}));
                normal_sequence_lengths_gpu.push_back((state.next_seq_len_gpu - 1).to(torch::kInt32).reshape({1}));
                ctx.input_lengths[ctx.batch_idx] = stream->inputLength();
            } else {
                auto currentTokens = stream->currentExecuteTokens(i);
                if (currentTokens[0] >= ctx.input_vocab_size) {
                    std::ostringstream error_msg;
                    error_msg << "stream [" << stream->streamId() << "] token_id " << currentTokens[0]
                              << " exceed vocab_size " << ctx.input_vocab_size;
                    return absl::InvalidArgumentError(error_msg.str());
                }
                ctx.merged_tokens[ctx.batch_idx]    = currentTokens[0];
                ctx.input_lengths[ctx.batch_idx]    = stream->inputLength();
                ctx.sequence_lengths[ctx.batch_idx] = stream->seqLength() - 1;
                if (ctx.need_cal_position_id) {
                    stream->generateNextPositionId(ctx.combo_position_ids
                                                   + ctx.batch_idx * config_.position_id_len_factor);
                }
            }
            copyKvCacheBlocksToModelInput(model_input, kv_cache, i, ctx.batch_idx, config_.kv_cache_groups);
            ctx.batch_idx += 1;
        }
        addCacheUpdateCopy(ctx, stream->streamCacheResource().getKVBlockUpdateMapping(), boundary_group_tags_);
        stream->step();
    }

    if (use_normal_device_state) {
        model_input.combo_tokens     = torch::cat(normal_combo_tokens_gpu, 0).to(torch::kInt32);
        model_input.sequence_lengths = torch::cat(normal_sequence_lengths_gpu, 0).to(torch::kInt32);
    }
    return absl::OkStatus();
}

absl::Status NormalModelInputGatherer::processContextStreams(GptModelInputs&     model_input,
                                                             const StreamGroups& stream_groups,
                                                             TensorHolder&       host_holder) const {
    RTP_LLM_PROFILE_SCOPE("normal_engine.model_input_gatherer.process_context_streams");
    std::vector<torch::Tensor> gathered_mm_features;
    std::vector<torch::Tensor> gathered_mm_extra_input;
    const auto                 context_batch_size = static_cast<int64_t>(stream_groups.totalContextBatchSize());
    auto                       prefix_lengths_host =
        torch::empty({context_batch_size}, torch::TensorOptions(torch::kInt32).pinned_memory(true));
    auto ctx                = createGatherContext(config_, model_input, stream_groups, GatherContextMode::CONTEXT);
    ctx.prefix_lengths_host = prefix_lengths_host.data_ptr<int32_t>();

    for (const auto& stream : stream_groups.contextStreams()) {
        model_input.need_all_logits =
            model_input.need_all_logits || stream->calculateLoss() || stream->returnPromptLogits();
        model_input.need_all_hidden_states = model_input.need_all_hidden_states || stream->needReturnHiddenStates();
        auto  current_batch_size           = stream->currentBatchSize();
        auto& kv_cache                     = *stream->kvCachePtr();
        if (config_.enable_detail_log) {
            RTP_LLM_LOG_DEBUG("context kv_cache: %s", kv_cache.debugString().c_str());
            RTP_LLM_LOG_DEBUG("context stream: %s", stream->debugString().c_str());
        } else {
            RTP_LLM_LOG_TRACE("context kv_cache: %s", kv_cache.debugString().c_str());
            RTP_LLM_LOG_TRACE("context stream: %s", stream->debugString().c_str());
        }

        for (auto i = 0; i < current_batch_size; ++i) {
            const auto prefill_batch_idx = ctx.batch_idx - ctx.total_decode_batch_size;
            model_input.trace_ids.push_back(stream->traceId());
            auto input_tokens = stream->currentExecuteTokens(i);
            auto input_masks  = stream->textTokensMask();
            memcpy(ctx.merged_tokens + ctx.token_idx, input_tokens.data(), input_tokens.size() * sizeof(int));

            for (int index = 0; index < (int)input_tokens.size(); ++index) {
                if (input_tokens[index] >= ctx.input_vocab_size
                    && (index >= (int)input_masks.size() || input_masks[index])) {
                    std::ostringstream error_msg;
                    error_msg << "stream [" << stream->streamId() << "] token_id " << input_tokens[index]
                              << " exceed vocab_size " << ctx.input_vocab_size;
                    return absl::InvalidArgumentError(error_msg.str());
                }
            }

            ctx.input_lengths[ctx.batch_idx]           = input_tokens.size();
            ctx.prefix_lengths_host[prefill_batch_idx] = stream->prefixLength();
            gatherMultimodalInputsForContextBatch(
                stream, ctx, gathered_mm_features, gathered_mm_extra_input, host_holder);

            if (ctx.need_cal_position_id) {
                auto context_pos_ids = stream->generateContextPositionIds();
                int  reuse_offset    = stream->reuseLength() * config_.position_id_len_factor;
                memcpy(ctx.combo_position_ids + ctx.token_idx * config_.position_id_len_factor,
                       context_pos_ids.data_ptr<int>() + reuse_offset,
                       (context_pos_ids.numel() - reuse_offset) * sizeof(int));
            }

            copyKvCacheBlocksToModelInput(model_input, kv_cache, i, ctx.batch_idx, config_.kv_cache_groups);

            if (ctx.max_blocks_num && config_.role_type == RoleType::PREFILL && stream->hasCacheKeys()) {
                RTP_LLM_CHECK_WITH_INFO(static_cast<int64_t>(stream->cacheKeys(i).size())
                                            <= model_input.cache_keys.size(1),
                                        "cache_keys overflow: stream keys=%zu tensor width=%ld",
                                        stream->cacheKeys(i).size(),
                                        model_input.cache_keys.size(1));
                std::memcpy(model_input.cache_keys.data_ptr<int64_t>()
                                + prefill_batch_idx * model_input.cache_keys.size(1),
                            stream->cacheKeys(i).data(),
                            stream->cacheKeys(i).size() * sizeof(int64_t));
            }

            *(model_input.request_id.data_ptr<int64_t>() + prefill_batch_idx) = stream->streamId();
            *(reinterpret_cast<bool*>(model_input.request_pd_separation.data_ptr()) + prefill_batch_idx) =
                stream->queryPdSep();

            ctx.batch_idx += 1;
            ctx.token_idx += input_tokens.size();
        }

        addCacheUpdateCopy(ctx, stream->streamCacheResource().getKVBlockUpdateMapping(), boundary_group_tags_);
        stream->step();
    }

    if (config_.is_multimodal && !gathered_mm_features.empty()) {
        model_input.multimodal_features = std::move(gathered_mm_features);
    }
    if (ctx.has_mm_extra_input && gathered_mm_extra_input.size() > 0) {
        model_input.mm_extra_input = std::move(gathered_mm_extra_input);
    }
    // mm_features_locs was over-allocated using raw stream->multimodalFeaturesLength();
    // slice down to the actual count written (post-reuse) so Python consumers see the
    // correct tensor size.
    if (ctx.has_multimodal_input && model_input.mm_features_locs.defined()
        && ctx.mm_feature_index < model_input.mm_features_locs.numel()) {
        model_input.mm_features_locs = model_input.mm_features_locs.slice(0, 0, ctx.mm_feature_index);
    }
    model_input.prefix_lengths =
        deviceInputEnabled() ? publishInt32ToCuda(prefix_lengths_host, host_holder) : prefix_lengths_host;
    return absl::OkStatus();
}

void NormalModelInputGatherer::gatherKvCacheKernelBlockIdToHost(const StreamGroups&       stream_groups,
                                                                PackedBlockTableSnapshot& snapshot,
                                                                bool                      include_pool_tables) const {
    const size_t total_batch_size = stream_groups.totalModelBatchSize();
    auto&        host_tensor      = snapshot.kernel_host;
    RTP_LLM_CHECK_WITH_INFO(host_tensor.device().is_cpu() && host_tensor.scalar_type() == torch::kInt32
                                && host_tensor.dim() == 1,
                            "kernel block staging tensor must be a flat CPU int32 tensor");
    RTP_LLM_CHECK_WITH_INFO(snapshot.plan.groupCount() == boundary_group_tags_.size()
                                && snapshot.plan.batchCapacity() == total_batch_size,
                            "kernel block packing plan does not match tags/batch");

    std::vector<BlockIndicesType> staged_rows;
    std::vector<BlockIndicesType> staged_pool_rows;
    staged_rows.reserve(total_batch_size * boundary_group_tags_.size());
    staged_pool_rows.reserve(total_batch_size * boundary_group_tags_.size());
    size_t     staged_batch_size = 0;
    const auto stage_one_stream  = [&](const GenerateStreamPtr& stream) {
        const auto& kv_cache           = *stream->kvCachePtr();
        const auto  current_batch_size = stream->currentBatchSize();
        RTP_LLM_CHECK_WITH_INFO(current_batch_size >= 0 && current_batch_size == kv_cache.batchSize(),
                                "stream batch size=%d does not match cache batch size=%d",
                                current_batch_size,
                                kv_cache.batchSize());
        for (int i = 0; i < current_batch_size; ++i) {
            const auto& rows_by_tag = kv_cache.blocksByTag(i);
            RTP_LLM_CHECK_WITH_INFO(rows_by_tag.size() == boundary_group_tags_.size(),
                                    "request cache resource tag count=%zu does not match expected tag count=%zu",
                                    rows_by_tag.size(),
                                    boundary_group_tags_.size());
            for (uint32_t ordinal = 0; ordinal < snapshot.plan.groupCount(); ++ordinal) {
                const auto& tag = snapshot.plan.group(ordinal).tag;
                RTP_LLM_CHECK_WITH_INFO(rows_by_tag.find(tag) != rows_by_tag.end(),
                                        "request cache resource missing expected tag=%s",
                                        tag.c_str());
                const auto group_it = config_.kv_cache_groups.find(tag);
                RTP_LLM_CHECK_WITH_INFO(group_it != config_.kv_cache_groups.end(),
                                        "kernel block refresh config missing tag=%s",
                                        tag.c_str());
                if (include_pool_tables) {
                    BlockIndicesType pool_blocks;
                    const auto       binding_snapshot = kv_cache.blockBinding(i, tag).snapshot();
                    pool_blocks.reserve(binding_snapshot.size());
                    for (const auto& block : binding_snapshot) {
                        pool_blocks.push_back(block.has_value() ? block->value : NULL_BLOCK_IDX);
                    }
                    RTP_LLM_CHECK_WITH_INFO(pool_blocks.size() <= snapshot.plan.group(ordinal).pool.row_width,
                                            "pool block refresh row overflow for tag=%s: blocks=%zu capacity=%zu",
                                            tag.c_str(),
                                            pool_blocks.size(),
                                            snapshot.plan.group(ordinal).pool.row_width);
                    staged_pool_rows.push_back(std::move(pool_blocks));
                }
                auto kernel_blocks = projectKernelBlocks(kv_cache.blockBinding(i, tag), group_it->second);
                RTP_LLM_CHECK_WITH_INFO(kernel_blocks.size() <= snapshot.plan.group(ordinal).kernel.row_width,
                                        "kernel block refresh row overflow for tag=%s: blocks=%zu capacity=%zu",
                                        tag.c_str(),
                                        kernel_blocks.size(),
                                        snapshot.plan.group(ordinal).kernel.row_width);
                staged_rows.push_back(std::move(kernel_blocks));
            }
            ++staged_batch_size;
        }
    };
    for (const auto& stream : stream_groups.decodeStreams()) {
        stage_one_stream(stream);
    }
    for (const auto& stream : stream_groups.contextStreams()) {
        stage_one_stream(stream);
    }
    RTP_LLM_CHECK_WITH_INFO(staged_batch_size == total_batch_size,
                            "staged kernel block batch=%zu does not match expected batch=%zu",
                            staged_batch_size,
                            total_batch_size);
    RTP_LLM_CHECK_WITH_INFO(staged_rows.size() == total_batch_size * boundary_group_tags_.size(),
                            "staged kernel block row count mismatch");
    RTP_LLM_CHECK_WITH_INFO(!include_pool_tables || staged_pool_rows.size() == staged_rows.size(),
                            "staged pool block row count mismatch");

    int32_t* dst_base      = host_tensor.data_ptr<int32_t>();
    int32_t* pool_dst_base = include_pool_tables ? snapshot.pool_host.data_ptr<int32_t>() : nullptr;
    for (uint32_t group_ordinal = 0; group_ordinal < snapshot.plan.groupCount(); ++group_ordinal) {
        const auto& region      = snapshot.plan.group(group_ordinal).kernel;
        const auto& pool_region = snapshot.plan.group(group_ordinal).pool;
        for (size_t batch_idx = 0; batch_idx < total_batch_size; ++batch_idx) {
            const auto& row      = staged_rows[batch_idx * boundary_group_tags_.size() + group_ordinal];
            const auto* pool_row = include_pool_tables ?
                                       &staged_pool_rows[batch_idx * boundary_group_tags_.size() + group_ordinal] :
                                       nullptr;
            int32_t*    dst      = dst_base + region.offset + batch_idx * region.row_width;
            std::memcpy(dst, row.data(), row.size() * sizeof(int32_t));
            if (include_pool_tables) {
                int32_t* pool_dst = pool_dst_base + pool_region.offset + batch_idx * pool_region.row_width;
                std::memcpy(pool_dst, pool_row->data(), pool_row->size() * sizeof(int32_t));
                snapshot.pool_valid_lengths[group_ordinal].data_ptr<int32_t>()[batch_idx] =
                    static_cast<int32_t>(pool_row->size());
            }
            snapshot.kernel_valid_lengths[group_ordinal].data_ptr<int32_t>()[batch_idx] =
                static_cast<int32_t>(row.size());
        }
    }
}

absl::StatusOr<PackedBlockTableSnapshot> NormalModelInputGatherer::gatherKvCacheKernelBlockId(
    const StreamGroups& stream_groups, TensorHolder& host_holder, bool include_pool_tables) const {
    const size_t             total_batch_size = stream_groups.totalModelBatchSize();
    const size_t             max_blocks_num   = stream_groups.curBlocksNum();
    static const auto        pinned_i32       = torch::TensorOptions(torch::kInt32).pinned_memory(true);
    static const auto        cuda_i32         = torch::TensorOptions(torch::kInt32).device(torch::kCUDA);
    PackedBlockTableSnapshot snapshot;
    if (max_blocks_num == 0 || total_batch_size == 0) {
        snapshot.pool_host     = torch::empty({0}, pinned_i32);
        snapshot.pool_device   = torch::empty({0}, cuda_i32);
        snapshot.kernel_host   = torch::empty({0}, pinned_i32);
        snapshot.kernel_device = torch::empty({0}, cuda_i32);
        return snapshot;
    }

    RTP_LLM_CHECK_WITH_INFO(config_.cache_config.has_value(), "kernel block refresh requires CacheConfig");
    snapshot.plan = makeBlockTablePackingPlan(stream_groups);
    snapshot.pool_host =
        include_pool_tables ?
            torch::full({static_cast<int64_t>(snapshot.plan.poolNumel())}, NULL_BLOCK_IDX, pinned_i32) :
            torch::empty({0}, pinned_i32);
    snapshot.pool_device = include_pool_tables ?
                               torch::empty({static_cast<int64_t>(snapshot.plan.poolNumel())}, cuda_i32) :
                               torch::empty({0}, cuda_i32);
    snapshot.kernel_host = torch::full({static_cast<int64_t>(snapshot.plan.kernelNumel())}, NULL_BLOCK_IDX, pinned_i32);
    snapshot.kernel_device = torch::empty({static_cast<int64_t>(snapshot.plan.kernelNumel())}, cuda_i32);
    snapshot.kernel_valid_lengths.reserve(snapshot.plan.groupCount());
    if (include_pool_tables) {
        snapshot.pool_valid_lengths.reserve(snapshot.plan.groupCount());
    }
    for (size_t ordinal = 0; ordinal < snapshot.plan.groupCount(); ++ordinal) {
        if (include_pool_tables) {
            snapshot.pool_valid_lengths.push_back(torch::zeros({static_cast<int64_t>(total_batch_size)}, pinned_i32));
        }
        snapshot.kernel_valid_lengths.push_back(torch::zeros({static_cast<int64_t>(total_batch_size)}, pinned_i32));
    }
    gatherKvCacheKernelBlockIdToHost(stream_groups, snapshot, include_pool_tables);
    if (include_pool_tables) {
        host_holder.hold_host(snapshot.pool_host);
        snapshot.pool_device.copy_(snapshot.pool_host, /*non_blocking=*/true);
    }
    host_holder.hold_host(snapshot.kernel_host);
    snapshot.kernel_device.copy_(snapshot.kernel_host, /*non_blocking=*/true);
    return snapshot;
}

absl::StatusOr<GptModelInputs> NormalModelInputGatherer::gather(const StreamGroups& stream_groups,
                                                                TensorHolder&       host_holder) const {
    RTP_LLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    RTP_LLM_LOG_DEBUG("context_streams size = %d, decode_streams size = %d",
                      stream_groups.contextStreams().size(),
                      stream_groups.decodeStreams().size());
    auto model_input = allocateModelInputBuffers(stream_groups);
    initializeKvCacheMetadata(model_input);
    RETURN_IF_STATUS_ERROR(processDecodeStreams(model_input, stream_groups));
    RETURN_IF_STATUS_ERROR(processContextStreams(model_input, stream_groups, host_holder));
    publishPackedBlockTablesToCuda(model_input, host_holder);
    // No host mirrors are kept for ModelInputsLogger: it snapshots every tensor
    // in place (device-side clone + per-device c10::Event) and only pays the D2H
    // on its own worker thread, so it reads post-publish CUDA members directly.
    if (deviceInputEnabled()) {
        publishModelInputCoreTensorsToCuda(model_input, host_holder);
        model_input.lm_output_indexes = buildLmOutputIndexesOnCuda(model_input, stream_groups);
    } else {
        model_input.lm_output_indexes = buildLmOutputIndexesOnHost(model_input, stream_groups);
    }
    return model_input;
}

}  // namespace rtp_llm
