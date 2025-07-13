#pragma once

#include <string>

#include "common/exception/binder.h"
#include "common/types/types.h"
#include "function/gds/gds.h"

namespace kuzu {
namespace function {

struct SimilarityThreshold {
    // Minimum similarity threshold to include in results. Must be in [0, 1].
    static constexpr const char* NAME = "threshold";
    static constexpr common::LogicalTypeID TYPE = common::LogicalTypeID::DOUBLE;
    static constexpr double DEFAULT_VALUE = 0.0;

    static void validate(double threshold) {
        if (threshold < 0 || threshold > 1) {
            throw common::BinderException{"Threshold must be in the [0, 1]."};
        }
    }
};

struct TopK {
    // Return only top K similar pairs for each node. 0 means return all.
    static constexpr const char* NAME = "topk";
    static constexpr common::LogicalTypeID TYPE = common::LogicalTypeID::INT64;
    static constexpr int64_t DEFAULT_VALUE = 0;

    static void validate(int64_t topk) {
        if (topk < 0) {
            throw common::BinderException{"TopK must be a positive integer."};
        }
    }

};

struct JaccardConfig final : public GDSConfig {
    double threshold = SimilarityThreshold::DEFAULT_VALUE;
    int64_t topK = TopK::DEFAULT_VALUE;

    JaccardConfig() = default;
};

} // namespace function
} // namespace kuzu
