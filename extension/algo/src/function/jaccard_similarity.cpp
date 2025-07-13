#include "binder/binder.h"
#include "binder/expression/expression_util.h"
#include "common/exception/binder.h"
#include "common/string_utils.h"
#include "common/task_system/progress_bar.h"
#include "function/algo_function.h"
#include "function/config/jaccard_config.h"
#include "function/gds/gds.h"
#include "function/gds/gds_utils.h"
#include "function/gds/gds_vertex_compute.h"
#include "processor/execution_context.h"
#include "main/client_context.h"
#include <unordered_map>

using namespace kuzu::processor;
using namespace kuzu::common;
using namespace kuzu::binder;
using namespace kuzu::storage;
using namespace kuzu::graph;
using namespace kuzu::function;

namespace kuzu {
namespace algo_extension {

struct JaccardOptionalParams final : public GDSOptionalParams {
    std::shared_ptr<Expression> threshold;
    std::shared_ptr<Expression> topK;

    explicit JaccardOptionalParams(const expression_vector& optionalParams);

    std::unique_ptr<GDSConfig> getConfig() const override;

    std::unique_ptr<GDSOptionalParams> copy() const override {
        return std::make_unique<JaccardOptionalParams>(*this);
    }
};

JaccardOptionalParams::JaccardOptionalParams(const expression_vector& optionalParams) {
    for (auto& optionalParam : optionalParams) {
        auto paramName = StringUtils::getLower(optionalParam->getAlias());
        if (paramName == SimilarityThreshold::NAME) {
            threshold = optionalParam;
        } else if (paramName == TopK::NAME) {
            topK = optionalParam;
        } else {
            throw BinderException{stringFormat("Unrecognized parameter: {}.", paramName)};
        }
    }
}

std::unique_ptr<GDSConfig> JaccardOptionalParams::getConfig() const {
    auto config = std::make_unique<JaccardConfig>();
    if (threshold != nullptr) {
        config->threshold = ExpressionUtil::getLiteralValue<double>(*threshold);
        SimilarityThreshold::validate(config->threshold);
    }
    if (topK != nullptr) {
        config->topK = ExpressionUtil::evaluateLiteral<int64_t>(*topK,
          LogicalType::INT64(), TopK::validate);
    }
    return config;
}

struct JaccardSimilarityState {
    // Map from node offset to neighbors (per table)
    std::unordered_map<table_id_t, std::vector<std::vector<nodeID_t>>> nodeNeighborsByTable;
    std::vector<std::pair<std::pair<nodeID_t, nodeID_t>, double>> similarities;
    std::mutex resultMutex;

    JaccardSimilarityState() = default;

    std::vector<nodeID_t>& getNeighbors(nodeID_t nodeID) {
        return nodeNeighborsByTable[nodeID.tableID][nodeID.offset];
    }

    const std::vector<nodeID_t>& getNeighbors(nodeID_t nodeID) const {
        return nodeNeighborsByTable.at(nodeID.tableID)[nodeID.offset];
    }
};

class JaccardSimilarityVertexCompute : public GDSResultVertexCompute {
public:
    JaccardSimilarityVertexCompute(storage::MemoryManager* mm, GDSFuncSharedState* sharedState,
                                   JaccardSimilarityState& state, const JaccardConfig& config)
        : GDSResultVertexCompute{mm, sharedState}, state{state}, config{config} {
        node1Vector = createVector(LogicalType::INTERNAL_ID());
        node2Vector = createVector(LogicalType::INTERNAL_ID());
        similarityVector = createVector(LogicalType::DOUBLE());
    }

    void beginOnTableInternal(table_id_t) override {}

    void vertexCompute(offset_t startOffset, offset_t endOffset, table_id_t tableID) override {
        for (auto i = startOffset; i < endOffset; ++i) {
            if (skip(i)) continue;

            auto nodeID = nodeID_t{i, tableID};
            if (state.nodeNeighborsByTable.find(tableID) == state.nodeNeighborsByTable.end() ||
                static_cast<size_t>(i) >= state.nodeNeighborsByTable[tableID].size()) continue;
            auto& nodeANeighbors = state.getNeighbors(nodeID);
            if (nodeANeighbors.empty()) continue;

            std::vector<std::pair<nodeID_t, double>> nodeSimilarities;

            // Compare with all other nodes (upper triangular) - same table only for now
            if (state.nodeNeighborsByTable.find(tableID) != state.nodeNeighborsByTable.end()) {
                for (auto j = i + 1; j < static_cast<offset_t>(state.nodeNeighborsByTable[tableID].size()); j++) {
                    auto otherNodeID = nodeID_t{j, tableID};
                    auto& nodeBNeighbors = state.getNeighbors(otherNodeID);
                    if (nodeBNeighbors.empty()) continue;

                // Calculate Jaccard similarity using sorted neighbor lists
                double jaccard = calculateJaccardSimilarity(nodeANeighbors, nodeBNeighbors);

                    if (jaccard >= config.threshold) {
                        nodeSimilarities.emplace_back(otherNodeID, jaccard);
                    }
                }
            }

            // Apply topK filter if specified
            if (config.topK > 0 && nodeSimilarities.size() > (size_t)config.topK) {
                std::partial_sort(nodeSimilarities.begin(),
                                nodeSimilarities.begin() + config.topK,
                                nodeSimilarities.end(),
                                [](const auto& a, const auto& b) { return a.second > b.second; });
                nodeSimilarities.resize(config.topK);
            }

            // Output results
            for (const auto& sim : nodeSimilarities) {
                node1Vector->setValue<nodeID_t>(0, nodeID);
                node2Vector->setValue<nodeID_t>(0, sim.first);
                similarityVector->setValue<double>(0, sim.second);
                localFT->append(vectors);
            }
        }
    }

    std::unique_ptr<VertexCompute> copy() override {
        return std::make_unique<JaccardSimilarityVertexCompute>(mm, sharedState, state, config);
    }

private:
    double calculateJaccardSimilarity(const std::vector<nodeID_t>& setA,
                                     const std::vector<nodeID_t>& setB) {
        if (setA.empty() && setB.empty()) return 1.0;
        if (setA.empty() || setB.empty()) return 0.0;

        size_t intersection = 0;
        size_t i = 0, j = 0;

        // Count intersection using two pointers on sorted arrays
        while (i < setA.size() && j < setB.size()) {
            if (setA[i] == setB[j]) {
                intersection++;
                i++;
                j++;
            } else if (setA[i] < setB[j]) {
                i++;
            } else {
                j++;
            }
        }

        size_t unionSize = setA.size() + setB.size() - intersection;
        return unionSize > 0 ? static_cast<double>(intersection) / unionSize : 0.0;
    }

    JaccardSimilarityState& state;
    const JaccardConfig& config;
    std::unique_ptr<ValueVector> node1Vector;
    std::unique_ptr<ValueVector> node2Vector;
    std::unique_ptr<ValueVector> similarityVector;
};

struct JaccardSimilarityBindData final : public GDSBindData {
    JaccardSimilarityBindData(expression_vector columns, graph::NativeGraphEntry graphEntry,
        std::shared_ptr<Expression> nodeOutput,
        std::unique_ptr<JaccardOptionalParams> optionalParams)
        : GDSBindData{std::move(columns), std::move(graphEntry), std::move(nodeOutput),
              std::move(optionalParams)} {}

    std::unique_ptr<TableFuncBindData> copy() const override {
        return std::make_unique<JaccardSimilarityBindData>(*this);
    }
};

static std::unique_ptr<TableFuncBindData> bindFunc(main::ClientContext* context,
    const TableFuncBindInput* input) {
    auto graphName = input->getLiteralVal<std::string>(0);
    auto graphEntry = GDSFunction::bindGraphEntry(*context, graphName);
    auto nodeOutput = GDSFunction::bindNodeOutput(*input, graphEntry.getNodeEntries());
    expression_vector columns;
    columns.push_back(input->binder->createVariable("node1", LogicalType::INTERNAL_ID()));
    columns.push_back(input->binder->createVariable("node2", LogicalType::INTERNAL_ID()));
    columns.push_back(input->binder->createVariable("similarity", LogicalType::DOUBLE()));
    return std::make_unique<JaccardSimilarityBindData>(std::move(columns), std::move(graphEntry), nodeOutput,
        std::make_unique<JaccardOptionalParams>(input->optionalParamsLegacy));
}

static offset_t tableFunc(const TableFuncInput& input, TableFuncOutput&) {
    auto clientContext = input.context->clientContext;
    auto transaction = clientContext->getTransaction();
    auto sharedState = input.sharedState->ptrCast<GDSFuncSharedState>();
    auto graph = sharedState->graph.get();
    auto jaccardBindData = input.bindData->constPtrCast<JaccardSimilarityBindData>();
    auto config = jaccardBindData->getConfig()->constCast<JaccardConfig>();

    // Initialize state
    auto state = JaccardSimilarityState{};
    auto maxOffsetMap = graph->getMaxOffsetMap(transaction);

    // Initialize neighbor vectors for each table
    for (auto& [tableID, maxOffset] : maxOffsetMap) {
        state.nodeNeighborsByTable[tableID].resize(maxOffset);
    }

    // Build neighbor lists for all nodes using proper edge scanning

    // Get relationship information for edge scanning
    auto& graphEntry = jaccardBindData->graphEntry;
    auto nodeEntries = graphEntry.getNodeEntries();

    for (auto& [tableID, maxOffset] : maxOffsetMap) {
        // Find the node table entry for this tableID
        const catalog::TableCatalogEntry* nodeTableEntry = nullptr;
        for (auto nodeEntry : nodeEntries) {
            if (nodeEntry->getTableID() == tableID) {
                nodeTableEntry = nodeEntry;
                break;
            }
        }
        if (!nodeTableEntry) continue;

        // Get relationship information for this node table
        auto nbrInfos = graph->getRelInfos(tableID);
        if (nbrInfos.empty()) continue;

        // Use the first relationship type for now (could be extended to handle multiple edge types)
        auto& nbrInfo = nbrInfos[0];
        auto scanState = graph->prepareRelScan(*nbrInfo.relGroupEntry, nbrInfo.relTableID,
            nbrInfo.dstTableID, {});

        for (auto offset = 0u; offset < maxOffset; ++offset) {
            auto nodeID = nodeID_t{offset, tableID};

            // Scan forward edges to get neighbors
            for (auto chunk : graph->scanFwd(nodeID, *scanState)) {
                chunk.forEach([&](auto neighbors, auto, auto i) {
                    auto nbrNodeID = neighbors[i];
                    state.nodeNeighborsByTable[tableID][offset].push_back(nbrNodeID);
                });
            }

            // Sort neighbors for efficient intersection during Jaccard calculation
            std::sort(state.nodeNeighborsByTable[tableID][offset].begin(),
                     state.nodeNeighborsByTable[tableID][offset].end());
        }
    }

    // Run computation
    auto outputVC = std::make_unique<JaccardSimilarityVertexCompute>(
        clientContext->getMemoryManager(), sharedState, state, config);
    GDSUtils::runVertexCompute(input.context, GDSDensityState::DENSE, graph, *outputVC);
    sharedState->factorizedTablePool.mergeLocalTables();
    return 0;
}

function::function_set JaccardSimilarityFunction::getFunctionSet() {
    function_set result;
    auto func = std::make_unique<TableFunction>(name, std::vector<LogicalTypeID>{LogicalTypeID::ANY});
    func->bindFunc = bindFunc;
    func->tableFunc = tableFunc;
    func->initSharedStateFunc = GDSFunction::initSharedState;
    func->initLocalStateFunc = TableFunction::initEmptyLocalState;
    func->canParallelFunc = [] { return false; };
    func->getLogicalPlanFunc = GDSFunction::getLogicalPlan;
    func->getPhysicalPlanFunc = GDSFunction::getPhysicalPlan;
    result.push_back(std::move(func));
    return result;
}

} // namespace algo_extension
} // namespace kuzu
