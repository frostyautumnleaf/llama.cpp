#pragma once

#include "llama.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct ggml_tensor;
struct llama_model;
struct llama_model_loader;

//
// Where the routed experts of a model lie in the file.
//
// A routed expert is a slice of a big 3-D weight tensor: expert e of w starts at
// (uint8_t *) w->data + e * w->nb[2] and is w->nb[2] bytes long. With a memory-mapped load and
// no weight repacking that slice is a window of the file - it can be read whole and dropped
// again without disturbing any other tensor. The pool records, per MoE layer, the weight
// tensors that hold experts and where their data sits; see llama_expert_pool_build.
//

struct llama_expert_pool {
    struct ref {
        const ggml_tensor * w = nullptr; // [n_ff, n_embd, n_expert]
        int      file     = -1;          // index into files
        uint64_t file_off = 0;           // offset of w's data inside that file
    };

    struct layer {
        std::vector<ref> refs;           // gate/up/down (or gate_up/down); empty: dense layer
        size_t expert_bytes = 0;         // one expert of this layer, all refs summed
    };

    bool enabled() const {
        return n_expert > 0 && !layers.empty();
    }

    std::vector<std::string> files;      // model file paths, in loader order
    std::vector<const void *> base;      // where each file is mapped, null when it is not
    std::vector<layer>       layers;     // indexed by layer id, dense layers carry no refs
    int      n_expert   = 0;
    size_t   pool_bytes = 0;             // every routed expert of every layer

    // why this model cannot stream its experts, empty when it can
    std::string blocked;
};

// Record the routed experts of the model. Called at the end of llama_model_base::load_tensors,
// where the file handles and the mappings are still around. It is cheap - a handful of pointers
// per MoE layer - so it runs for every model, and the outcome is only that a context may or may
// not be able to stream.
void llama_expert_pool_build(const llama_model & model, llama_model_loader & ml, llama_expert_pool & pool);

//
// The streaming engine of one context.
//
// llama.cpp leaves the paging of a too-big model to the operating system, which fetches 4 KiB at
// a time with no idea which expert comes next. The engine knows: when the router of a layer has
// chosen its experts, every missing one is read whole and in parallel before the mat-mul runs,
// kept while it pays for its place, and dropped in the background when it goes cold. It moves
// bytes only - the weights are neither copied nor approximated - so the output stays bit-exact.
//

class llama_expert_stream {
public:
    llama_expert_stream(const llama_model & model, const llama_expert_stream_params & params);
    ~llama_expert_stream();

    llama_expert_stream(const llama_expert_stream &) = delete;
    llama_expert_stream & operator=(const llama_expert_stream &) = delete;

    bool enabled() const;

    // the reason streaming is not running, empty when it is
    const std::string & error() const;

    // The router of layer il picked these experts for a ubatch of n_tokens tokens. Every missing
    // one is fetched before the caller is handed back control, so the expert mat-muls that follow
    // read memory that is there instead of faulting it in page by page.
    void on_selected(int il, const int32_t * ids, size_t n_ids, int64_t n_tokens);

    void get_stats(llama_expert_stream_stats * stats) const;

    // One line of statistics through the llama.cpp logger.
    void log_stats() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
