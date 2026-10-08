//
// mtmd-remote.h - client side of "remote mmproj" vision encoding.
//
// When llama.cpp is started with --mmproj-remote-url <URL>, every image batch that would
// normally be encoded by the local mmproj (clip_image_batch_encode) is instead sent over
// HTTP to a remote host running mmproj-processing-server, which owns the mmproj GGUF,
// encodes the preprocessed patches and returns the resulting embeddings.
//
// The returned embeddings are byte-for-byte the same buffer that clip_image_batch_encode()
// would have produced, so everything downstream (chunk insertion into the LLM context,
// M-RoPE positions, KV cache, batching) is unchanged.
//
// The wire protocol is deliberately tiny and binary (no JSON in the hot path):
//
//   GET  /health        -> 200 "ok"                                        (liveness, text)
//   GET  /v1/info       -> binary info response                            (capability check)
//   POST /v1/encode     -> binary encode request, binary encode response
//
// All integers are little-endian. See mmproj_processing_server/README.md in the fork.
//

#pragma once

#include <cstdint>
#include <cstring>
#include <netdb.h>
#include <sys/socket.h>
#include <sys/types.h>

#include <string>
#include <vector>

struct clip_image_f32_batch;

namespace mtmd_remote {

// protocol / framing ------------------------------------------------------------

static const uint32_t k_magic           = 0x504d524cu; // 'L','R','M','P'
static const uint32_t k_protocol        = 1;
static const int      k_default_port    = 8080;
static const int      k_connect_tmo_ms  = 5000;
static const int      k_io_tmo_ms       = 300000; // a big image batch can take a while

// byte writer, little-endian
struct writer {
    std::vector<char> buf;

    void u8(uint8_t v)   { buf.push_back((char) v); }
    void u32(uint32_t v) { for (int i = 0; i < 4; i++) buf.push_back((char) ((v >> (8*i)) & 0xff)); }
    void u64(uint64_t v) { for (int i = 0; i < 8; i++) buf.push_back((char) ((v >> (8*i)) & 0xff)); }

    void raw(const void * p, size_t n) {
        const char * c = (const char *) p;
        buf.insert(buf.end(), c, c + n);
    }

    void header() {
        u32(k_magic);
        u32(k_protocol);
    }
};

// byte reader with bounds checking, little-endian
struct reader {
    const char * data;
    size_t       size;
    size_t       pos = 0;

    reader(const void * d, size_t n) : data((const char *) d), size(n) {}

    bool left(size_t n) const { return pos + n <= size; }

    bool u8(uint8_t & v) {
        if (!left(1)) return false;
        v = (uint8_t) data[pos++];
        return true;
    }
    bool u32(uint32_t & v) {
        if (!left(4)) return false;
        v = 0;
        for (int i = 0; i < 4; i++) v |= ((uint32_t) (uint8_t) data[pos++]) << (8*i);
        return true;
    }
    bool u64(uint64_t & v) {
        if (!left(8)) return false;
        v = 0;
        for (int i = 0; i < 8; i++) v |= ((uint64_t) (uint8_t) data[pos++]) << (8*i);
        return true;
    }
    bool take(void * dst, size_t n) {
        if (!left(n)) return false;
        memcpy(dst, data + pos, n);
        pos += n;
        return true;
    }
    bool strings(std::string & s) {
        uint64_t n;
        if (!u64(n) || !left((size_t) n)) return false;
        s.assign(data + pos, (size_t) n);
        pos += (size_t) n;
        return true;
    }
};

// URL handling ------------------------------------------------------------------

struct url {
    std::string scheme = "http";
    std::string host;
    int         port   = k_default_port;
    std::string path; // base path without trailing slash ("" when the URL has none)

    // accepts "http://host:8080/base", "host", "host:8080", "http://host/"
    static bool parse(const std::string & s, url & out, std::string & err);
};

// true when a is a loopback / RFC1918 / link-local / ULA address, i.e. usable on a LAN
bool is_lan_address(const struct sockaddr * a);

// resolve host and report whether the resulting addresses are LAN-only
bool host_is_lan(const std::string & host, bool * out_is_lan, std::string & err);

// HTTP/1.1 over plain TCP (no TLS). Returns false on transport failure.
struct response {
    int          status = 0;
    std::string  body;
    std::string  content_type;
};

bool request(const url & u,
             const char * method,
             const std::string & path,
             const std::vector<char> & body,
             const char * content_type,
             int io_tmo_ms,
             response & res,
             std::string & err);

// encode batch <-> wire --------------------------------------------------------

// n_embd_expected is the embedding width the local text model needs; the server refuses
// to answer when its mmproj produces a different width (guards against a wrong mmproj).
// n_floats_expected is the exact number of floats the encode must produce (n_embd *
// n_tokens for this batch): clip_image_batch_encode() only copies into a buffer the caller
// has already sized, so the server has to size it exactly as llama.cpp would.
std::vector<char> build_encode_request(const clip_image_f32_batch & batch, uint32_t n_embd_expected,
                                       uint64_t n_floats_expected);

bool parse_encode_response(const std::string & body, std::vector<float> & out_embd,
                           uint32_t & out_n_embd, std::string & err);

bool parse_info_response(const std::string & body, std::string & out_model,
                         uint32_t & out_n_embd, bool & out_has_vision, std::string & err);

// convenience: one-shot round trip
bool encode(const url & u, const clip_image_f32_batch & batch, uint32_t n_embd_expected,
            uint64_t n_floats_expected, int io_tmo_ms, std::vector<float> & out_embd,
            std::string & err);

} // namespace mtmd_remote
