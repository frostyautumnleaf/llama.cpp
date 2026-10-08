//
// mtmd-remote.cpp - minimal HTTP/1.1 client + binary protocol for remote mmproj encoding.
//
// Deliberately dependency-free (POSIX sockets only) because libmtmd must stay a standalone
// library: it must not link llama-common and must build with LLAMA_BUILD_COMMON=OFF, where
// the vendored cpp-httplib target is not even defined.
//

#include "mtmd-remote.h"
#include "clip.h"
#include "clip-impl.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <cstdlib>

#include <fcntl.h>
#include <netdb.h>
#include <poll.h>
#include <unistd.h>

#ifdef _WIN32
#  error "remote mmproj currently requires POSIX sockets"
#endif

namespace mtmd_remote {

//
// URL
//

static std::string trim(const std::string & s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

bool url::parse(const std::string & s0, url & out, std::string & err) {
    std::string s = trim(s0);
    if (s.empty()) {
        err = "empty URL";
        return false;
    }

    out = url();

    // scheme
    size_t p = s.find("://");
    if (p != std::string::npos) {
        out.scheme = s.substr(0, p);
        s = s.substr(p + 3);
    }
    if (out.scheme != "http") {
        err = string_format("remote mmproj URL must use the http:// scheme (got '%s'); "
                            "this is a LAN-only feature and does not support https", out.scheme.c_str());
        return false;
    }

    // path is split off first
    size_t slash = s.find('/');
    if (slash != std::string::npos) {
        out.path = s.substr(slash);
        s = s.substr(0, slash);
    }
    while (out.path.size() > 1 && out.path.back() == '/') {
        out.path.pop_back(); // normalise, "" means no base path
    }

    // strip a trailing ?query / #fragment if any
    size_t q = out.path.find_first_of("?#");
    if (q != std::string::npos) {
        out.path = out.path.substr(0, q);
    }

    // host[:port] - also tolerate brackets for IPv6
    if (!s.empty() && s[0] == '[') {
        size_t close = s.find(']');
        if (close == std::string::npos) {
            err = "malformed IPv6 host in URL";
            return false;
        }
        out.host = s.substr(1, close - 1);
        s = s.substr(close + 1);
        if (!s.empty() && s[0] == ':') {
            s = s.substr(1);
        }
    } else {
        size_t colon = s.rfind(':');
        if (colon != std::string::npos) {
            out.host = s.substr(0, colon);
            s = s.substr(colon + 1);
        } else {
            out.host = s;
            s = "";
        }
    }

    if (out.host.empty()) {
        err = "URL has no host";
        return false;
    }

    if (!s.empty()) {
        if (s.find_first_not_of("0123456789") != std::string::npos) {
            err = string_format("invalid port '%s' in URL", s.c_str());
            return false;
        }
        long v = strtol(s.c_str(), nullptr, 10);
        if (v <= 0 || v > 65535) {
            err = string_format("port %ld out of range in URL", v);
            return false;
        }
        out.port = (int) v;
    }
    return true;
}

//
// LAN address classification
//

bool is_lan_address(const struct sockaddr * a) {
    if (a == nullptr) {
        return false;
    }
    if (a->sa_family == AF_INET) {
        const struct sockaddr_in * in = (const struct sockaddr_in *) a;
        uint32_t h = ntohl(in->sin_addr.s_addr);
        bool loopback   = (h & 0xff000000u) == 0x7f000000u;  // 127.0.0.0/8
        bool private_a  = (h & 0xff000000u) == 0x0a000000u;  // 10.0.0.0/8
        bool private_b  = (h & 0xfff00000u) == 0xac100000u;  // 172.16.0.0/12
        bool private_c  = (h & 0xffff0000u) == 0xc0a80000u;  // 192.168.0.0/16
        bool link_local = (h & 0xffff0000u) == 0xa9fe0000u;  // 169.254.0.0/16
        bool cgnat      = (h & 0xffc00000u) == 0x64400000u;  // 100.64.0.0/10 (common on tailnets)
        return loopback || private_a || private_b || private_c || link_local || cgnat;
    }
    if (a->sa_family == AF_INET6) {
        const struct sockaddr_in6 * in6 = (const struct sockaddr_in6 *) a;
        const uint8_t * b = in6->sin6_addr.s6_addr;
        if (in6->sin6_addr.s6_addr16[7] == htons(0x0001) &&
            in6->sin6_addr.s6_addr16[6] == 0 &&
            in6->sin6_addr.s6_addr16[5] == 0 &&
            in6->sin6_addr.s6_addr16[4] == 0 &&
            in6->sin6_addr.s6_addr16[3] == 0 &&
            in6->sin6_addr.s6_addr16[2] == 0 &&
            in6->sin6_addr.s6_addr16[1] == 0 &&
            in6->sin6_addr.s6_addr16[0] == 0) {
            return true; // ::1
        }
        bool ula    = (b[0] & 0xfe) == 0xfc;              // fc00::/7
        bool link6  = (b[0] == 0xfe) && ((b[1] & 0xc0) == 0x80); // fe80::/10
        // IPv4-mapped ::ffff:a.b.c.d - classify by the embedded v4 address
        bool mapped = true;
        for (int i = 0; i < 10; i++) {
            if (b[i] != 0) { mapped = false; break; }
        }
        if (mapped && b[10] == 0xff && b[11] == 0xff) {
            struct sockaddr_in in4;
            memset(&in4, 0, sizeof(in4));
            in4.sin_family = AF_INET;
            memcpy(&in4.sin_addr.s_addr, b + 12, 4);
            return is_lan_address((const struct sockaddr *) &in4);
        }
        return ula || link6;
    }
    return false;
}

bool host_is_lan(const std::string & host, bool * out_is_lan, std::string & err) {
    struct addrinfo hints;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family   = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    struct addrinfo * res = nullptr;
    int rc = getaddrinfo(host.c_str(), nullptr, &hints, &res);
    if (rc != 0) {
        err = string_format("failed to resolve '%s': %s", host.c_str(), gai_strerror(rc));
        return false;
    }
    bool all_lan = true;
    bool any     = false;
    for (struct addrinfo * p = res; p != nullptr; p = p->ai_next) {
        any = true;
        if (!is_lan_address(p->ai_addr)) {
            all_lan = false;
        }
    }
    freeaddrinfo(res);
    if (!any) {
        err = string_format("host '%s' resolved to no addresses", host.c_str());
        return false;
    }
    if (out_is_lan) {
        *out_is_lan = all_lan;
    }
    return true;
}

//
// HTTP/1.1 client
//

static bool send_all(int fd, const char * data, size_t n) {
    size_t sent = 0;
    while (sent < n) {
        struct pollfd pfd;
        pfd.fd     = fd;
        pfd.events = POLLOUT;
        int pr = poll(&pfd, 1, k_io_tmo_ms);
        if (pr <= 0) return false;
        ssize_t w = send(fd, data + sent, n - sent, MSG_NOSIGNAL);
        if (w < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        sent += (size_t) w;
    }
    return true;
}

bool request(const url & u,
             const char * method,
             const std::string & path,
             const std::vector<char> & body,
             const char * content_type,
             int io_tmo_ms,
             response & res,
             std::string & err) {
    res = response();

    struct addrinfo hints;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family   = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    char port_str[16];
    snprintf(port_str, sizeof(port_str), "%d", u.port);

    struct addrinfo * ais = nullptr;
    int rc = getaddrinfo(u.host.c_str(), port_str, &hints, &ais);
    if (rc != 0) {
        err = string_format("failed to resolve '%s': %s", u.host.c_str(), gai_strerror(rc));
        return false;
    }

    int fd = -1;
    for (struct addrinfo * p = ais; p != nullptr; p = p->ai_next) {
        fd = socket(p->ai_family, p->ai_socktype, p->ai_protocol);
        if (fd < 0) {
            continue;
        }
        // non-blocking connect so we can honour the connect timeout
        int flags = fcntl(fd, F_GETFL, 0);
        fcntl(fd, F_SETFL, flags | O_NONBLOCK);
        int r = connect(fd, p->ai_addr, p->ai_addrlen);
        if (r == 0) {
            break;
        }
        if (errno == EINPROGRESS) {
            struct pollfd pfd;
            pfd.fd     = fd;
            pfd.events = POLLOUT;
            int pr = poll(&pfd, 1, k_connect_tmo_ms);
            if (pr > 0) {
                int soerr = 0;
                socklen_t len = sizeof(soerr);
                if (getsockopt(fd, SOL_SOCKET, SO_ERROR, &soerr, &len) == 0 && soerr == 0) {
                    break; // connected
                }
            }
        }
        close(fd);
        fd = -1;
    }
    freeaddrinfo(ais);

    if (fd < 0) {
        err = string_format("could not connect to %s://%s:%d (is the mmproj server running and reachable on the LAN?)",
                            u.scheme.c_str(), u.host.c_str(), u.port);
        return false;
    }

    // back to blocking mode with I/O timeouts
    fcntl(fd, F_SETFL, fcntl(fd, F_GETFL, 0) & ~O_NONBLOCK);
    struct timeval tv;
    tv.tv_sec  = io_tmo_ms / 1000;
    tv.tv_usec = (io_tmo_ms % 1000) * 1000;
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

    std::string req;
    req  = method;
    req += " ";
    req += path.empty() ? "/" : path;
    req += " HTTP/1.1\r\n";
    req += "Host: " + u.host + ":" + std::to_string(u.port) + "\r\n";
    req += "User-Agent: llama.cpp-mmproj-client/1\r\n";
    req += "Accept: application/octet-stream, text/plain\r\n";
    req += "Connection: close\r\n";
    if (!body.empty()) {
        req += "Content-Length: " + std::to_string(body.size()) + "\r\n";
        req += "Content-Type: ";
        req += (content_type && *content_type) ? content_type : "application/octet-stream";
        req += "\r\n";
    }
    req += "\r\n";

    if (!send_all(fd, req.c_str(), req.size())) {
        close(fd);
        err = "failed to send HTTP request";
        return false;
    }
    if (!body.empty() && !send_all(fd, body.data(), body.size())) {
        close(fd);
        err = "failed to send HTTP request body";
        return false;
    }

    // read the whole response (server always sets Content-Length and closes)
    std::vector<char> raw;
    for (;;) {
        char tmp[65536];
        ssize_t n = recv(fd, tmp, sizeof(tmp), 0);
        if (n > 0) {
            raw.insert(raw.end(), tmp, tmp + n);
            continue;
        }
        if (n == 0) {
            break; // server closed
        }
        if (errno == EINTR) {
            continue;
        }
        close(fd);
        err = (errno == EAGAIN || errno == EWOULDBLOCK)
                  ? "timed out waiting for the mmproj server response (increase the timeout or check the server)"
                  : std::string("recv() failed: ") + strerror(errno);
        return false;
    }
    close(fd);

    // split headers / body
    size_t hdr_end = 0;
    bool found = false;
    if (raw.size() >= 4) {
        for (size_t i = 0; i + 4 <= raw.size(); i++) {
            if (raw[i] == '\r' && raw[i+1] == '\n' && raw[i+2] == '\r' && raw[i+3] == '\n') {
                hdr_end = i + 4;
                found = true;
                break;
            }
        }
    }
    if (!found) {
        err = "malformed HTTP response from the mmproj server (no header terminator)";
        return false;
    }

    std::string head(raw.data(), hdr_end - 4);
    std::vector<char> body_buf(raw.begin() + hdr_end, raw.end());

    // status line
    size_t sp1 = head.find(' ');
    size_t sp2 = head.find(' ', sp1 == std::string::npos ? 0 : sp1 + 1);
    if (sp1 == std::string::npos) {
        err = "malformed HTTP status line from the mmproj server";
        return false;
    }
    size_t st_end = (sp2 == std::string::npos) ? head.size() : sp2;
    res.status = atoi(head.substr(sp1 + 1, st_end - (sp1 + 1)).c_str());

    // headers we care about
    size_t status_end = head.find("\r\n");
    std::string headers = (status_end == std::string::npos) ? "" : head.substr(status_end + 2);
    size_t cl = std::string::npos;
    size_t cl_len = 0;
    bool chunked = false;
    {
        std::string lh = headers;
        std::transform(lh.begin(), lh.end(), lh.begin(), ::tolower);
        size_t k = lh.find("content-length:");
        if (k != std::string::npos) {
            cl = k + 15;
            cl_len = strtoul(lh.substr(cl, lh.find("\r\n", cl) - cl).c_str(), nullptr, 10);
        }
        chunked = lh.find("transfer-encoding: chunked") != std::string::npos;
        size_t k2 = lh.find("content-type:");
        if (k2 != std::string::npos) {
            size_t vs = k2 + 13;
            size_t ve = headers.find("\r\n", vs);
            res.content_type = headers.substr(vs, (ve == std::string::npos ? headers.size() : ve) - vs);
        }
    }
    if (chunked) {
        err = "the mmproj server used chunked transfer encoding, which is not supported";
        return false;
    }
    if (cl != std::string::npos && body_buf.size() < cl_len) {
        err = string_format("truncated HTTP response from the mmproj server (%zu of %zu body bytes)",
                            body_buf.size(), cl_len);
        return false;
    }
    if (cl != std::string::npos && body_buf.size() > cl_len) {
        body_buf.resize(cl_len);
    }

    res.body.assign(body_buf.begin(), body_buf.end());
    return true;
}

//
// wire format
//

std::vector<char> build_encode_request(const clip_image_f32_batch & batch, uint32_t n_embd_expected,
                                       uint64_t n_floats_expected) {
    writer w;
    w.header();
    w.u32(batch.is_audio ? 1u : 0u);
    w.u32(n_embd_expected);
    w.u32((uint32_t) batch.entries.size());
    w.u64(n_floats_expected);

    for (const auto & e : batch.entries) {
        uint32_t flags = 0;
        if (e.is_placeholder()) flags |= 1u;
        if (e.add_viewsep)      flags |= 2u;
        if (e.add_newline)      flags |= 4u;

        w.u32(e.nx());
        w.u32(e.ny());
        w.u32(flags);
        w.u32((uint32_t) e.lead_pad);
        w.u32((uint32_t) e.anyres.grid_x);
        w.u32((uint32_t) e.anyres.grid_y);
        w.u32((uint32_t) e.anyres.orig_nx);
        w.u32((uint32_t) e.anyres.orig_ny);

        if (e.is_placeholder()) {
            w.u64(0);
        } else {
            const auto & buf = e.get_ro_buf();
            w.u64((uint64_t) buf.size());
            w.raw(buf.data(), buf.size() * sizeof(float));
        }
    }
    return std::move(w.buf);
}

bool parse_encode_response(const std::string & body, std::vector<float> & out_embd,
                           uint32_t & out_n_embd, std::string & err) {
    reader r(body.data(), body.size());
    uint32_t magic, ver;
    if (!r.u32(magic) || !r.u32(ver)) {
        err = "short response from the mmproj server";
        return false;
    }
    if (magic != k_magic) {
        err = "unexpected magic from the mmproj server (is this URL really an mmproj-processing-server?)";
        return false;
    }
    if (ver != k_protocol) {
        err = string_format("mmproj server speaks protocol v%u, this llama.cpp expects v%u", ver, k_protocol);
        return false;
    }
    uint32_t n_embd;
    uint64_t n_floats;
    if (!r.u32(n_embd) || !r.u64(n_floats)) {
        err = "short encode response header from the mmproj server";
        return false;
    }
    if (n_floats > (uint64_t) (r.size - r.pos) / sizeof(float)) {
        err = "truncated embedding data from the mmproj server";
        return false;
    }
    out_embd.clear();
    out_embd.resize((size_t) n_floats);
    if (!r.take(out_embd.data(), (size_t) n_floats * sizeof(float))) {
        err = "failed to read embedding data from the mmproj server";
        return false;
    }
    out_n_embd = n_embd;
    return true;
}

bool parse_info_response(const std::string & body, std::string & out_model,
                         uint32_t & out_n_embd, bool & out_has_vision, std::string & err) {
    reader r(body.data(), body.size());
    uint32_t magic, ver, has_vision, n_embd;
    std::string proj;
    if (!r.u32(magic) || !r.u32(ver) || !r.u32(has_vision) || !r.u32(n_embd) || !r.strings(proj)) {
        err = "short info response from the mmproj server";
        return false;
    }
    if (magic != k_magic) {
        err = "unexpected magic from the mmproj server (is this URL really an mmproj-processing-server?)";
        return false;
    }
    if (ver != k_protocol) {
        err = string_format("mmproj server speaks protocol v%u, this llama.cpp expects v%u", ver, k_protocol);
        return false;
    }
    out_model      = proj;
    out_n_embd     = n_embd;
    out_has_vision = has_vision != 0;
    return true;
}

bool encode(const url & u, const clip_image_f32_batch & batch, uint32_t n_embd_expected,
            uint64_t n_floats_expected, int io_tmo_ms, std::vector<float> & out_embd, std::string & err) {
    auto req = build_encode_request(batch, n_embd_expected, n_floats_expected);
    response res;
    if (!request(u, "POST", u.path + "/v1/encode", req, "application/octet-stream", io_tmo_ms, res, err)) {
        return false;
    }
    if (res.status != 200) {
        std::string msg = res.body;
        while (!msg.empty() && (msg.back() == '\n' || msg.back() == '\r')) msg.pop_back();
        err = string_format("mmproj server returned HTTP %d: %s", res.status, msg.c_str());
        return false;
    }
    uint32_t n_embd = 0;
    return parse_encode_response(res.body, out_embd, n_embd, err);
}

} // namespace mtmd_remote
